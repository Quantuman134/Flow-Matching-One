"""Comparison tables, timestep flags, and figures from final streaming reports."""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .artifacts import file_sha256, read_json, write_json
from .final_evaluation import cell_complete
from .filesystem import resolve_project_path


PRIMARY_METRICS = ("rgb_rmse", "luminance_rmse", "identity_rmse", "latent_rmse")
METRIC_LABELS = {
    "rgb_rmse": "RGB RMSE [0,1]", "luminance_rmse": "Luminance RMSE [0,1]",
    "identity_rmse": "Identity RMSE [0,1]", "latent_rmse": "Latent RMSE [scaled VAE units]",
}


def flatten_statistics(values, prefix=""):
    """Keep unavailable scalar groups as empty CSV cells rather than inventing zeros."""
    result = {}
    for name, value in values.items():
        key = f"{prefix}/{name}" if prefix else name
        if isinstance(value, dict):
            nested = flatten_statistics(value, key)
            result.update(nested)
        elif value is None or type(value) in (int, float):
            result[key] = value
    return result


def _write_csv(path, rows):
    fields = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_reports(root, settings, jobs):
    """Require final EMA cells, full counts, identical datasets, and paired streams."""
    cells = []
    initialization = set()
    streams = set()
    dataset_fingerprints = None
    for path, config in jobs:
        run_dir = resolve_project_path(config.training.output_dir)
        weights_hash = file_sha256(run_dir / "averaged.pt")
        performance = read_json(run_dir / "performance.json")
        initialization.add(performance["initial_model_sha256"])
        streams.add(performance["stream_sha256"])
        for nfe in settings.nfe:
            folder = run_dir / "final-evaluation" / f"nfe-{nfe}"
            report = read_json(folder / "metrics.json")
            protocol = report["protocol"]
            fingerprints = protocol["signature"]["fingerprints"]
            if dataset_fingerprints is None:
                dataset_fingerprints = fingerprints
            expected = config.data.expected_validation_images * len(config.data.evaluation_exposures)
            #region Final report validation
            if protocol["weights_sha256"] != weights_hash or not cell_complete(folder, protocol):
                raise ValueError(f"Final evaluation is incomplete or uses different EMA weights: {folder}")
            if report["training_timesteps"] != config.training.timesteps or report["nfe"] != nfe:
                raise ValueError(f"Final evaluation cell labels differ: {folder}")
            if report["optimizer_update"] != config.training.updates:
                raise ValueError(f"Final evaluation uses a different update budget: {folder}")
            if report["overall"]["comparisons"] != expected:
                raise ValueError(f"Final evaluation has an incomplete source/exposure count: {folder}")
            if report["source_images"] != config.data.expected_validation_images:
                raise ValueError(f"Final evaluation does not cover the complete validation split: {folder}")
            if report["exposures"] != config.data.evaluation_exposures or fingerprints != dataset_fingerprints:
                raise ValueError("Comparison cells must use the same exposures and source indexes.")
            #endregion
            relative = folder.relative_to(root)
            cell = {"folder": str(relative), "report": report}
            cells.append(cell)
    #region Paired comparison validation
    if len(initialization) != 1 or len(streams) != 1:
        raise ValueError("Comparison models must share initialization and consumed data/random streams.")
    #endregion
    paired = {
        "initial_model_sha256": initialization.pop(), "stream_sha256": streams.pop(),
        "dataset_fingerprints": dataset_fingerprints,
    }
    return cells, paired


def _tables(root, settings, cells):
    """Export overall/exposure rows and one matrix per primary error metric."""
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    overall = []
    exposures = []
    lookup = {}
    for cell in cells:
        report = cell["report"]
        timesteps = report["training_timesteps"]
        nfe = report["nfe"]
        lookup[(timesteps, nfe)] = report
        common = {
            "training_timesteps": timesteps, "nfe": nfe,
            "timestep_coverage": report["timestep_coverage"],
            "includes_unsupervised_timesteps": report["includes_unsupervised_timesteps"],
            "optimizer_update": report["optimizer_update"], "source_images": report["source_images"],
            "folder": cell["folder"],
        }
        row = dict(common)
        statistics = flatten_statistics(report["overall"])
        row.update(statistics)
        latency = flatten_statistics(report["latency"], "latency")
        row.update(latency)
        row["comparisons_per_second"] = report["comparisons_per_second"]
        row["peak_memory_bytes"] = report["peak_memory_bytes"]
        overall.append(row)
        for exposure, metrics in report["by_exposure"].items():
            row = dict(common)
            row["exposure_ev"] = float(exposure)
            statistics = flatten_statistics(metrics)
            row.update(statistics)
            exposures.append(row)
    _write_csv(tables / "overall.csv", overall)
    _write_csv(tables / "by-exposure.csv", exposures)
    for metric in PRIMARY_METRICS:
        rows = []
        for timesteps in settings.timesteps:
            row = {"training_timesteps": timesteps}
            for nfe in settings.nfe:
                report = lookup[(timesteps, nfe)]
                row[f"nfe_{nfe}"] = report["overall"][metric]
                row[f"nfe_{nfe}_includes_unsupervised"] = report["includes_unsupervised_timesteps"]
            rows.append(row)
        _write_csv(tables / f"{metric}.csv", rows)
    return lookup


def _save_figure(figure, folder, name):
    figure.tight_layout()
    path = folder / name
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _figures(root, settings, jobs, lookup):
    """Show inference/exposure errors, matched luminance groups, and training curves."""
    folder = root / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    # Compare inference cost and mark points outside the supervised discrete grid.
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    for index, metric in enumerate(PRIMARY_METRICS):
        axis = axes[index]
        for timesteps in settings.timesteps:
            values = []
            for nfe in settings.nfe:
                report = lookup[(timesteps, nfe)]
                value = report["overall"][metric]
                values.append(value)
                if report["includes_unsupervised_timesteps"]:
                    axis.scatter(nfe, value, marker="x", color="black", zorder=5)
            label = str(timesteps)
            axis.plot(settings.nfe, values, marker="o", label=label)
        axis.set_xscale("log", base=2)
        axis.set_xticks(settings.nfe, settings.nfe)
        axis.set_xlabel("Euler evaluations (NFE); × marks unsupervised timesteps")
        axis.set_ylabel(METRIC_LABELS[metric])
        axis.legend(title="Training timesteps")
    _save_figure(figure, folder, "rmse-vs-nfe.png")

    # Keep each model's exposure response together across inference step counts.
    model_count = len(settings.timesteps)
    figure, axes = plt.subplots(1, model_count, figsize=(4 * model_count, 3.5), squeeze=False)
    for index, timesteps in enumerate(settings.timesteps):
        axis = axes[0, index]
        for nfe in settings.nfe:
            report = lookup[(timesteps, nfe)]
            exposures = []
            errors = []
            for exposure, metrics in report["by_exposure"].items():
                exposure_value = float(exposure)
                exposures.append(exposure_value)
                errors.append(metrics["rgb_rmse"])
            axis.plot(exposures, errors, marker="o", label=f"NFE {nfe}")
        axis.set_title(f"Training timesteps: {timesteps}")
        axis.set_xlabel("Exposure (EV)")
        axis.set_ylabel("RGB RMSE [0,1]")
        axis.legend()
    _save_figure(figure, folder, "error-vs-exposure.png")

    # Compare luminance components within matching valid and nearly-flat groups.
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    groups = (
        ("gain_fit", ("mean_brightness_mse", "global_gain_mse", "remaining_spatial_mse")),
        ("flat_reference", ("mean_brightness_mse", "centered_luminance_mse")),
    )
    for axis, (group, components) in zip(axes, groups):
        bottoms = [0.0] * model_count
        labels = []
        first_nfe = settings.nfe[0]
        for timesteps in settings.timesteps:
            values = lookup[(timesteps, first_nfe)]["overall"][group]
            labels.append(f"{timesteps}\ncount={values['count']}")
        for component in components:
            heights = []
            for timesteps in settings.timesteps:
                value = lookup[(timesteps, first_nfe)]["overall"][group][component]
                if value is None:
                    value = float("nan")
                heights.append(value)
            axis.bar(labels, heights, bottom=bottoms, label=component)
            for index, height in enumerate(heights):
                bottoms[index] += height
        axis.set_title(f"{group}; NFE {first_nfe}")
        axis.set_ylabel("Luminance MSE [0,1]²")
        axis.legend(fontsize=7)
    _save_figure(figure, folder, "luminance-components.png")

    # Resume can repeat logged updates; retain the latest event at each update.
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for _, config in jobs:
        run_dir = resolve_project_path(config.training.output_dir)
        metric_path = run_dir / "metrics.jsonl"
        if not metric_path.is_file():
            continue
        training = {}
        validation = {}
        with metric_path.open() as stream:
            for line in stream:
                event = json.loads(line)
                if event["event"] == "train":
                    training[event["step"]] = event["velocity_mse"]
                elif event["event"] == "validation":
                    validation[event["step"]] = event["rgb_rmse"]
        for axis, values in zip(axes, (training, validation)):
            steps = sorted(values)
            measurements = []
            for step in steps:
                measurements.append(values[step])
            label = str(config.training.timesteps)
            axis.plot(steps, measurements, label=label)
            axis.set_xlabel("Optimizer update")
            axis.legend(title="Training timesteps")
    axes[0].set_ylabel("Training velocity MSE")
    axes[1].set_ylabel("Monitor RGB RMSE [0,1]")
    _save_figure(figure, folder, "monitoring-curves.png")


def _markdown(root, settings, cells, lookup):
    """Explain metric scales and present a compact table for each primary metric."""
    first = cells[0]["report"]
    lines = [
        "# Brightness timestep comparison", "",
        f"- Sources: {first['source_images']} validation images; exposures: {first['exposures']} EV.",
        f"- Training: seed {first['protocol']['signature']['seed']}; {first['optimizer_update']} updates; final EMA weights.",
        "- RMSE: typical matching-value error; smaller is better and zero is exact.",
        "- Units: RGB/identity use normalized sRGB; luminance uses normalized linear light; latent uses scaled VAE units.",
        "- †: the Euler trajectory includes timesteps outside the model's supervised discrete grid.",
        "- Continuous training at NFE 1 evaluates the t=0 boundary.", "",
    ]
    for metric in PRIMARY_METRICS:
        lines.extend([f"## {metric}", ""])
        header = "| Training timesteps |"
        separator = "| --- |"
        for nfe in settings.nfe:
            header += f" NFE {nfe} |"
            separator += " --- |"
        lines.extend([header, separator])
        for timesteps in settings.timesteps:
            line = f"| {timesteps} |"
            for nfe in settings.nfe:
                report = lookup[(timesteps, nfe)]
                value = report["overall"][metric]
                suffix = "†" if report["includes_unsupervised_timesteps"] else ""
                line += f" {value:.6g}{suffix} |"
            lines.append(line)
        lines.append("")
    lines.extend(["## Artifacts", "", "- [Overall CSV](tables/overall.csv); [exposure CSV](tables/by-exposure.csv).",
                  "- Figures: inference cost, exposure errors, luminance components, and monitoring curves.", ""])
    text = "\n".join(lines)
    (root / "report.md").write_text(text)


def export_comparison(root, settings, jobs):
    """Rebuild shareable tables/plots after validating every requested final cell."""
    cells, paired = collect_reports(root, settings, jobs)
    lookup = _tables(root, settings, cells)
    _figures(root, settings, jobs, lookup)
    _markdown(root, settings, cells, lookup)
    summary = {"paired": paired, "cells": cells}
    write_json(root / "summary.json", summary)
    return summary
