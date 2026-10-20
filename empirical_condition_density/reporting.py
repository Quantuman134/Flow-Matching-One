"""Coordinate figures and single-seed metric tables."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import write_json


def _metric_values(row, threshold):
    values = {key: value for key, value in threshold.items() if key != "epsilon"}
    values["sliced_wasserstein_1"] = row["sliced_wasserstein_1"]
    values["spread_variance_error"] = row["spread"]["mean_absolute_variance_error"]
    if "one_step_reference_rmse" in row:
        values["one_step_reference_rmse"] = row["one_step_reference_rmse"]
    return values


def _csv(path, rows):
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def export_tables(root, rows):
    """Export nested results, one CSV row per threshold, and long-form metric values."""
    write_json(root / "metrics.json", rows)
    detailed = []
    summary = []
    for row in rows:
        for threshold in row["thresholds"]:
            keys = dict(num_conditions=row["num_conditions"], seed=row["seed"],
                        steps=row["steps"], epsilon=threshold["epsilon"])
            values = _metric_values(row, threshold)
            detailed.append(dict(**keys, **values))
            for name, value in values.items():
                summary.append(dict(**keys, metric=name, value=value))
    _csv(root / "metrics.csv", detailed)
    _csv(root / "summary.csv", summary)
    write_json(root / "summary.json", summary)
    return summary


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _indices(count, maximum):
    return np.linspace(0, count - 1, min(count, maximum), dtype=int)


def _scatter_figures(config, targets, folder):
    seed = config.seed
    # Use the same plotted indices and axis limits across every K/S panel.
    fig, axes = plt.subplots(len(config.condition_counts), len(config.evaluation.scatter_steps),
                             figsize=(3 * len(config.evaluation.scatter_steps),
                                      2.7 * len(config.condition_counts)), squeeze=False)
    limit = max(3.5, float(np.abs(targets).max()) * 1.1)
    selection = _indices(len(targets), config.evaluation.max_plot_points)
    for row, count in enumerate(config.condition_counts):
        source = config.output_root / f"K-{count:05d}" / f"seed-{seed}" / "evaluation"
        with np.load(source / "inputs.npz") as inputs:
            colors = inputs["conditions"][selection, 0]
        for column, steps in enumerate(config.evaluation.scatter_steps):
            with np.load(source / f"samples-S-{steps:03d}.npz") as stored:
                samples = stored["samples"]
            limit = max(limit, float(np.abs(samples).max()) * 1.05)
            ax = axes[row, column]
            ax.scatter(targets[selection, 0], targets[selection, 1], s=2, c="lightgray")
            ax.scatter(samples[selection, 0], samples[selection, 1], s=2,
                       c=colors, cmap="viridis", vmin=-1, vmax=1)
            ax.set_title(f"K={count:,} · S={steps}")
    for ax in axes.flat:
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
    fig.suptitle(f"Seed {seed} · gray: targets · color: condition\n"
                 f"Showing {len(selection):,} of {len(targets):,} samples per panel")
    fig.tight_layout()
    _save(fig, folder / f"scatter-seed-{seed}.png")


def _metric_figures(config, summary, folder):
    primary = [row for row in summary if row["epsilon"] == config.evaluation.primary_threshold]
    names = ["geometric_precision", "geometric_recall", "conditional_precision",
             "sliced_wasserstein_1", "spread_variance_error", "one_step_reference_rmse"]
    for axis_name in ("num_conditions", "steps"):
        other = "steps" if axis_name == "num_conditions" else "num_conditions"
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, name in zip(axes.flat, names):
            selection = [r for r in primary if r["metric"] == name]
            for value in sorted({r[other] for r in selection}):
                series = sorted([r for r in selection if r[other] == value], key=lambda r: r[axis_name])
                ax.plot([r[axis_name] for r in series], [r["value"] for r in series],
                        marker=".", label=f"{'S' if other == 'steps' else 'K'}={value}")
            ax.set(title=name.replace("_", " "), xlabel="K" if axis_name == "num_conditions" else "S",
                   xscale="log")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=6)
        fig.suptitle(f"Seed {config.seed} · ε={config.evaluation.primary_threshold}")
        fig.tight_layout()
        _save(fig, folder / f"metrics-vs-{axis_name}.png")


def _training_and_trajectories(config, targets, folder):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for index, count in enumerate(config.condition_counts):
        color = plt.get_cmap("viridis")(index / max(1, len(config.condition_counts) - 1))
        seed = config.seed
        source = config.output_root / f"K-{count:05d}" / f"seed-{seed}"
        # A resumed update supersedes any duplicate record from an interrupted session.
        events = {"train": {}, "validation": {}}
        for line in (source / "metrics.jsonl").read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A process may have been interrupted while writing a line.
            if row["event"] in events:
                events[row["event"]][row["step"]] = row
        for ax, event, key in zip(axes, ("train", "validation"), ("loss", "velocity_mse")):
            rows = sorted(events[event].values(), key=lambda r: r["step"])
            ax.plot([r["step"] for r in rows], [r[key] for r in rows],
                    color=color, alpha=0.6, linewidth=0.7, label=f"K={count}, seed={seed}")
        with np.load(source / "evaluation" / "trajectories.npz") as stored:
            paths, conditions = stored["positions"], stored["conditions"][:, 0]
            steps = int(stored["steps"])
        trajectory_fig, ax = plt.subplots(figsize=(6, 6))
        selection = _indices(len(targets), config.evaluation.max_plot_points)
        ax.scatter(targets[selection, 0], targets[selection, 1], s=2, c="lightgray")
        for path_index, condition in enumerate(conditions):
            path = paths[:, path_index]
            path_color = plt.get_cmap("viridis")((condition + 1) / 2)
            ax.plot(path[:, 0], path[:, 1], color=path_color, linewidth=0.8)
            marks = np.arange(0, steps, max(1, steps // 8))
            displacement = path[marks + 1] - path[marks]
            ax.quiver(path[marks, 0], path[marks, 1], displacement[:, 0], displacement[:, 1],
                      angles="xy", scale_units="xy", scale=1, color=path_color, width=0.003)
            ax.scatter(*path[-1], c=[path_color], s=12)
        limit = max(3.5, float(np.abs(paths).max()) * 1.05)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal",
               title=f"K={count:,} · seed={seed} · S={steps}\nArrows: Euler displacement v/S")
        _save(trajectory_fig, folder / f"trajectories-K-{count:05d}-seed-{seed}.png")
    for ax, title in zip(axes, ("Training velocity MSE", "Fixed validation velocity MSE")):
        ax.set(title=title, xlabel="Optimizer update", ylabel="Velocity MSE")
    axes[1].legend(fontsize=5, ncol=3)
    fig.tight_layout()
    _save(fig, folder / "loss-curves.png")


def export_figures(config, targets, summary):
    """Render saved endpoints and trajectories without rerunning the model."""
    folder = config.output_root / "figures"
    folder.mkdir(exist_ok=True)
    _scatter_figures(config, targets, folder)
    _metric_figures(config, summary, folder)
    _training_and_trajectories(config, targets, folder)
