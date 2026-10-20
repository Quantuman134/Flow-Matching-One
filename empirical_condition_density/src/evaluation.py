"""Balanced fresh-noise evaluation and conditional spread."""

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import distributions, version
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import EvaluationConfig
from .data import ConditionScheme, SpiralData
from .metrics import GeometryMetrics, endpoint_rmse
from .sampling import sample_euler


def write_json(path: Path, value):
    """Publish a complete JSON file only after serialization succeeds."""
    temporary_suffix = path.suffix + ".tmp"
    temporary = path.with_suffix(temporary_suffix)
    content = json.dumps(value, indent=2, allow_nan=False)
    content = content + "\n"

    temporary.write_text(content)
    temporary.replace(path)


def save_arrays(path: Path, **arrays):
    """Write coordinate arrays atomically through a temporary file."""
    temporary_suffix = path.suffix + ".tmp"
    temporary = path.with_suffix(temporary_suffix)

    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)

    temporary.replace(path)


def generate(model, noise, conditions, steps, batch_size):
    """Keep full evaluation inputs on CPU and bound GPU working memory."""
    parameters = model.parameters()
    first_parameter = next(parameters)
    device = first_parameter.device
    num_samples = len(noise)
    chunks = []

    for start in range(0, num_samples, batch_size):
        stop = start + batch_size
        batch_noise = noise[start:stop]
        batch_conditions = conditions[start:stop]
        batch_noise = batch_noise.to(device)
        batch_conditions = batch_conditions.to(device)

        result = sample_euler(model, batch_noise, batch_conditions, steps)
        cpu_samples = result.samples.cpu()
        chunks.append(cpu_samples)

    samples = torch.cat(chunks)
    samples_array = samples.numpy()
    return samples_array


def _noise(count, seed):
    normalized_seed = seed % 2**32
    rng = torch.Generator(device="cpu")
    rng.manual_seed(normalized_seed)
    noise = torch.randn(count, 2, generator=rng, dtype=torch.float32, device="cpu")
    return noise


def evaluate_model(
    model, data: SpiralData, scheme: ConditionScheme,
    config: EvaluationConfig, seed: int, run_dir: Path,
    geometry: GeometryMetrics, logger,
):
    """Evaluate one trained model over all Euler step counts; never update its weights.

    Main inputs: N noise/condition pairs, [N,2] and [N,1]. Additional draws
    measure within-condition spread using the same training conditions.
    """
    folder = run_dir / "evaluation"
    folder.mkdir(parents=True, exist_ok=True)
    parameters = model.parameters()
    first_parameter = next(parameters)
    device = first_parameter.device

    # Resolve environment values before assembling the saved metadata.
    started_time = datetime.now(timezone.utc)
    started_utc = started_time.isoformat()
    evaluation_settings = asdict(config)
    device_text = str(device)
    device_name = "CPU"
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)

    torch_version = str(torch.__version__)
    scipy_version = version("scipy")
    run_metadata = {
        "started_utc": started_utc,
        "seed": seed,
        "num_conditions": scheme.num_conditions,
        "evaluation": evaluation_settings,
        "device": device_text,
        "device_name": device_name,
        "evaluation_processes": 1,
        "precision": "fp32",
        "float32_matmul_precision": "highest",
        "torch": torch_version,
        "numpy": np.__version__,
        "scipy": scipy_version,
    }
    metadata_yaml = yaml.safe_dump(run_metadata, sort_keys=False)
    metadata_path = folder / "run.yaml"
    metadata_path.write_text(metadata_yaml)

    installed_packages = distributions()
    package_records = set()
    for package in installed_packages:
        package_name = package.metadata.get("Name")
        if not package_name:
            continue
        package_record = f"{package_name}=={package.version}"
        package_records.add(package_record)

    packages = sorted(package_records)
    environment_text = "\n".join(packages)
    environment_text = environment_text + "\n"
    environment_path = folder / "environment.txt"
    environment_path.write_text(environment_text)

    # Ordered target indices give exactly N/K samples per condition.
    count = len(data.targets)
    indices = torch.arange(count, dtype=torch.int64)
    groups = scheme.group_indices(indices)
    conditions = scheme.encode(indices)

    # Recreate the same held-out noise for each K and reuse it for every step count.
    noise_seed = seed + config.noise_seed_offset
    noise = _noise(count, noise_seed)
    means = scheme.group_means(data.targets)
    reference_tensor = means[groups]
    reference = reference_tensor.numpy()

    noise_array = noise.numpy()
    conditions_array = conditions.numpy()
    groups_array = groups.numpy()
    indices_array = indices.numpy()
    inputs_path = folder / "inputs.npz"
    save_arrays(
        inputs_path,
        noise=noise_array,
        conditions=conditions_array,
        group_ids=groups_array,
        target_indices=indices_array,
    )

    # Extra noise draws reveal whether one condition still produces diverse outputs.
    num_spread_conditions = min(config.spread_conditions, scheme.num_conditions)
    last_group = scheme.num_conditions - 1
    group_positions = np.linspace(
        0, last_group, num_spread_conditions, dtype=np.int64,
    )
    selected = np.unique(group_positions)
    num_selected = len(selected)

    condition_values = 2 * (selected + 0.5) / scheme.num_conditions - 1
    spread_conditions = torch.tensor(condition_values, dtype=torch.float32)
    spread_conditions = spread_conditions.repeat_interleave(config.spread_samples)
    spread_conditions = spread_conditions.unsqueeze(1)
    num_spread_samples = len(spread_conditions)
    spread_seed = noise_seed + 1
    spread_noise = _noise(num_spread_samples, spread_seed)
    spread_conditions_array = spread_conditions.numpy()
    spread_noise_array = spread_noise.numpy()

    # Variance is the mean squared Euclidean distance from the group's mean.
    targets_fp64 = data.targets.double()
    all_target_groups = targets_fp64.reshape(scheme.num_conditions, -1, 2)
    target_groups = all_target_groups[selected]
    target_means = target_groups.mean(dim=1, keepdim=True)
    centered_targets = target_groups - target_means
    squared_targets = centered_targets.square()
    squared_target_distances = squared_targets.sum(dim=2)
    target_variance_tensor = squared_target_distances.mean(dim=1)
    target_variance = target_variance_tensor.numpy()

    rows = []
    started = time.perf_counter()
    progress_path = folder / "progress.jsonl"
    with progress_path.open("a") as progress:
        for index, steps in enumerate(config.steps):
            logger.info(
                "Evaluation | K=%s | seed=%s | Euler steps=%s | samples=%s",
                scheme.num_conditions, seed, steps, count,
            )
            step_started = time.perf_counter()
            samples = generate(model, noise, conditions, steps, config.batch_size)

            # GeometryMetrics owns the distance formulas; this module schedules evaluation.
            metrics = geometry.evaluate(samples, groups_array, scheme, config.thresholds)
            row = {
                "num_conditions": scheme.num_conditions,
                "seed": seed,
                "steps": steps,
                "num_samples": count,
            }
            row.update(metrics)

            if steps == 1:
                reference_error = endpoint_rmse(samples, reference)
                row["one_step_reference_rmse"] = reference_error

            samples_path = folder / f"samples-S-{steps:03d}.npz"
            save_arrays(samples_path, samples=samples)

            spread = generate(model, spread_noise, spread_conditions, steps, config.batch_size)
            spread_fp64 = spread.astype(np.float64)
            grouped_spread = spread_fp64.reshape(num_selected, config.spread_samples, 2)
            spread_mean = grouped_spread.mean(axis=1, keepdims=True)
            centered_spread = grouped_spread - spread_mean
            squared_spread = np.square(centered_spread)
            squared_spread_distances = squared_spread.sum(axis=2)
            generated_variance = squared_spread_distances.mean(axis=1)

            variance_difference = generated_variance - target_variance
            variance_error = np.abs(variance_difference)
            mean_variance_error = variance_error.mean()
            mean_variance_error = float(mean_variance_error)
            selected_groups = selected.tolist()
            generated_variance_values = generated_variance.tolist()
            target_variance_values = target_variance.tolist()
            row["spread"] = {
                "group_ids": selected_groups,
                "draws_per_condition": config.spread_samples,
                "generated_variance": generated_variance_values,
                "target_variance": target_variance_values,
                "mean_absolute_variance_error": mean_variance_error,
            }

            spread_path = folder / f"spread-S-{steps:03d}.npz"
            save_arrays(
                spread_path,
                samples=spread,
                conditions=spread_conditions_array,
                noise=spread_noise_array,
            )

            step_finished = time.perf_counter()
            evaluation_seconds = step_finished - step_started
            row["evaluation_seconds"] = evaluation_seconds
            rows.append(row)
            progress_line = json.dumps(row, allow_nan=False)
            progress_line = progress_line + "\n"
            progress.write(progress_line)
            progress.flush()

            primary = None
            threshold_metrics = row["thresholds"]
            for threshold in threshold_metrics:
                if threshold["epsilon"] == config.primary_threshold:
                    primary = threshold
                    break

            #region Metric validation
            if primary is None:
                raise StopIteration("Primary threshold is missing from evaluation metrics.")
            #endregion

            # Euler work varies by step count; estimate remaining time by evaluations.
            current_time = time.perf_counter()
            elapsed = current_time - started
            completed_steps = config.steps[:index + 1]
            remaining_steps = config.steps[index + 1:]
            work_done = sum(completed_steps)
            work_remaining = sum(remaining_steps)
            eta = elapsed * work_remaining / work_done
            samples_per_second = count / evaluation_seconds

            precision = primary["geometric_precision"]
            recall = primary["geometric_recall"]
            conditional_precision = primary["conditional_precision"]
            wasserstein = row["sliced_wasserstein_1"]
            logger.info(
                "Evaluated | K=%s | seed=%s | S=%s | precision=%.4f | recall=%.4f "
                "| conditional=%.4f | SW1=%.6f | samples/s=%.0f | elapsed=%.1fs | ETA=%.1fs",
                scheme.num_conditions, seed, steps, precision, recall,
                conditional_precision, wasserstein, samples_per_second, elapsed, eta,
            )

    # Keep a small full path for trajectory figures; main metrics use all N endpoints.
    num_trajectories = min(config.trajectory_count, count)
    last_index = count - 1
    selection = torch.linspace(0, last_index, num_trajectories)
    selection = selection.long()
    trajectory_noise = noise[selection]
    trajectory_conditions = conditions[selection]
    trajectory_noise = trajectory_noise.to(device)
    trajectory_conditions = trajectory_conditions.to(device)
    trajectories = sample_euler(
        model, trajectory_noise, trajectory_conditions,
        config.trajectory_steps, return_trajectory=True,
    )

    positions_cpu = trajectories.trajectory.cpu()
    positions_array = positions_cpu.numpy()
    selected_conditions = conditions[selection]
    selected_conditions_array = selected_conditions.numpy()
    selection_array = selection.numpy()
    trajectory_steps = np.asarray(config.trajectory_steps)
    trajectories_path = folder / "trajectories.npz"
    save_arrays(
        trajectories_path,
        positions=positions_array,
        conditions=selected_conditions_array,
        target_indices=selection_array,
        steps=trajectory_steps,
    )

    metrics_path = folder / "metrics.json"
    write_json(metrics_path, rows)
    return rows


def evaluation_complete(folder, config):
    """A metrics marker is written last; also require every coordinate artifact."""
    expected = ["metrics.json", "inputs.npz", "trajectories.npz", "run.yaml", "environment.txt"]

    for steps in config.steps:
        samples_name = f"samples-S-{steps:03d}.npz"
        spread_name = f"spread-S-{steps:03d}.npz"
        expected.append(samples_name)
        expected.append(spread_name)

    for filename in expected:
        path = folder / filename
        exists = path.is_file()
        if not exists:
            return False

    return True
