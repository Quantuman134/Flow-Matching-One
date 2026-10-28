"""Distributed streaming metrics and fixed source/reference/output panels."""

from dataclasses import dataclass
import json
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

from flow_matching.samplers import euler_sample
from .aggregation import MetricAccumulator
from .brightness import apply_exposure
from .metrics import image_errors, latent_mse


@dataclass(frozen=True)
class PreviewPanel:
    pixels: np.ndarray
    caption: str
    relative_path: str


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summarize_timings(local_timings, accelerator, batch_size):
    """Combine device timings and turn measured work into per-image costs."""
    # Wall time follows the slowest rank; per-image costs pool device work/counts.
    names = list(local_timings)
    values = list(local_timings.values())
    timing_tensor = torch.tensor([values], dtype=torch.float64, device=accelerator.device)
    gathered = accelerator.gather(timing_tensor)
    slowest_values = gathered.max(dim=0).values.cpu().tolist()
    combined_values = gathered.sum(dim=0).cpu().tolist()
    slowest = dict(zip(names, slowest_values))
    combined = dict(zip(names, combined_values))

    # Source encoding is shared by every exposure generated from that source.
    source_images = int(combined["source_images"])
    generated_images = int(combined["generated_images"])
    encoding_seconds = combined["source_encoding_seconds"]
    integration_seconds = combined["sampler_seconds"]
    decoding_seconds = combined["decoding_seconds"]
    generation_seconds = encoding_seconds + integration_seconds + decoding_seconds
    latency = {
        "batch_size": batch_size,
        "source_images": source_images, "generated_images": generated_images,
        "source_encoding_ms_per_source": 1000 * encoding_seconds / source_images,
        "integration_ms_per_image": 1000 * integration_seconds / generated_images,
        "decoding_ms_per_image": 1000 * decoding_seconds / generated_images,
        "generation_ms_per_image": 1000 * generation_seconds / generated_images,
    }
    report = {
        "elapsed_seconds": slowest["elapsed_seconds"],
        "sampler_seconds": slowest["sampler_seconds"],
        "reference_seconds": slowest["reference_seconds"],
        "latency": latency,
    }
    return report


@torch.no_grad()
def evaluate(model, vae, dataset, exposures, config, accelerator, *, nfe=1):
    """Load sources, generate each exposure, accumulate errors, then report.

    Each rank owns a disjoint source subset. Collectives run after all local
    batches so ranks can process different numbers of batches safely.
    """
    # Set up this rank's source loader with its own RNG stream.
    indices = range(accelerator.process_index, len(dataset), accelerator.num_processes)
    partition = Subset(dataset, indices)
    loader_rng = torch.Generator().manual_seed(config.seed + 6)
    loader = DataLoader(
        partition, batch_size=config.validation.batch_size, num_workers=config.validation.workers,
        generator=loader_rng, pin_memory=accelerator.device.type == "cuda",
        multiprocessing_context="spawn" if config.validation.workers else None,
    )

    # Initialize metric sums and timing/image counters for this validation event.
    totals = MetricAccumulator(exposures, accelerator.device)
    _synchronize(accelerator.device)
    started = time.perf_counter()
    sampler_seconds = 0.0
    reference_seconds = 0.0
    source_encoding_seconds = 0.0
    decoding_seconds = 0.0
    source_images = 0
    generated_images = 0

    for batch in loader:
        # Encode the clean sources once; reuse their latents across exposures.
        source = batch["image"].to(accelerator.device, non_blocking=True)
        _synchronize(accelerator.device)
        encoding_started = time.perf_counter()
        source_latent = vae.encode(source)
        _synchronize(accelerator.device)
        source_encoding_seconds += time.perf_counter() - encoding_started
        source_images += len(source)

        for index, exposure in enumerate(exposures):
            # Construct the brightness reference and its target latent.
            condition = torch.full((len(source), 1), exposure, device=source.device, dtype=torch.float32)
            _synchronize(accelerator.device)
            reference_started = time.perf_counter()
            reference = apply_exposure(source, condition)
            reference_latent = vae.encode(reference.image)
            _synchronize(accelerator.device)
            reference_seconds += time.perf_counter() - reference_started

            # Integrate the flow from source latent to requested exposure.
            sampler_started = time.perf_counter()
            transported = euler_sample(model, source_latent, condition, nfe, config.training.mixed_precision)
            _synchronize(accelerator.device)
            sampler_seconds += time.perf_counter() - sampler_started

            # Decode the generated latent to RGB.
            decoding_started = time.perf_counter()
            output = vae.decode(transported.samples)
            _synchronize(accelerator.device)
            decoding_seconds += time.perf_counter() - decoding_started
            generated_images += len(source)

            # Compare generated RGB/latent with the reference and accumulate errors.
            values = image_errors(output, reference.image, config.metrics.flat_variance_threshold, reference.saturated)
            values["latent_mse"] = latent_mse(transported.samples, reference_latent)
            totals.update(index, values)

    # Combine timing records and metric sums across GPUs after local work ends.
    _synchronize(accelerator.device)
    elapsed = time.perf_counter() - started
    local_timings = {
        "elapsed_seconds": elapsed, "sampler_seconds": sampler_seconds,
        "reference_seconds": reference_seconds,
        "source_encoding_seconds": source_encoding_seconds,
        "decoding_seconds": decoding_seconds,
        "source_images": source_images, "generated_images": generated_images,
    }
    timing_report = _summarize_timings(local_timings, accelerator, config.validation.batch_size)
    totals.reduce(accelerator)

    # Assemble the report consumed by local logs and W&B.
    report = totals.summary()
    comparisons = report["overall"]["comparisons"]
    comparisons_per_second = comparisons / timing_report["elapsed_seconds"]
    report.update({
        "source_images": len(dataset), "nfe": nfe, "exposures": list(exposures),
        "comparisons_per_second": comparisons_per_second,
    })
    report.update(timing_report)
    return report


@torch.no_grad()
def render_previews(model, vae, datasets, config, device, *, step, nfe):
    """One uint8 panel per fixed case: source | reference | generated image."""
    panels = []
    size = config.validation.batch_size
    for start in range(0, len(datasets.preview_sources), size):
        stop = min(start + size, len(datasets.preview_sources))
        images = []
        conditions = []
        for index in range(start, stop):
            sample = datasets.preview_sources[index]
            images.append(sample["image"])
            conditions.append([datasets.index.previews[index].exposure])
        source = torch.stack(images).to(device)
        condition = torch.tensor(conditions, dtype=torch.float32, device=device)
        reference = apply_exposure(source, condition)
        latent = vae.encode(source)
        transported = euler_sample(model, latent, condition, nfe, config.training.mixed_precision)
        output = vae.decode(transported.samples)
        joined = torch.cat([source, reference.image, output], dim=3)
        pixels = (joined.clamp(0, 1) * 255).round().to(torch.uint8)
        pixels = pixels.permute(0, 2, 3, 1).cpu().numpy()
        for offset, index in enumerate(range(start, stop)):
            case = datasets.index.previews[index]
            caption = f"{case.record.relative_path} | exposure={case.exposure:g} EV | NFE={nfe} | update={step}"
            panel = PreviewPanel(pixels[offset], caption, case.record.relative_path)
            panels.append(panel)
    return panels


def save_validation(run_dir, step, report, panels):
    folder = run_dir / "validation" / f"update-{step:07d}"
    folder.mkdir(parents=True, exist_ok=True)
    preview_records = []
    for index, panel in enumerate(panels):
        filename = f"preview-{index:02d}.png"
        image = Image.fromarray(panel.pixels)
        image.save(folder / filename)
        preview_records.append({"image": filename, "caption": panel.caption})
    report = {**report, "optimizer_update": step, "previews": preview_records}
    path = folder / "metrics.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    latest = {"report": str(path.relative_to(run_dir)), "optimizer_update": step}
    (run_dir / "validation-latest.json").write_text(json.dumps(latest) + "\n")
    return report
