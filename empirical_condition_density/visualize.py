"""Save the prior and ideal one-step group-mean references as a figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .data import ConditionScheme, SpiralData


def save_overview(
    data: SpiralData,
    schemes: list[ConditionScheme],
    prior: torch.Tensor,
    destination: Path,
    max_points: int = 2000,
) -> None:
    """Use all targets for means, with capped display counts for readability."""
    columns = min(4, len(schemes) + 1)
    rows = (len(schemes) + columns) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False
    )
    axes = axes.ravel()
    axes[0].scatter(prior[:, 0], prior[:, 1], s=3, alpha=0.35, color="#195ec8")
    axes[0].set_title("Prior: standard 2D Gaussian")
    prior_limit = prior.abs().max().item() * 1.05
    axes[0].set(xlim=(-prior_limit, prior_limit), ylim=(-prior_limit, prior_limit))

    indices = torch.linspace(
        0, len(data.targets) - 1, min(max_points, len(data.targets))
    ).long()
    curve = data.targets[indices].numpy()
    target_limit = data.targets.abs().max().item() * 1.15
    for axis, scheme in zip(axes[1:], schemes):
        means = scheme.group_means(data.targets)
        selected = torch.linspace(0, len(means) - 1, min(max_points, len(means))).long()
        points = means[selected].numpy()
        axis.plot(curve[:, 0], curve[:, 1], color="#cbd5df", linewidth=1, zorder=0)
        axis.scatter(
            points[:, 0], points[:, 1], c=selected.numpy(), cmap="viridis",
            s=12 if len(points) < 100 else 3,
        )
        axis.set_title(
            f"K={scheme.num_conditions:,} | {scheme.targets_per_condition:,} "
            f"{'target' if scheme.targets_per_condition == 1 else 'targets'}/condition"
        )
        axis.set(xlim=(-target_limit, target_limit), ylim=(-target_limit, target_limit))
    for axis in axes[:len(schemes) + 1]:
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(alpha=0.15)
    for axis in axes[len(schemes) + 1:]:
        axis.set_visible(False)
    figure.suptitle(
        f"Gaussian-to-spiral | targets: {len(data.targets):,}\n"
        "Colored points: ideal one-step group means; gray line: target spiral",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.93))
    figure.text(
        0.5, 0.015,
        "Reference calculation only; no model trained. "
        "Display points are capped; group means use the full dataset.",
        ha="center", fontsize=9,
    )
    figure.savefig(destination, dpi=170)
    plt.close(figure)
