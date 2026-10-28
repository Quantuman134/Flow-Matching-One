"""Fixed-memory FP64 statistics for distributed image/exposure evaluation."""

import math

import torch


BASIC_FIELDS = (
    "rgb_mse", "raw_rgb_mse", "luminance_mse", "latent_mse",
    "signed_mean_luminance_error", "mean_brightness_mse",
    "centered_luminance_mse", "range_violation_fraction",
)
GAIN_FIELDS = (
    "luminance_mse", "mean_brightness_mse", "global_gain_mse",
    "remaining_spatial_mse", "gain", "offset",
)
FLAT_FIELDS = ("luminance_mse", "mean_brightness_mse", "centered_luminance_mse")


def _columns():
    names = list(BASIC_FIELDS)
    for name in GAIN_FIELDS:
        names.append(f"gain_fit/{name}")
    for name in FLAT_FIELDS:
        names.append(f"flat_reference/{name}")
    names.extend(["saturated", "unsaturated"])
    return names


class MetricAccumulator:
    """Store a sum/count pair for each metric and exposure, independent of N."""

    def __init__(self, exposures, device):
        self.exposures = list(exposures)
        self.names = _columns()
        shape = (len(exposures), len(self.names), 2)
        self.totals = torch.zeros(shape, dtype=torch.float64, device=device)

    def _add(self, exposure_index, name, values):
        column = self.names.index(name)
        self.totals[exposure_index, column, 0] += values.double().sum()
        self.totals[exposure_index, column, 1] += values.numel()

    def update(self, exposure_index, values):
        for name in BASIC_FIELDS:
            self._add(exposure_index, name, values[name])
        valid = values["gain_valid"]
        for name in GAIN_FIELDS:
            selected = values[name][valid]
            self._add(exposure_index, f"gain_fit/{name}", selected)
        for name in FLAT_FIELDS:
            selected = values[name][~valid]
            self._add(exposure_index, f"flat_reference/{name}", selected)
        for group in ("saturated", "unsaturated"):
            column = self.names.index(group)
            error_sum = values[f"{group}_squared_error_sum"].sum()
            count = values[f"{group}_value_count"].sum()
            self.totals[exposure_index, column, 0] += error_sum
            self.totals[exposure_index, column, 1] += count

    def reduce(self, accelerator):
        self.totals = accelerator.reduce(self.totals, reduction="sum")

    def _row(self, row):
        result = {}
        for column, name in enumerate(self.names):
            error_sum, count = row[column].tolist()
            mean = error_sum / count if count else None
            result[name] = {"mean": mean, "count": int(count)}
        return result

    def _report(self, row):
        values = self._row(row)
        report = {"comparisons": values["rgb_mse"]["count"]}
        for name in BASIC_FIELDS:
            mean = values[name]["mean"]
            if name in ("rgb_mse", "raw_rgb_mse", "luminance_mse", "latent_mse"):
                name = name.replace("_mse", "_rmse")
                mean = math.sqrt(mean) if mean is not None else None
            report[name] = mean
        for group, fields in (("gain_fit", GAIN_FIELDS), ("flat_reference", FLAT_FIELDS)):
            count = values[f"{group}/luminance_mse"]["count"]
            report[group] = {"count": count}
            for name in fields:
                report[group][name] = values[f"{group}/{name}"]["mean"]
        report["saturation"] = {}
        for group in ("saturated", "unsaturated"):
            mean = values[group]["mean"]
            report["saturation"][group] = {
                "rgb_rmse": math.sqrt(mean) if mean is not None else None,
                "value_count": values[group]["count"],
            }
        return report

    def summary(self):
        totals = self.totals.cpu()
        by_exposure = {}
        for index, exposure in enumerate(self.exposures):
            key = format(exposure, "g")
            by_exposure[key] = self._report(totals[index])
        overall = self._report(totals.sum(dim=0))
        # Primary metrics give each exposure equal weight before taking the root.
        for name in BASIC_FIELDS:
            column = self.names.index(name)
            counts = totals[:, column, 1]
            if torch.any(counts == 0):
                raise ValueError("Every evaluated exposure must have at least one comparison.")
            means = totals[:, column, 0] / counts
            mean = means.mean().item()
            if name in ("rgb_mse", "raw_rgb_mse", "luminance_mse", "latent_mse"):
                name = name.replace("_mse", "_rmse")
                mean = math.sqrt(mean)
            overall[name] = mean
        if "0" in by_exposure:
            overall["identity_rmse"] = by_exposure["0"]["rgb_rmse"]
        return {"overall": overall, "by_exposure": by_exposure}
