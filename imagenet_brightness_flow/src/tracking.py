"""Main-process W&B scalars, fixed previews, and checkpoint-linked run IDs."""

from dataclasses import asdict
import json
import os
import uuid


def flatten_scalars(values, prefix=""):
    flattened = {}
    for name, value in values.items():
        key = f"{prefix}/{name}" if prefix else name
        if isinstance(value, dict):
            nested = flatten_scalars(value, key)
            flattened.update(nested)
        elif type(value) in (int, float):
            flattened[key] = value
    return flattened


class WandbTracker:
    def __init__(self, run, sdk, image_count):
        self.run = run
        self.sdk = sdk
        self.image_count = image_count

    @classmethod
    def start(cls, config, state, run_dir, *, resumed, record_config=True):
        import wandb

        settings = config.wandb
        if settings.mode == "online":
            key = os.environ.get(settings.api_key_env)
            if not key:
                raise RuntimeError(f"Set {settings.api_key_env} in the runtime environment for online W&B logging.")
            wandb.login(key=key, verify=True)
        known_run = state.wandb_run_id is not None
        if not known_run:
            state.wandb_run_id = uuid.uuid4().hex[:12]
        folder = run_dir / "wandb"
        folder.mkdir(parents=True, exist_ok=True)
        config_values = None
        if record_config:
            config_values = asdict(config)
        entity = settings.entity or os.environ.get("WANDB_ENTITY")
        run = wandb.init(
            project=settings.project, entity=entity, group=settings.group,
            name=run_dir.name, id=state.wandb_run_id,
            resume="must" if resumed and known_run and settings.mode == "online" else None,
            mode=settings.mode, dir=str(folder), config=config_values,
        )
        run.define_metric("optimizer_update")
        run.define_metric("processed_images")
        run.define_metric("train/*", step_metric="optimizer_update")
        run.define_metric("validation/*", step_metric="optimizer_update")
        run.define_metric("final/*", step_metric="optimizer_update")
        record = {
            "run_id": run.id, "entity": run.entity, "project": settings.project,
            "group": settings.group, "mode": settings.mode, "url": run.url,
        }
        (run_dir / "wandb-run.json").write_text(json.dumps(record, indent=2) + "\n")
        tracker = cls(run, wandb, settings.images_per_validation)
        return tracker

    def training(self, step, processed_images, values):
        payload = {"optimizer_update": step, "processed_images": processed_images}
        for name, value in values.items():
            if type(value) in (int, float):
                payload[f"train/{name}"] = value
        # W&B's own history step stays monotonic when optimizer updates are replayed.
        self.run.log(payload)

    def validation(self, step, processed_images, report, panels):
        if len(panels) != self.image_count:
            raise ValueError("W&B validation must contain exactly 16 preview panels.")
        payload = flatten_scalars(report, "validation")
        payload["optimizer_update"] = step
        payload["processed_images"] = processed_images
        images = []
        for panel in panels:
            image = self.sdk.Image(panel.pixels, caption=panel.caption)
            images.append(image)
        payload["validation/previews"] = images
        self.run.log(payload)

    def final_evaluation(self, step, processed_images, nfe, report, panels):
        """Keep final metrics and the same 16 cases under a separate NFE key."""
        #region Final preview validation
        if len(panels) != self.image_count:
            raise ValueError("Final evaluation requires exactly 16 preview panels.")
        #endregion
        prefix = f"final/nfe-{nfe}"
        payload = flatten_scalars(report, prefix)
        payload["optimizer_update"] = step
        payload["processed_images"] = processed_images
        images = []
        for panel in panels:
            image = self.sdk.Image(panel.pixels, caption=panel.caption)
            images.append(image)
        payload[f"{prefix}/previews"] = images
        self.run.log(payload)

    def finish(self, exit_code=0):
        self.run.finish(exit_code=exit_code)
