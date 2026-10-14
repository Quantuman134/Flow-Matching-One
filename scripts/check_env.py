"""Check package imports, a training step, and distributed communication."""

import argparse
import importlib
import json
import platform
from importlib.metadata import version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--cpu", action="store_true")
    device.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    packages = {
        "torch": "torch", "torchvision": "torchvision", "accelerate": "accelerate",
        "transformers": "transformers", "diffusers": "diffusers",
        "huggingface-hub": "huggingface_hub", "datasets": "datasets",
        "safetensors": "safetensors", "numpy": "numpy", "scipy": "scipy",
        "matplotlib": "matplotlib", "pillow": "PIL", "PyYAML": "yaml",
        "tqdm": "tqdm", "tensorboard": "tensorboard",
    }
    for module in packages.values():
        importlib.import_module(module)

    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable; check the driver and PyTorch build.")
    accelerator = Accelerator(cpu=args.cpu, mixed_precision="no")
    set_seed(0)
    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)
    inputs = torch.ones(8, 4, device=accelerator.device)
    before = accelerator.unwrap_model(model).weight.detach().clone()
    optimizer.zero_grad()
    loss = model(inputs).square().mean()
    accelerator.backward(loss)
    optimizer.step()
    after = accelerator.unwrap_model(model).weight.detach()
    if not torch.isfinite(loss) or torch.equal(before, after):
        raise RuntimeError("Training step failed to produce a finite loss and parameter update.")

    rank_value = torch.tensor([accelerator.process_index + 1.0], device=accelerator.device)
    total = accelerator.reduce(rank_value, reduction="sum").item()
    expected = accelerator.num_processes * (accelerator.num_processes + 1) / 2
    if total != expected:
        raise RuntimeError(f"Distributed reduction failed: {total} != {expected}.")
    weights = accelerator.gather(after.reshape(1, -1).contiguous())
    if not torch.allclose(weights, weights[0].expand_as(weights)):
        raise RuntimeError("Model parameters differ across processes after the training step.")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(json.dumps({
            "status": "passed", "python": platform.python_version(),
            "device": str(accelerator.device), "processes": accelerator.num_processes,
            "visible_gpu_count": torch.cuda.device_count(), "torch_cuda": torch.version.cuda,
            "packages": {package: version(package) for package in packages},
        }, indent=2))
    accelerator.end_training()


if __name__ == "__main__":
    main()
