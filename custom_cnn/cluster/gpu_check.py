"""One real-image CUDA forward/backward pass; no optimizer update or epoch."""

import json
import os
from pathlib import Path
import sys

# This file can be invoked directly by the batch script from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn
from custom_cnn import build_custom_cnn, parameter_counts
from datapreprocessing import CatsDogsDataLoader


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the Slurm GPU allocation")
    torch.manual_seed(42)
    project = Path(os.environ["CNN_PROJECT"])
    data = CatsDogsDataLoader(project / "datasets", batch_size=32, num_workers=2,
                             pin_memory=True, seed=42)
    images, labels = next(iter(data.train_loader))
    assert images.shape == (32, 3, 224, 224)
    device = torch.device("cuda")
    model = build_custom_cnn(device=device)
    torch.cuda.reset_peak_memory_stats()
    logits = model(images.to(device))
    loss = nn.CrossEntropyLoss()(logits, labels.to(device))
    loss.backward()
    torch.cuda.synchronize()
    assert logits.shape == (32, 2) and bool(torch.isfinite(logits).all())
    assert bool(torch.isfinite(loss))
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    result = {"status": "passed", "job_id": os.environ.get("SLURM_JOB_ID"),
              "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
              "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
              "parameters": parameter_counts(model), "batch_shape": list(images.shape),
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
              "scope": "One real training batch, forward/backward only; no parameter update or epoch"}
    (project / "repo/custom_cnn/outputs/setup/gpu_check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
