"""CPU checks for KD mathematics, probe conversion, frozen teacher and resume.

Run: python -m custom_cnn.verify_distillation (no downloads or real data).
"""

from pathlib import Path
import tempfile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .distillation import DinoTeacher, distillation_loss
from .model import build_custom_cnn
from .train import DEFAULTS, fit, load_checkpoint, seed_everything


def main():
    torch.set_num_threads(2)
    seed_everything(42, deterministic=True)
    x = np.random.default_rng(42).normal(size=(16, 3))
    y = (x[:, 0] > 0).astype(int)
    scaler = StandardScaler().fit(x)
    probe = LogisticRegression().fit(scaler.transform(x), y)
    teacher = DinoTeacher(nn.Identity(), scaler.mean_, scaler.scale_,
                          probe.coef_, probe.intercept_)
    probabilities = teacher(torch.tensor(x, dtype=torch.float32)).softmax(1).numpy()
    np.testing.assert_allclose(probabilities, probe.predict_proba(scaler.transform(x)),
                               atol=1e-6)
    print("PASS: torch teacher probabilities match sklearn scaler + binary probe")

    student = torch.randn(16, 2, requires_grad=True)
    labels = torch.tensor(y)
    target = torch.randn(16, 2, requires_grad=True)
    loss, ce, kd = distillation_loss(student, target, labels, alpha=0, temperature=4)
    torch.testing.assert_close(loss, F.cross_entropy(student, labels))
    loss, ce, kd = distillation_loss(student, target, labels, alpha=1, temperature=4)
    expected = sum(F.kl_div(F.log_softmax(student[i] / 4, dim=0),
                           F.softmax(target[i].detach() / 4, dim=0),
                           reduction="sum") for i in range(16))
    torch.testing.assert_close(loss, expected)  # mean over 16 samples * T^2=16
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert target.grad is None
    same = distillation_loss(target, target, labels)[2]
    torch.testing.assert_close(same, torch.tensor(0.), atol=1e-6, rtol=0)
    print("PASS: CE/KL endpoints, batch reduction, temperature scaling and detached targets")

    # A tiny frozen image teacher avoids downloading DINOv2 in smoke tests.
    backbone = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 3))
    teacher = DinoTeacher(backbone, scaler.mean_, scaler.scale_, probe.coef_, probe.intercept_)
    initial_teacher = {key: value.clone() for key, value in teacher.state_dict().items()}
    images = torch.randn(4, 3, 32, 32)
    dataset = TensorDataset(images, torch.tensor([0, 1, 0, 1]))

    def setup():
        seed_everything(11, deterministic=True)
        model = build_custom_cnn()
        loaders = [DataLoader(dataset, batch_size=2, shuffle=split == 0,
                              generator=torch.Generator().manual_seed(11 + split))
                   for split in range(2)]
        return model, loaders

    config = dict(DEFAULTS, epochs=2, early_stopping_patience=0)
    with tempfile.TemporaryDirectory(prefix="ie4483-kd-") as temporary:
        full, resumed = Path(temporary) / "full", Path(temporary) / "resumed"
        model, loaders = setup()
        fit(model, *loaders, config, full, torch.device("cpu"), teacher=teacher)
        model, loaders = setup()
        fit(model, *loaders, dict(config, epochs=1), resumed, torch.device("cpu"), teacher=teacher)
        model, loaders = setup()
        fit(model, *loaders, config, resumed, torch.device("cpu"),
            resume=resumed / "latest.pth", teacher=teacher)
        a, b = [load_checkpoint(path / "latest.pth") for path in (full, resumed)]
        for key in a["model_state"]:
            torch.testing.assert_close(a["model_state"][key], b["model_state"][key], rtol=0, atol=0)
        assert a["history"][-1]["train_kd"] > 0
        assert "backbone" not in " ".join(a["model_state"])
        for key, value in teacher.state_dict().items():
            torch.testing.assert_close(value, initial_teacher[key], rtol=0, atol=0)
        assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    print("PASS: frozen teacher, student-only checkpoint, KD logging and exact CPU resume")


if __name__ == "__main__":
    main()
