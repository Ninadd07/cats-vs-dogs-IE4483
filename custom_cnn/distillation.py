"""Frozen DINOv2 + exported sklearn binary probe and temperature-scaled KD."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def distillation_loss(student, teacher, labels, alpha=0.5, temperature=4.0):
    """alpha weights KL(teacher || student); alpha=0 is supervised CE."""
    if not 0 <= alpha <= 1 or not temperature > 0:
        raise ValueError("Require alpha in [0, 1] and positive temperature")
    ce = F.cross_entropy(student, labels)
    kd = F.kl_div(F.log_softmax(student / temperature, dim=1),
                  F.softmax(teacher.detach() / temperature, dim=1),
                  reduction="batchmean") * temperature ** 2
    return (1 - alpha) * ce + alpha * kd, ce, kd


class DinoTeacher(nn.Module):
    def __init__(self, backbone, mean, scale, coef, intercept):
        super().__init__()
        self.backbone = backbone
        for name, value in (("mean", mean), ("scale", scale),
                            ("coef", coef), ("intercept", intercept)):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.requires_grad_(False)
        self.eval()

    def forward(self, images):
        features = self.backbone(images).float()
        score = ((features - self.mean) / self.scale) @ self.coef.T + self.intercept
        # sklearn binary predict_proba is sigmoid(score). [0, score] exactly
        # reproduces it; [-score, score] would incorrectly double confidence.
        return torch.cat((torch.zeros_like(score), score), dim=1)


def load_teacher(path, image_size, device):
    # Lazy import keeps timm optional for ordinary CNN training/inference.
    from dino_vit.common import MODEL, load_backbone

    with np.load(path, allow_pickle=False) as probe:
        if str(probe["model"].item()) != MODEL or int(probe["image_size"]) != image_size:
            raise ValueError("Teacher model/image size differs from the exported probe")
        if not np.array_equal(probe["classes"], [0, 1]):
            raise ValueError("Teacher must use cat=0, dog=1")
        arrays = {key: probe[key].copy() for key in ("mean", "scale", "coef", "intercept")}
    dimension = arrays["mean"].size
    if (arrays["mean"].shape != (dimension,) or arrays["scale"].shape != (dimension,)
            or arrays["coef"].shape != (1, dimension) or arrays["intercept"].shape != (1,)
            or any(not np.isfinite(a).all() for a in arrays.values())
            or (arrays["scale"] <= 0).any()):
        raise ValueError("Invalid binary probe parameters")
    backbone = load_backbone(img_size=image_size)
    if backbone.num_features != dimension:
        raise ValueError("Probe feature dimension differs from backbone")
    return DinoTeacher(backbone, **arrays).to(device)
