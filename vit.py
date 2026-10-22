import torch
import torch.nn as nn
from torchvision import models


class ViTBinaryClassifier(nn.Module):
    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        freeze_backbone=False,
        dropout=0.2,
    ):
        super().__init__()

        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None

        self.model = models.vit_b_16(weights=weights)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.heads.head.in_features

        self.model.heads.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def build_vit_model(
    num_classes=2,
    pretrained=True,
    freeze_backbone=False,
    dropout=0.2,
    device=None,
):
    model = ViTBinaryClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )

    if device is not None:
        model = model.to(device)

    return model