import torch
import torch.nn as nn
from torchvision import models


class ResNetBinaryClassifier(nn.Module):
    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        freeze_backbone=False,
        dropout=0.2,
    ):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None

        self.model = models.resnet50(weights=weights)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def build_resnet_model(
    num_classes=2,
    pretrained=True,
    freeze_backbone=False,
    dropout=0.2,
    device=None,
):
    model = ResNetBinaryClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )

    if device is not None:
        model = model.to(device)

    return model