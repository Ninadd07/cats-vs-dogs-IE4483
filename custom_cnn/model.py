"""Conventional CNN: local shared kernels, pooling and a two-logit classifier."""

import torch.nn as nn


def convolution_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CustomCNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        if num_classes < 1 or not 0 <= dropout < 1:
            raise ValueError("Require num_classes >= 1 and 0 <= dropout < 1")
        self.features = nn.Sequential(
            convolution_block(3, 32),
            convolution_block(32, 64),
            convolution_block(64, 128),
            convolution_block(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, num_classes),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, images):
        return self.classifier(self.features(images))


def parameter_counts(model):
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def build_custom_cnn(num_classes=2, dropout=0.3, device=None):
    model = CustomCNN(num_classes=num_classes, dropout=dropout)
    return model.to(device) if device is not None else model
