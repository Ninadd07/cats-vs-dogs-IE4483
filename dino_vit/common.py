"""Shared pieces for the DINOv2 ViT pipeline: paths, device, preprocessing, test-set loader, backbone loading."""
from pathlib import Path

import timm
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE.parent / "datasets"
FEATURES_DIR = HERE / "features"
WEIGHTS_DIR = HERE / "weights"
SUBMISSIONS_DIR = HERE / "submissions"

# DINOv2 expects ImageNet statistics.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

MODEL = "vit_base_patch14_dinov2.lvd142m"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transform(img_size=224):
    """Plain resize to a square + ImageNet normalisation. No augmentation: the backbone is frozen and
    features are extracted once, so there is nothing to randomise."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class TestImages(Dataset):
    """Unlabelled test images named <id>.jpg; yields (image, id)."""

    def __init__(self, root, transform):
        self.paths = sorted(Path(root).glob("*.jpg"), key=lambda p: int(p.stem))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        return self.transform(Image.open(path).convert("RGB")), int(path.stem)


def load_backbone(name=MODEL, img_size=224):
    """Frozen feature extractor returning the pooled CLS embedding.

    The first call downloads the weights through timm and keeps a copy in dino_vit/weights/, so later
    runs work offline.
    """
    local = WEIGHTS_DIR / f"{name.split('.')[0]}.pt"
    if local.exists():
        model = timm.create_model(name, pretrained=False, num_classes=0, img_size=img_size)
        model.load_state_dict(torch.load(local, map_location="cpu"))
    else:
        model = timm.create_model(name, pretrained=True, num_classes=0, img_size=img_size)
        WEIGHTS_DIR.mkdir(exist_ok=True)
        torch.save(model.state_dict(), local)
    return model.eval()
