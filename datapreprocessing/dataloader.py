"""Independent loader with the teammate's accessors and return conventions."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS

from .transforms import build_eval_transform, build_train_transform

CLASS_TO_IDX = {"cat": 0, "dog": 1}
TEST_EXTENSIONS = (".jpg", ".jpeg", ".png")


def resolve_data_dir(data_dir=None):
    """Prefer an explicit path, else repo/datasets then sibling datasets."""
    repo = Path(__file__).resolve().parent.parent
    candidates = [Path(data_dir).expanduser()] if data_dir is not None else [
        repo / "datasets", repo.parent / "datasets"]
    for candidate in candidates:
        if all((candidate / split).is_dir() for split in ("train", "val", "test")):
            return candidate.resolve()
    raise FileNotFoundError("Expected train/, val/, test/ under: " +
                            ", ".join(str(p) for p in candidates))


def load_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def seed_worker(worker_id):
    """PyTorch supplies a distinct deterministic seed for each worker."""
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = sorted(p for p in self.root_dir.iterdir()
                                  if p.is_file() and p.suffix.lower() in TEST_EXTENSIONS)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = load_rgb(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, path.name


class CatsDogsDataLoader:
    def __init__(self, data_dir=None, batch_size=32, image_size=224,
                 num_workers=0, pin_memory=False, mode="enhanced", seed=42,
                 **augmentation_parameters):
        self.data_dir = resolve_data_dir(data_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mode = mode
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)
        self.train_transform = build_train_transform(image_size, mode,
                                                     **augmentation_parameters)
        self.eval_transform = build_eval_transform(image_size)
        for split in ("train", "val"):
            dataset = ImageFolder(self.data_dir / split, loader=load_rgb,
                                  transform=(self.train_transform if split == "train"
                                             else self.eval_transform))
            if dataset.class_to_idx != CLASS_TO_IDX:
                raise ValueError(f"{split}: expected {CLASS_TO_IDX}, found {dataset.class_to_idx}")
            setattr(self, f"{split}_dataset", dataset)
        self.test_dataset = TestDataset(self.data_dir / "test", self.eval_transform)
        for offset, split in enumerate(("train", "val", "test")):
            generator = None if seed is None else torch.Generator().manual_seed(seed + offset)
            loader = DataLoader(getattr(self, f"{split}_dataset"), batch_size=batch_size,
                                shuffle=(split == "train"), num_workers=num_workers,
                                pin_memory=pin_memory, worker_init_fn=seed_worker,
                                generator=generator)
            setattr(self, f"{split}_loader", loader)

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

    def get_test_loader(self):
        return self.test_loader

    def get_classes(self):
        return self.train_dataset.classes

    def get_class_to_idx(self):
        return self.train_dataset.class_to_idx.copy()
