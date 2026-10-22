import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []

        for file_name in os.listdir(root_dir):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root_dir, file_name)
                self.image_paths.append(image_path)

        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        file_name = os.path.basename(image_path)

        return image, file_name


class CatsDogsDataLoader:
    def __init__(
        self,
        data_dir="datasets",
        batch_size=32,
        image_size=224,
        num_workers=0,
        pin_memory=False,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")
        test_dir = os.path.join(data_dir, "test")

        self.train_dataset = datasets.ImageFolder(
            root=train_dir,
            transform=self.train_transform,
        )

        self.val_dataset = datasets.ImageFolder(
            root=val_dir,
            transform=self.eval_transform,
        )

        self.test_dataset = TestDataset(
            root_dir=test_dir,
            transform=self.eval_transform,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

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
        return self.train_dataset.class_to_idx