import argparse
import csv

import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import CatsDogsDataLoader
from vit import build_vit_model
from resnet import build_resnet_model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 10 == 0:
            current_loss = running_loss / total
            current_acc = correct / total

            print(
                f"Epoch {epoch} | "
                f"Batch [{batch_idx + 1}/{len(train_loader)}] | "
                f"Loss: {current_loss:.4f} | "
                f"Acc: {current_acc:.4f}",
                flush=True,
            )

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, val_loader, criterion, device, epoch):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    print(f"Evaluating epoch {epoch}...", flush=True)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 10 == 0:
                current_loss = running_loss / total
                current_acc = correct / total

                print(
                    f"Val Batch [{batch_idx + 1}/{len(val_loader)}] | "
                    f"Loss: {current_loss:.4f} | "
                    f"Acc: {current_acc:.4f}",
                    flush=True,
                )

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def predict_test(model, test_loader, device, idx_to_class):
    model.eval()

    results = []

    print("Predicting test images...", flush=True)

    with torch.no_grad():
        for batch_idx, (images, file_names) in enumerate(test_loader):
            images = images.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            preds = preds.cpu().tolist()

            for file_name, pred_idx in zip(file_names, preds):
                pred_class = idx_to_class[pred_idx]
                results.append((file_name, pred_class))

            if batch_idx % 10 == 0:
                print(
                    f"Test Batch [{batch_idx + 1}/{len(test_loader)}]",
                    flush=True,
                )

    return results


def save_predictions(results, save_path):
    with open(save_path, mode="w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["filename", "label"])

        for file_name, label in results:
            writer.writerow([file_name, label])


def build_model(args, device):
    if args.model == "vit":
        model = build_vit_model(
            num_classes=2,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone,
            dropout=args.dropout,
            device=device,
        )

    elif args.model == "resnet":
        model = build_resnet_model(
            num_classes=2,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone,
            dropout=args.dropout,
            device=device,
        )

    else:
        raise ValueError(f"Unknown model type: {args.model}")

    return model


def get_args():
    parser = argparse.ArgumentParser(
        description="Train ViT or ResNet for Cats vs Dogs classification"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="vit",
        choices=["vit", "resnet"],
        help="Choose model architecture: vit or resnet",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets",
        help="Dataset directory containing train, val, and test folders",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input image size",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers. Use 0 for Mac/MPS.",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout before classifier head",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW",
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained ImageNet weights",
    )

    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze backbone and train only classifier head",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="best_model.pth",
        help="Path to save best model weights",
    )

    parser.add_argument(
        "--pred_path",
        type=str,
        default="test_predictions.csv",
        help="Path to save test predictions CSV",
    )

    return parser.parse_args()


def main():
    args = get_args()

    device = get_device()

    print(f"Using device: {device}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Learning rate: {args.lr}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)

    data = CatsDogsDataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    train_loader, val_loader, test_loader = data.get_loaders()

    print("Classes:", data.get_classes(), flush=True)
    print("Class to idx:", data.get_class_to_idx(), flush=True)

    print(f"Train images: {len(data.train_dataset)}", flush=True)
    print(f"Val images: {len(data.val_dataset)}", flush=True)
    print(f"Test images: {len(data.test_dataset)}", flush=True)

    print(f"Train batches: {len(train_loader)}", flush=True)
    print(f"Val batches: {len(val_loader)}", flush=True)
    print(f"Test batches: {len(test_loader)}", flush=True)

    class_to_idx = data.get_class_to_idx()

    idx_to_class = {
        idx: class_name
        for class_name, idx in class_to_idx.items()
    }

    model = build_model(args, device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = 0.0

    print("Starting training...", flush=True)

    for epoch in range(args.epochs):
        epoch_number = epoch + 1

        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch_number,
        )

        val_loss, val_acc = evaluate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch_number,
        )

        print(
            f"Epoch [{epoch_number}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                args.save_path,
            )

            print(f"Saved best model to {args.save_path}", flush=True)

    print("Loading best model...", flush=True)

    model.load_state_dict(
        torch.load(
            args.save_path,
            map_location=device,
        )
    )

    test_results = predict_test(
        model=model,
        test_loader=test_loader,
        device=device,
        idx_to_class=idx_to_class,
    )

    save_predictions(
        results=test_results,
        save_path=args.pred_path,
    )

    print(f"Best Val Acc: {best_val_acc:.4f}", flush=True)
    print(f"Saved test predictions to {args.pred_path}", flush=True)


if __name__ == "__main__":
    main()