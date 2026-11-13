"""CIFAR-10: DINOv2 teacher -> custom CNN student knowledge distillation.

Three stages, run from the repository root:

    # 1. Fit the 10-class teacher (frozen DINOv2 ViT-B/14 + StandardScaler + logistic regression).
    python -m cifar10.kd teacher --data-dir datasets/cifar10 --out cifar10/outputs/teacher

    # 2. Train the CNN student. --kd-alpha 0 (no --teacher) gives the plain cross-entropy baseline.
    python -m cifar10.kd train --data-dir datasets/cifar10 \
        --teacher cifar10/outputs/teacher/teacher.npz --output-dir cifar10/outputs/kd_seed42
    python -m cifar10.kd train --data-dir datasets/cifar10 --kd-alpha 0 \
        --output-dir cifar10/outputs/ce_seed42

    # 3. Final report on the official 10,000-image test set.
    python -m cifar10.kd evaluate --data-dir datasets/cifar10 \
        --checkpoint cifar10/outputs/kd_seed42/best.pth

The official 50,000 training images are split (stratified and seeded) into 45,000 train and 5,000
validation images. The teacher's C and the student's best epoch are chosen on validation only; the
test set is used only for final reporting.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

from custom_cnn.distillation import distillation_loss
from custom_cnn.model import build_custom_cnn, parameter_counts
from custom_cnn.train import atomic_write, get_device, seed_everything

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HISTORY_FIELDS = ["epoch", "train_loss", "train_ce", "train_kd", "train_accuracy",
                  "teacher_train_accuracy", "val_loss", "val_accuracy", "val_balanced_accuracy",
                  "learning_rate",
                  "duration_seconds", "new_best"]


# --------------------------------------------------------------------------- data

def split_indices(targets, val_size, seed):
    """Stratified train/val split of the official training set; identical for teacher and student."""
    targets = np.asarray(targets)
    if val_size % len(CLASSES):
        raise ValueError("--val-size must be divisible by 10 for a stratified split")
    rng = np.random.default_rng(seed)
    val = np.concatenate([rng.permutation(np.flatnonzero(targets == c))[:val_size // len(CLASSES)]
                          for c in range(len(CLASSES))])
    mask = np.zeros(len(targets), dtype=bool)
    mask[val] = True
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def limit(indices, n, seed):
    """Debug/smoke-test subset; n=None keeps everything."""
    if n is None or n >= len(indices):
        return indices
    return np.sort(np.random.default_rng(seed).choice(indices, n, replace=False))


def imbalance_spec(args):
    """Step imbalance: the listed classes keep only `fraction` of their labelled images."""
    if not args.minority_classes or args.minority_fraction >= 1:
        return None
    if not 0 < args.minority_fraction < 1 or not set(args.minority_classes) <= set(range(10)):
        raise ValueError("Need 0 < --minority-fraction < 1 and class ids in 0..9")
    return dict(classes=sorted(set(args.minority_classes)), fraction=args.minority_fraction)


def make_imbalanced(indices, targets, spec, seed):
    if spec is None:
        return indices
    targets, rng = np.asarray(targets), np.random.default_rng(seed + 1)
    keep = []
    for c in range(len(CLASSES)):
        idx = indices[targets[indices] == c]
        if c in spec["classes"]:
            idx = rng.permutation(idx)[:max(1, round(len(idx) * spec["fraction"]))]
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def labelled_splits(targets, args):
    """Train/val indices after the stratified split, optional imbalance, and optional --limit.

    Validation is subsampled with the same imbalance, so model selection uses no labels that the
    imbalanced setting pretends do not exist. The official test set stays balanced.
    """
    spec = imbalance_spec(args)
    train_idx, val_idx = split_indices(targets, args.val_size, args.split_seed)
    train_idx = limit(make_imbalanced(train_idx, targets, spec, args.split_seed),
                      args.limit, args.split_seed)
    val_idx = limit(make_imbalanced(val_idx, targets, spec, args.split_seed + 7),
                    args.limit, args.split_seed)
    return train_idx, val_idx, spec


def class_counts(targets, indices):
    return np.bincount(np.asarray(targets)[indices], minlength=len(CLASSES))


def cifar(data_dir, train, transform):
    return datasets.CIFAR10(data_dir, train=train, transform=transform, download=True)


# Images leave the loader in [0, 1]; each model applies its own normalisation on-device, so the
# teacher and student see exactly the same augmented image.
PLAIN = transforms.ToTensor()
AUGMENT = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                              transforms.RandomHorizontalFlip(), transforms.ToTensor()])


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, images):
        return (images - self.mean) / self.std


def conv_bn(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
              nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers, *([nn.MaxPool2d(2)] if pool else []))


class Scale(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(conv_bn(channels, channels), conv_bn(channels, channels))

    def forward(self, x):
        return x + self.body(x)


def build_resnet9(num_classes, dropout):
    """ResNet-9 (Myrtle/DAWNBench style) for 32x32 inputs: 64-128-256-512 channels, pooling only
    to 4x4, two residual blocks, logits scaled by 1/8. ~6.6M parameters versus the CustomCNN's 1.2M."""
    return nn.Sequential(
        conv_bn(3, 64), conv_bn(64, 128, pool=True), Residual(128),
        conv_bn(128, 256, pool=True), conv_bn(256, 512, pool=True), Residual(512),
        nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Dropout(dropout),
        nn.Linear(512, num_classes), Scale(0.125))  # logit scaling from the original ResNet-9


ARCHITECTURES = {"cnn": build_custom_cnn, "resnet9": build_resnet9}


class Student(nn.Module):
    """Student network (10 logits) with CIFAR-10 normalisation built in.

    cnn: the project's CustomCNN, unchanged. resnet9: a stronger CIFAR-specific residual network.
    """

    def __init__(self, dropout, arch="cnn"):
        super().__init__()
        self.normalize = Normalize(CIFAR_MEAN, CIFAR_STD)
        self.cnn = ARCHITECTURES[arch](num_classes=len(CLASSES), dropout=dropout)

    def forward(self, images):
        return self.cnn(self.normalize(images))


# --------------------------------------------------------------------------- teacher

class DinoTeacher(nn.Module):
    """Frozen DINOv2 -> standardise -> multinomial logistic regression -> 10 logits.

    For multiclass lbfgs, sklearn's predict_proba is softmax(decision_function), so these logits
    reproduce the fitted probe exactly (checked in `teacher`).
    """

    def __init__(self, backbone, image_size, mean, scale, coef, intercept, amp=True):
        super().__init__()
        self.backbone, self.image_size, self.amp = backbone, image_size, amp
        self.normalize = Normalize(IMAGENET_MEAN, IMAGENET_STD)
        for name, value in (("mean", mean), ("scale", scale), ("coef", coef), ("intercept", intercept)):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):  # never leave eval mode
        return super().train(False)

    @torch.no_grad()
    def features(self, images):
        images = F.interpolate(images, size=(self.image_size,) * 2, mode="bilinear",
                               align_corners=False)
        with torch.autocast("cuda", enabled=self.amp and images.is_cuda):
            return self.backbone(self.normalize(images)).float()

    @torch.no_grad()
    def forward(self, images):
        return ((self.features(images) - self.mean) / self.scale) @ self.coef.T + self.intercept


def load_backbone(image_size):
    from dino_vit.common import MODEL, load_backbone as load  # lazy: timm only needed for KD
    return MODEL, load(img_size=image_size)


def load_teacher(path, device, amp):
    from dino_vit.common import MODEL
    with np.load(path, allow_pickle=False) as probe:
        meta = json.loads(str(probe["meta"]))
        arrays = {k: probe[k].copy() for k in ("mean", "scale", "coef", "intercept")}
    if meta["model"] != MODEL or meta["classes"] != list(CLASSES):
        raise ValueError("Teacher file was exported for a different backbone or class list")
    _, backbone = load_backbone(meta["image_size"])
    teacher = DinoTeacher(backbone, meta["image_size"], amp=amp, **arrays).to(device)
    return teacher, meta


@torch.no_grad()
def extract(teacher, dataset, batch_size, num_workers, device):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    feats, labels = [], []
    for i, (images, y) in enumerate(loader):
        feats.append(teacher.features(images.to(device, non_blocking=True)).cpu())
        labels.append(y)
        if i % 20 == 0:
            print(f"  {sum(len(f) for f in feats)}/{len(dataset)}", flush=True)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def cmd_teacher(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    device = get_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_name, backbone = load_backbone(args.image_size)
    tag = f"{model_name.split('.')[0]}_{args.image_size}px"
    extractor = DinoTeacher(backbone.to(device), args.image_size, np.zeros(1), np.ones(1),
                            np.zeros((1, 1)), np.zeros(1), amp=not args.no_amp).to(device)

    train_set, test_set = cifar(args.data_dir, True, PLAIN), cifar(args.data_dir, False, PLAIN)
    train_idx, val_idx, spec = labelled_splits(train_set.targets, args)
    test_idx = limit(np.arange(len(test_set)), args.limit, 0)
    cache_dir = Path(args.cache_dir or out)
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{args.limit}_split{args.split_seed}_{args.val_size}" if args.limit else ""
    cache = {}
    for split, dataset, idx in (("train", train_set, np.concatenate([train_idx, val_idx])),
                                ("test", test_set, test_idx)):
        path = cache_dir / f"features_{tag}_{split}{suffix}.npz"
        if path.exists():
            with np.load(path) as d:
                cache[split] = (d["x"], d["y"])
        else:
            print(f"Extracting {len(idx)} {split} features on {device} ...", flush=True)
            # Feature rows follow dataset order (sorted idx), so they can be indexed directly.
            full = np.zeros(len(dataset), dtype=bool)
            full[idx] = True
            x, y = extract(extractor, Subset(dataset, np.flatnonzero(full)), args.batch_size,
                           args.num_workers, device)
            cache[split] = (np.zeros((len(dataset), x.shape[1]), np.float32), np.asarray(dataset.targets))
            cache[split][0][full] = x
            np.savez(path, x=cache[split][0], y=cache[split][1])
    (x_all, y_all), (x_test, y_test) = cache["train"], cache["test"]
    x_test, y_test = x_test[test_idx], y_test[test_idx]
    x_tr, y_tr, x_va, y_va = x_all[train_idx], y_all[train_idx], x_all[val_idx], y_all[val_idx]

    scaler = StandardScaler().fit(x_tr)
    x_tr_s, x_va_s, x_te_s = (scaler.transform(x) for x in (x_tr, x_va, x_test))
    results, best = {}, (-1.0, None, None)
    # Imbalanced: select C on balanced accuracy (mean per-class recall) of the imbalanced val set.
    score = ((lambda m, x, y: balanced_accuracy_score(y, m.predict(x))) if spec
             else (lambda m, x, y: m.score(x, y)))
    metric = "balanced acc" if spec else "acc"
    print(f"Teacher train/val {len(train_idx)}/{len(val_idx)} | class counts "
          f"{class_counts(train_set.targets, train_idx).tolist()}", flush=True)
    for c in args.cs:
        clf = LogisticRegression(C=c, max_iter=3000,
                                 class_weight="balanced" if args.balanced_teacher else None
                                 ).fit(x_tr_s, y_tr)
        acc = score(clf, x_va_s, y_va)
        results[f"C={c:g}"] = acc
        print(f"logreg C={c:<8g} val {metric}={acc:.4f}", flush=True)
        if acc > best[0]:
            best = (acc, c, clf)
    val_acc, c, clf = best
    test_acc = clf.score(x_te_s, y_test)
    print(f"Teacher: C={c:g} | val {metric} {val_acc:.4f} | test acc {test_acc:.4f}")

    # The exported torch head must reproduce sklearn's probabilities.
    logits = torch.from_numpy(x_va_s @ clf.coef_.T + clf.intercept_)
    if not np.allclose(logits.softmax(1).numpy(), clf.predict_proba(x_va_s), atol=1e-6):
        raise RuntimeError("Torch teacher head does not reproduce sklearn predict_proba")

    meta = dict(model=model_name, image_size=args.image_size, classes=list(CLASSES),
                split_seed=args.split_seed, val_size=args.val_size, train_size=len(train_idx),
                regularization_c=c, val_accuracy=val_acc, test_accuracy=test_acc,
                c_sweep=results, imbalance=spec, balanced_teacher=args.balanced_teacher,
                selection_metric=metric,
                train_class_counts=class_counts(train_set.targets, train_idx).tolist())
    with (out / "teacher.npz").open("wb") as stream:
        np.savez(stream, mean=scaler.mean_, scale=scaler.scale_, coef=clf.coef_,
                 intercept=clf.intercept_, meta=json.dumps(meta))
    (out / "teacher_results.json").write_text(json.dumps(meta, indent=2))
    print(f"Exported teacher to {out / 'teacher.npz'}")


# --------------------------------------------------------------------------- student

def make_criterion(counts, args, device):
    """Supervised loss for training. Imbalance remedies (both from the class frequencies):

    --class-weights inverse|effective: re-weight each example's CE by 1/n_c or by the
        "effective number" (1-beta)/(1-beta^n_c) of Cui et al. (CVPR 2019); normalised to mean 1.
    --logit-adjust tau: logit-adjusted loss of Menon et al. (ICLR 2021): CE on
        logits + tau*log(prior). Training must then beat a margin that is larger for rare classes;
        prediction uses the raw logits, which estimates the balanced-error-optimal classifier.
    """
    counts = torch.as_tensor(counts, dtype=torch.float32, device=device).clamp_min(1)
    weight = None
    if args.class_weights == "inverse":
        weight = 1 / counts
    elif args.class_weights == "effective":
        weight = (1 - args.cb_beta) / (1 - args.cb_beta ** counts)
    if weight is not None:
        weight = weight * len(counts) / weight.sum()
    adjust = args.logit_adjust * torch.log(counts / counts.sum()) if args.logit_adjust else None

    def criterion(logits, labels):
        return F.cross_entropy(logits if adjust is None else logits + adjust, labels, weight=weight)
    return criterion


def run_epoch(model, loader, device, optimizer=None, scheduler=None, teacher=None, alpha=0.0,
              temperature=4.0, criterion=F.cross_entropy):
    training = optimizer is not None
    model.train(training)
    sums = dict(loss=0.0, ce=0.0, kd=0.0, correct=0, teacher_correct=0, total=0)
    class_correct = torch.zeros(len(CLASSES), dtype=torch.int64)
    class_total = torch.zeros(len(CLASSES), dtype=torch.int64)
    with torch.enable_grad() if training else torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            if training and teacher is not None:
                teacher_logits = teacher(images)
                _, _, kd = distillation_loss(logits, teacher_logits, labels, alpha, temperature)
                ce = criterion(logits, labels)
                loss = (1 - alpha) * ce + alpha * kd
                sums["teacher_correct"] += (teacher_logits.argmax(1) == labels).sum().item()
            else:
                loss = ce = criterion(logits, labels) if training else F.cross_entropy(logits, labels)
                kd = logits.new_zeros(())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()  # OneCycle: once per optimisation step
            n = labels.size(0)
            sums["loss"] += loss.item() * n
            sums["ce"] += ce.item() * n
            sums["kd"] += kd.item() * n
            hit = logits.argmax(1) == labels
            sums["correct"] += hit.sum().item()
            sums["total"] += n
            class_correct += torch.bincount(labels[hit], minlength=len(CLASSES)).cpu()
            class_total += torch.bincount(labels, minlength=len(CLASSES)).cpu()
    t = sums["total"]
    return dict(loss=sums["loss"] / t, ce=sums["ce"] / t, kd=sums["kd"] / t,
                accuracy=sums["correct"] / t,
                balanced_accuracy=float((class_correct[class_total > 0]
                                         / class_total[class_total > 0]).mean()),
                teacher_accuracy=sums["teacher_correct"] / t if teacher is not None else None)


def cmd_train(args):
    if not 0 <= args.kd_alpha <= 1 or args.kd_temperature <= 0:
        raise ValueError("Require kd-alpha in [0, 1] and positive kd-temperature")
    if args.kd_alpha > 0 and not args.teacher:
        raise ValueError("--kd-alpha > 0 needs --teacher (use --kd-alpha 0 for the CE baseline)")
    out = Path(args.output_dir)
    if (out / "best.pth").exists():
        raise FileExistsError(f"{out} already holds a run; choose a new --output-dir")
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    teacher, teacher_meta = None, None
    if args.kd_alpha > 0:
        # Built before seeding so backbone construction cannot shift the student's RNG stream.
        teacher, teacher_meta = load_teacher(args.teacher, device, amp=not args.no_amp)
        if (teacher_meta["split_seed"], teacher_meta["val_size"]) != (args.split_seed, args.val_size):
            raise ValueError("Teacher was fitted on a different train/val split; its val images "
                             "would leak into the student's training data")
        if teacher_meta.get("imbalance") != imbalance_spec(args):
            raise ValueError("Teacher was fitted on a different imbalanced subset; it would carry "
                             "labels the student's setting removes. Refit it with the same "
                             "--minority-classes/--minority-fraction")
    seed_everything(args.seed, args.deterministic)

    train_aug, train_plain = cifar(args.data_dir, True, AUGMENT), cifar(args.data_dir, True, PLAIN)
    train_idx, val_idx, spec = labelled_splits(train_aug.targets, args)
    counts = class_counts(train_aug.targets, train_idx)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    if args.sampler == "balanced":
        # Class-balanced resampling: each class drawn equally often (minority images repeat,
        # with fresh augmentation each time). Epoch length is unchanged.
        weights = 1.0 / counts[np.asarray(train_aug.targets)[train_idx]]
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        num_samples=len(train_idx), replacement=True,
                                        generator=generator)
    common = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda",
                  persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(Subset(train_aug, train_idx), batch_size=args.batch_size,
                              shuffle=sampler is None, sampler=sampler, drop_last=True,
                              generator=generator if sampler is None else None, **common)
    val_loader = DataLoader(Subset(train_plain, val_idx), batch_size=512, **common)

    model = Student(args.dropout, args.arch).to(device)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader),
        pct_start=0.15)
    config = {k: v for k, v in vars(args).items() if k != "func"}
    config.update(device=str(device), classes=list(CLASSES), train_size=len(train_idx),
                  val_size_actual=len(val_idx), parameters=parameter_counts(model),
                  imbalance=spec, train_class_counts=counts.tolist(),
                  selection_metric="val_balanced_accuracy" if spec else "val_accuracy",
                  teacher=teacher_meta, torch_version=str(torch.__version__),
                  teacher_sha256=(hashlib.sha256(Path(args.teacher).read_bytes()).hexdigest()
                                  if teacher is not None else None))
    atomic_write(out / "config.json", lambda p: p.write_text(json.dumps(config, indent=2)))
    print(f"Device {device} | student {args.arch} params {parameter_counts(model)['total']:,} | "
          f"train/val {len(train_idx)}/{len(val_idx)} | class counts {counts.tolist()} | "
          f"weights={args.class_weights} sampler={args.sampler} logit_adjust={args.logit_adjust} | "
          f"{'KD alpha=%g T=%g' % (args.kd_alpha, args.kd_temperature) if teacher else 'CE baseline'}",
          flush=True)

    criterion = make_criterion(counts, args, device)
    select = "balanced_accuracy" if spec else "accuracy"
    history, best_acc = [], -1.0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train = run_epoch(model, train_loader, device, optimizer, scheduler, teacher,
                          args.kd_alpha, args.kd_temperature, criterion)
        val = run_epoch(model, val_loader, device)
        new_best = val[select] > best_acc
        if new_best:
            best_acc = val[select]
            checkpoint = dict(model_state=model.state_dict(), epoch=epoch,
                              val_accuracy=val["accuracy"],
                              val_balanced_accuracy=val["balanced_accuracy"], config=config)
            atomic_write(out / "best.pth", lambda p: torch.save(checkpoint, p))
        history.append(dict(epoch=epoch, train_loss=train["loss"], train_ce=train["ce"],
                            train_kd=train["kd"], train_accuracy=train["accuracy"],
                            teacher_train_accuracy=train["teacher_accuracy"],
                            val_loss=val["loss"], val_accuracy=val["accuracy"],
                            val_balanced_accuracy=val["balanced_accuracy"], learning_rate=lr,
                            duration_seconds=time.perf_counter() - started, new_best=new_best))

        def write(p):
            with p.open("w", newline="") as stream:
                w = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
                w.writeheader()
                w.writerows(history)
        atomic_write(out / "training_history.csv", write)
        print(f"Epoch {epoch}/{args.epochs} | train {train['loss']:.4f}/{train['accuracy']:.4f}"
              + (f" (teacher {train['teacher_accuracy']:.4f})" if teacher else "")
              + f" | val {val['loss']:.4f}/{val['accuracy']:.4f}"
              + (f" bal {val['balanced_accuracy']:.4f}" if spec else "") + f" | lr {lr:.2e}"
              f" | {history[-1]['duration_seconds']:.1f}s" + (" | best" if new_best else ""),
              flush=True)
    print(f"Best val {select.replace('_', ' ')} {best_acc:.4f}; checkpoint {out / 'best.pth'}")


# --------------------------------------------------------------------------- evaluation

def cmd_evaluate(args):
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = Student(config["dropout"], config.get("arch", "cnn")).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    test_set = cifar(args.data_dir, False, PLAIN)
    if args.limit:
        test_set = Subset(test_set, limit(np.arange(len(test_set)), args.limit, 0))
    loader = DataLoader(test_set, batch_size=512, num_workers=args.num_workers)
    k = len(CLASSES)
    confusion = torch.zeros(k, k, dtype=torch.int64)
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(1).cpu()
            confusion += torch.bincount(labels * k + predictions, minlength=k * k).view(k, k)
    m = confusion.double()
    tp = m.diag()
    precision, recall = tp / m.sum(0).clamp_min(1), tp / m.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-15)
    metrics = dict(checkpoint=str(Path(args.checkpoint).resolve()),
                   best_epoch=checkpoint["epoch"], val_accuracy=checkpoint["val_accuracy"],
                   test_accuracy=float(tp.sum() / m.sum()), test_macro_f1=float(f1.mean()),
                   samples=int(m.sum()), confusion_matrix=confusion.tolist(),
                   per_class={c: dict(precision=float(precision[i]), recall=float(recall[i]),
                                      f1=float(f1[i]), support=int(m[i].sum()))
                              for i, c in enumerate(CLASSES)})
    spec = config.get("imbalance")
    if spec:
        minority = [i for i in range(k) if i in spec["classes"]]
        majority = [i for i in range(k) if i not in spec["classes"]]
        metrics.update(imbalance=spec, train_class_counts=config.get("train_class_counts"),
                       test_minority_mean_recall=float(recall[minority].mean()),
                       test_majority_mean_recall=float(recall[majority].mean()),
                       val_balanced_accuracy=checkpoint.get("val_balanced_accuracy"))
    path = Path(args.checkpoint).with_name("test_metrics.json")
    atomic_write(path, lambda p: p.write_text(json.dumps(metrics, indent=2)))
    print(f"Test accuracy {metrics['test_accuracy']:.4f} | macro F1 {metrics['test_macro_f1']:.4f}"
          + (f" | minority recall {metrics['test_minority_mean_recall']:.4f}"
             f" | majority recall {metrics['test_majority_mean_recall']:.4f}" if spec else ""))
    for c, v in metrics["per_class"].items():
        print(f"  {c:<11} recall {v['recall']:.4f}  precision {v['precision']:.4f}")
    print(f"Wrote {path}")


# --------------------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)

    def add_common(p):
        p.add_argument("--data-dir", default="datasets/cifar10", help="Downloaded here if missing")
        p.add_argument("--device", default="auto")
        p.add_argument("--num-workers", type=int, default=4)
        p.add_argument("--limit", type=int, help="Smoke test: cap train/val/test sizes")
        return p

    def add_split(p):
        p.add_argument("--val-size", type=int, default=5000)
        p.add_argument("--split-seed", type=int, default=42)
        p.add_argument("--no-amp", action="store_true", help="Run the teacher in full fp32")
        p.add_argument("--minority-classes", type=int, nargs="*", default=[],
                       help="Class ids that keep only --minority-fraction of their labels")
        p.add_argument("--minority-fraction", type=float, default=1.0)

    p = add_common(sub.add_parser("teacher", help="Fit and export the DINOv2 teacher"))
    add_split(p)
    p.add_argument("--out", default="cifar10/outputs/teacher")
    p.add_argument("--image-size", type=int, default=224, help="Teacher input (multiple of 14)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--cs", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0])
    p.add_argument("--cache-dir", help="Feature cache directory (default: --out)")
    p.add_argument("--balanced-teacher", action="store_true",
                   help="class_weight='balanced' in the teacher's logistic regression")
    p.set_defaults(func=cmd_teacher)

    p = add_common(sub.add_parser("train", help="Train the CNN student (KD or CE baseline)"))
    add_split(p)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--teacher", help="teacher.npz from the `teacher` stage")
    p.add_argument("--kd-alpha", type=float, default=0.5)
    p.add_argument("--kd-temperature", type=float, default=4.0)
    p.add_argument("--class-weights", choices=("none", "inverse", "effective"), default="none")
    p.add_argument("--cb-beta", type=float, default=0.999, help="Effective-number beta")
    p.add_argument("--sampler", choices=("random", "balanced"), default="random")
    p.add_argument("--logit-adjust", type=float, default=0.0, help="tau; 1.0 is standard")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--arch", choices=sorted(ARCHITECTURES), default="cnn", help="Student network")
    p.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw",
                   help="sgd = Nesterov momentum 0.9")
    p.add_argument("--lr", type=float, default=3e-3, help="OneCycle peak learning rate")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.set_defaults(func=cmd_train)

    p = add_common(sub.add_parser("evaluate", help="Final metrics on the official test set"))
    p.add_argument("--checkpoint", required=True)
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
