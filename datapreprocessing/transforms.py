"""Baseline and conservative training augmentation; deterministic evaluation."""

from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_eval_transform(image_size=224):
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_train_transform(image_size=224, mode="enhanced", scale=(0.8, 1.0),
                          ratio=(0.9, 1.1), flip_probability=0.5,
                          rotation_degrees=10, brightness=0.1, contrast=0.1,
                          saturation=0.1, hue=0.0):
    """Baseline ignores enhanced parameters to reproduce the original exactly.

    Rotation uses bilinear interpolation and ImageNet-mean RGB fill to reduce
    harsh black corners. Small filled corners may still be visible.
    """
    evaluation = build_eval_transform(image_size)
    if mode == "baseline":
        return T.Compose([evaluation.transforms[0], T.RandomHorizontalFlip(),
                          *evaluation.transforms[1:]])
    if mode != "enhanced":
        raise ValueError("mode must be 'baseline' or 'enhanced'")
    if not (0 < scale[0] <= scale[1] <= 1):
        raise ValueError("scale must satisfy 0 < minimum <= maximum <= 1")
    if not (0 < ratio[0] <= ratio[1]):
        raise ValueError("ratio must contain two ordered positive values")
    if not 0 <= flip_probability <= 1:
        raise ValueError("flip_probability must lie in [0, 1]")
    if rotation_degrees < 0:
        raise ValueError("rotation_degrees must be nonnegative")
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=scale, ratio=ratio,
                            interpolation=InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=flip_probability),
        T.RandomRotation(rotation_degrees,
                         interpolation=InterpolationMode.BILINEAR,
                         fill=tuple(round(255 * x) for x in IMAGENET_MEAN)),
        T.ColorJitter(brightness=brightness, contrast=contrast,
                      saturation=saturation, hue=hue),
        *evaluation.transforms[1:],
    ])
