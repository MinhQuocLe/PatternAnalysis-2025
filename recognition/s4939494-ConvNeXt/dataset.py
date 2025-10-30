"""
dataset.py — Data loading and preprocessing utilities (COMP3710 Project 8)

Purpose:
    Build PyTorch DataLoaders for the ADNI (AD vs NC) grayscale MRI slice dataset,
    with a patient-wise split to prevent data leakage across train/val.

What it does:
    • Pads native 256x240 slices to 256x256 (letterbox) so we can resize square
    • Applies mild, medically plausible augmentations for training
    • Normalises grayscale to mean=0.5, std=0.5
    • Splits patients (not slices) into train/val; test is provided as-is
    • Returns train/val/test loaders and the class_to_idx mapping
Author: Minh Quoc Le (s4939494)
"""
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import torch
import random
import os

def pad_256x240(img):
    """
    Pad a 256x240 image to 256x256 by adding 8 px to top and bottom (black bars).
    This preserves anatomy without non-uniform scaling before square resize.

    Args:
        img (PIL.Image): Input image (expected size 256x240).

    Returns:
        PIL.Image: Padded to 256x256.
    """
    w, h = img.size
    # left, top, right, bottom
    pad = (0, 8, 0, 8)
    return TF.pad(img, pad, fill=0)


def build_transforms(img_size=256, is_train=True):
    """
    Construct the torchvision transform pipeline for training/evaluation.

    Design:
        • Always: pad → grayscale(1ch) → resize(square) → ToTensor → normalise
        • Train: light affine, brightness/contrast, slight gamma jitter,
                 optional blur, small noise, random erasing.

    Args:
        img_size (int): Final square size after padding and resize.
        is_train (bool): If True, include augmentations.

    Returns:
        torchvision.transforms.Compose
    """
    base = [
        pad_256x240,                                   # 256×240 → 256×256
        transforms.Grayscale(num_output_channels=1),   # enforce 1-channel
        transforms.Resize((img_size, img_size), antialias=True),
    ]
    if is_train:
        # Mild geometric + photometric augmentation (conservative ranges)
        base += [
            transforms.RandomAffine(degrees=7, translate=(0.04, 0.04), scale=(0.97, 1.03)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.Lambda(lambda img: TF.adjust_gamma(img, random.uniform(0.90, 1.10))),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        ]

    base += [transforms.ToTensor()]                    # [0,1] tensor
    if is_train:
        base += [
            # Small random noise + erasing for regularisation (post ToTensor)
            transforms.Lambda(lambda x: torch.clamp(x + 0.02 * torch.randn_like(x), 0.0, 1.0)),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.06), ratio=(0.33, 3.0), value=0),
        ]

    # Normalise around mid-grey; pairs with model initialisation and loss stability
    base += [transforms.Normalize(mean=[0.5], std=[0.5])]

    return transforms.Compose(base)


def patient_id_from_path(p: str) -> str:
    """
    Extract patient ID from filenames like '808819_106.png' → '808819'.

    Assumes the dataset naming convention is '<patient>_<slice>.png'.

    Args:
        p (str): File path string.

    Returns:
        str: Patient identifier.
    """
    base = os.path.basename(p)
    stem = os.path.splitext(base)[0]
    return stem.split("_")[0]


def build_loaders(root, batch_size=32, num_workers=4, img_size=224, val_split=0.2, seed=42):
    """
    Build train/val/test DataLoaders with a patient-wise split.

    Rationale:
        All slices from the same patient must remain in the same split to avoid
        data leakage (over-optimistic validation). We first index patients using
        the raw ImageFolder samples, split IDs, then create Subsets.

    Args:
        root (str | Path): Dataset root containing 'train/' and 'test/' subdirs.
        batch_size (int): Batch size for all splits.
        num_workers (int): DataLoader worker processes.
        img_size (int): Target square size (after padding).
        val_split (float): Fraction of TRAIN PATIENTS reserved for validation.
        seed (int): Seed for reproducible patient shuffling.

    Returns:
        tuple:
            train_loader (DataLoader)
            val_loader   (DataLoader)
            test_loader  (DataLoader)
            class_to_idx (Dict[str, int])
    """
    rng = random.Random(seed)

    train_dir = Path(root) / "train"
    test_dir  = Path(root) / "test"

    tfms_train = build_transforms(img_size, is_train=True)
    tfms_eval  = build_transforms(img_size, is_train=False)

    # Load without transforms first to map patient IDs → sample indices
    base = datasets.ImageFolder(str(train_dir))

    # Map patient → indices (in the base dataset)
    id_to_indices = {}
    for idx, (img_path, _lbl) in enumerate(base.samples):
        pid = patient_id_from_path(img_path)
        id_to_indices.setdefault(pid, []).append(idx)

    # Shuffle patient IDs and split into train/val sets
    patients = list(id_to_indices.keys())
    rng.shuffle(patients)

    n_val = max(1, int(len(patients) * val_split))
    val_patients   = set(patients[:n_val])
    train_patients = set(patients[n_val:])

    # Collect indices for each split
    train_indices = [i for pid in train_patients for i in id_to_indices[pid]]
    val_indices   = [i for pid in val_patients   for i in id_to_indices[pid]]

    # Wrap with transforms after indices are chosen
    train_base = datasets.ImageFolder(str(train_dir), transform=tfms_train)
    val_base   = datasets.ImageFolder(str(train_dir), transform=tfms_eval)

    train_ds = Subset(train_base, train_indices)
    val_ds   = Subset(val_base,   val_indices)

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Test set: no splitting; use eval transforms
    test_ds = datasets.ImageFolder(str(test_dir), transform=tfms_eval)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Logs for sanity check / reproducibility tracking
    print(f"[Split] Patients: total={len(patients)} | train={len(train_patients)} | val={len(val_patients)}")
    print(f"[Split] Images:   train={len(train_indices)} | val={len(val_indices)} | test={len(test_ds)}")

    return train_loader, val_loader, test_loader, base.class_to_idx
