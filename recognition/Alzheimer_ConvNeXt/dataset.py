# dataset.py
# Data loading and preprocessing utilities for MRI slice classification.
# Handles patient-wise splitting to avoid data leakage between train/val.
# Author: Minh Quoc Le (s4939494)

from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import torch
import random
import os


# Pads 256×240 images to 256×256 (adds 8 px top and bottom)
def pad_256x240(img):
    w, h = img.size  
    pad = (0, 8, 0, 8)  # left, top, right, bottom
    return TF.pad(img, pad, fill=0) # fill with black

def build_transforms(img_size=256, is_train=True):
    """Return a torchvision transform pipeline for train/eval."""
    base = [
        pad_256x240,                                   
        transforms.Grayscale(num_output_channels=1),   
        transforms.Resize((img_size, img_size), antialias=True),  
    ]
    if is_train:
        # Mild geometric + photometric augmentation
        base += [
            transforms.RandomAffine(degrees=7, translate=(0.04, 0.04), scale=(0.97, 1.03)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.Lambda(lambda img: TF.adjust_gamma(img, random.uniform(0.90, 1.10))),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        ]

    base += [transforms.ToTensor()]                    
    if is_train:
        base += [
            # Small random noise and erasing for regularisation
            transforms.Lambda(lambda x: torch.clamp(x + 0.02 * torch.randn_like(x), 0.0, 1.0)),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.06), ratio=(0.33, 3.0), value=0),
        ]
    base += [transforms.Normalize(mean=[0.5], std=[0.5])]

    return transforms.Compose(base)

def patient_id_from_path(p: str) -> str:
    """Extract patient ID from filenames like '808819_106.png' → '808819'."""
    base = os.path.basename(p)             
    stem = os.path.splitext(base)[0]        
    return stem.split("_")[0]               


## ---- Patient-wise split loaders ----
def build_loaders(root, batch_size=32, num_workers=4, img_size=224, val_split=0.2, seed=42):
    """
    Build patient-wise train/val/test DataLoaders.
    Ensures all slices from the same patient stay in the same split.
    """
    rng = random.Random(seed)

    train_dir = Path(root) / "train"
    test_dir  = Path(root) / "test"

    tfms_train = build_transforms(img_size, is_train=True)
    tfms_eval  = build_transforms(img_size, is_train=False) 

    # Load without transforms to map patient IDs to indices
    base = datasets.ImageFolder(str(train_dir)) 

    # Map patient → list of sample indices
    id_to_indices = {}
    for idx, (img_path, _lbl) in enumerate(base.samples):
        pid = patient_id_from_path(img_path)
        id_to_indices.setdefault(pid, []).append(idx)

    patients = list(id_to_indices.keys())
    rng.shuffle(patients)

    n_val = max(1, int(len(patients) * val_split))
    val_patients   = set(patients[:n_val])
    train_patients = set(patients[n_val:])

    train_indices = [i for pid in train_patients for i in id_to_indices[pid]]
    val_indices   = [i for pid in val_patients   for i in id_to_indices[pid]]

    train_base = datasets.ImageFolder(str(train_dir), transform=tfms_train)
    val_base   = datasets.ImageFolder(str(train_dir), transform=tfms_eval)

    train_ds = Subset(train_base, train_indices)
    val_ds   = Subset(val_base,   val_indices)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Test set (no splitting)
    test_ds     = datasets.ImageFolder(str(test_dir), transform=tfms_eval)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Logs for sanity
    print(f"[Split] Patients: total={len(patients)} | train={len(train_patients)} | val={len(val_patients)}")
    print(f"[Split] Images:   train={len(train_indices)} | val={len(val_indices)} | test={len(test_ds)}")

    return train_loader, val_loader, test_loader, base.class_to_idx

