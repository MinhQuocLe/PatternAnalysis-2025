#Containing the data loader for loading and preprocessing your data

# dataset.py


from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import torch
import random
import os


#Pad top/bottom 8px to make our 256x240 to 256x256
def pad_256x240(img):
    w, h = img.size  # should be 256x240
    pad = (0, 8, 0, 8)  # left, top, right, bottom
    return TF.pad(img, pad, fill=0) #fill = black

def build_transforms(img_size=256, is_train=True):
    base = [
        pad_256x240,                                   
        transforms.Grayscale(num_output_channels=1),   
        transforms.Resize((img_size, img_size), antialias=True),  
    ]
    if is_train:
        base += [
            transforms.RandomAffine(degrees=7, translate=(0.04, 0.04), scale=(0.97, 1.03)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.Lambda(lambda img: TF.adjust_gamma(img, random.uniform(0.90, 1.10))),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        ]

    base += [transforms.ToTensor()]                    
    if is_train:
        base += [
            transforms.Lambda(lambda x: torch.clamp(x + 0.02 * torch.randn_like(x), 0.0, 1.0)),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.06), ratio=(0.33, 3.0), value=0),
        ]
    base += [transforms.Normalize(mean=[0.5], std=[0.5])]

    return transforms.Compose(base)


# ---- Helper: extract patient id from "808819_106.png" -> "808819" ----
def patient_id_from_path(p: str) -> str:
    base = os.path.basename(p)              # e.g. "808819_106.png"
    stem = os.path.splitext(base)[0]        # e.g. "808819_106"
    return stem.split("_")[0]               # e.g. "808819"


## ---- Patient-wise split loaders ----
def build_loaders(root, batch_size=32, num_workers=4, img_size=224, val_split=0.2, seed=42):
    """
    Patient-wise split:
      - All slices for a given patient go to exactly one of {train, val}
      - Test set is loaded as-is from root/test
    """
    rng = random.Random(seed)

    train_dir = Path(root) / "train"
    test_dir  = Path(root) / "test"

    tfms_train = build_transforms(img_size, is_train=True)
    tfms_eval  = build_transforms(img_size, is_train=False) 

    # Base dataset to enumerate samples & classes (NO heavy augments here)
    base = datasets.ImageFolder(str(train_dir))  # no transform; we only need .samples and .class_to_idx

    # Map patient_id -> list of dataset indices
    id_to_indices = {}
    for idx, (img_path, _lbl) in enumerate(base.samples):
        pid = patient_id_from_path(img_path)
        id_to_indices.setdefault(pid, []).append(idx)

    patients = list(id_to_indices.keys())
    rng.shuffle(patients)

    n_val = max(1, int(len(patients) * val_split))
    val_patients   = set(patients[:n_val])
    train_patients = set(patients[n_val:])

    # Flatten indices
    train_indices = [i for pid in train_patients for i in id_to_indices[pid]]
    val_indices   = [i for pid in val_patients   for i in id_to_indices[pid]]

    # Optional sanity: ensure no overlap
    assert set(train_indices).isdisjoint(val_indices), "Leak: train/val indices overlap"
    assert len(train_indices) + len(val_indices) == len(base.samples), "Mismatch in split sizes"

    # Create two *separate* dataset instances so we can give different transforms if we want later
    train_base = datasets.ImageFolder(str(train_dir), transform=tfms_train)
    val_base   = datasets.ImageFolder(str(train_dir), transform=tfms_eval)

    train_ds = Subset(train_base, train_indices)
    val_ds   = Subset(val_base,   val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Test loader (no split). Reuse eval transforms.
    test_ds     = datasets.ImageFolder(str(test_dir), transform=tfms_eval)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Logs for sanity
    print(f"[Split] Patients: total={len(patients)} | train={len(train_patients)} | val={len(val_patients)}")
    print(f"[Split] Images:   train={len(train_indices)} | val={len(val_indices)} | test={len(test_ds)}")

    return train_loader, val_loader, test_loader, base.class_to_idx

