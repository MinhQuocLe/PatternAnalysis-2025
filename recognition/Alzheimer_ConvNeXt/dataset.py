#Containing the data loader for loading and preprocessing your data

# dataset.py


from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import random_split


#Pad top/bottom 8px to make our 256x240 to 256x256
def pad_256x240(img):
    w, h = img.size  # should be 256x240
    pad = (0, 8, 0, 8)  # left, top, right, bottom
    return TF.pad(img, pad, fill=0) #fill = black

def build_transforms(img_size=256):
    return transforms.Compose([
        pad_256x240,  
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def build_loaders(root, batch_size=32, num_workers=4, img_size=224, val_split=0.2):
    train_dir = Path(root) / "train"
    test_dir  = Path(root) / "test"

    tfms = build_transforms(img_size)
    full_train = datasets.ImageFolder(str(train_dir), transform=tfms)

    val_len = int(len(full_train) * val_split)
    train_len = len(full_train) - val_len
    train_ds, val_ds = random_split(full_train, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(datasets.ImageFolder(str(test_dir), transform=tfms),
                              batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, full_train.class_to_idx

