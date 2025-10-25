#Containing the data loader for loading and preprocessing your data

# dataset.py
# Minimal JPEG dataset loader for ConvNeXt (224x224x3) using torchvision ImageFolder.

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF


#Pad top/bottom 8px to make our 256x240 to 256x256
def pad_256x240(img):
    w, h = img.size  # should be 256x240
    pad = (0, 8, 0, 8)  # left, top, right, bottom
    return TF.pad(img, pad, fill=0) #fill = black

def build_transforms(img_size=256):
    return transforms.Compose([
        pad_256x240,  
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])



def build_loaders(root, batch_size= 32, num_workers = 4, img_size = 256):
    
    root = Path(root)
    train_dir = root / "train"
    test_dir = root / "test"

    train_tfms = build_transforms(img_size=img_size)

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tfms)
    test_ds  = datasets.ImageFolder(str(test_dir),  transform=train_tfms)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True    
    )
    return train_loader, test_loader, train_ds.class_to_idx #since ad is before nc: ad = 0, nc =1 
