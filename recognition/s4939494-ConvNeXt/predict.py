"""
predict.py — Evaluate a trained ConvNeXt-Tiny on a chosen split (COMP3710 Project 8)

Loads the best checkpoint (saved by train.py) and evaluates on one of:
    SPLIT ∈ {"train", "val", "test"}.

What it does:
    • Builds DataLoaders via dataset.build_loaders()
    • Recreates the ConvNeXt-Tiny model (same hyper-params as training)
    • Loads weights from CKPT_PATH
    • Runs inference and computes:
        - Accuracy
        - Classification report (precision/recall/F1)
        - Confusion matrix image saved to OUT_DIR
    • Optionally, if --image <path> is given, predicts a single image.

Outputs:
    logs/confusion_matrix_<split>.png
Author: Minh Quoc Le (s4939494)
"""
import os
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import numpy as np

from dataset import build_loaders
from modules import ConvNeXt

DATA_ROOT = "/home/groups/comp3710/ADNI/AD_NC"   
CKPT_PATH = "logs/best_convnext.pth"              # Best checkpoint from training
OUT_DIR   = "logs"                              
SPLIT     = "test"                                # {"train", "val", "test"}

IMG_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 4
# -------------------------------


@torch.inference_mode()
def evaluate(model, loader, device):
    """
    Run a forward pass over `loader` and compute predictions.

    Args:
        model (nn.Module): Trained classifier.
        loader (DataLoader): DataLoader for the chosen split.
        device (torch.device): CUDA/CPU.

    Returns:
        tuple:
            labels_all (np.ndarray): Ground-truth labels (N,).
            preds      (np.ndarray): Binary predictions (N,) after thresholding p(NC).
            probs      (np.ndarray): Softmax probabilities (N, num_classes).
            acc        (float): Accuracy of `preds` vs labels.
    """
    model.eval()
    logits_all, labels_all = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        logits_all.append(logits.detach().cpu())
        labels_all.append(y.detach().cpu())

    logits_all = torch.cat(logits_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0).numpy()

    # Convert logits → probabilities
    probs = F.softmax(logits_all, dim=1).numpy()

    # ------------------------
    # Thresholding choice:
    # THRESH_NC is the probability threshold on class "NC" (index 1 here).
    THRESH_NC = 0.94
    preds = (probs[:, 1] >= THRESH_NC).astype(int)
    # ------------------------

    acc = (preds == labels_all).mean()
    return labels_all, preds, probs, float(acc)


def save_confusion_matrix(y_true, y_pred, class_names, out_path):
    """
    Save a confusion matrix plot.

    Args:
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted labels.
        class_names (List[str]): Label names in index order.
        out_path (str): Destination filepath (PNG).
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap="magma")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


@torch.inference_mode()
def predict_single_image(model, image_path, device, idx_to_class):
    """Predict a single image and print a table of class confidences."""
    from PIL import Image
    from torchvision import transforms

    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    img = Image.open(image_path).convert("L")
    x = tfm(img).unsqueeze(0).to(device)

    model.eval()
    logits = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # shape [C]

    # Argmax prediction
    pred_idx = int(np.argmax(probs))
    pred_class = idx_to_class[pred_idx]

    # ---- Pretty console table ----
    print("\n=== SINGLE IMAGE PREDICTION ===")
    print(f"Image: {image_path}")
    print(f"Predicted class: {pred_class}\n")

    width_cls = max(len("Class"), *(len(c) for c in idx_to_class))
    print(f"{'Class'.ljust(width_cls)} | Confidence")
    print(f"{'-'*width_cls}-+-----------")
    for i, cls in enumerate(idx_to_class):
        print(f"{cls.ljust(width_cls)} | {probs[i]:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate ConvNeXt or predict a single image.")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image for prediction (e.g., /path/to/img.jpg)")
    parser.add_argument("--eval", action="store_true",
                        help="Run dataset evaluation (accuracy, report, confusion matrix).")
    parser.add_argument("--ckpt", type=str, default=CKPT_PATH,
                        help="Path to checkpoint (.pth) to load.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_to_idx = build_loaders(
        root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        img_size=IMG_SIZE,
        val_split=0.2
    )
    
    # Invert mapping to a list ["AD", "NC"] in index order
    idx_to_class = [k for k, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    num_classes = len(idx_to_class)
    print("Class mapping:", class_to_idx)

    # ---- Load model & checkpoint ----
    model = ConvNeXt(num_classes=num_classes, in_chans=1, drop_path_rate=0.2).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {args.ckpt}")

    # ---- Single image prediction mode ----
    if args.image:
        predict_single_image(model, args.image, device, idx_to_class)
        return

    # ---- Dataset evaluation mode (explicit with --eval, or default when no --image) ----
    if args.eval or not args.image:
        # ---- Select split ----
        split_map = {"train": train_loader, "val": val_loader, "test": test_loader}
        loader = split_map[SPLIT]
        print(f"Evaluating split: {SPLIT}")
        
        # ---- Evaluate ----
        y_true, y_pred, probs, acc = evaluate(model, loader, device)
        print(f"\nAccuracy: {acc:.4f}")
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, target_names=idx_to_class, digits=4))

        # ---- Visualisations ----
        cm_path = os.path.join(OUT_DIR, f"confusion_matrix_{SPLIT}.png")
        save_confusion_matrix(y_true, y_pred, idx_to_class, cm_path)
        print(f"Saved confusion matrix → {cm_path}")


if __name__ == "__main__":
    main()
