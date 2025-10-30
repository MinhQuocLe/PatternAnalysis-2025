"""
train.py — ConvNeXt-Tiny training script (COMP3710 Project 8, Hard Difficulty)

Trains a ConvNeXt-Tiny classifier from scratch on the ADNI (AD vs NC) grayscale
MRI slice dataset.

Pipeline summary:
    • Build PyTorch DataLoaders (train/val/test) via dataset.build_loaders()
    • Create ConvNeXt-Tiny model (see modules.py)
    • Optimise with AdamW + label smoothing
    • Cosine LR schedule with a linear warm-up phase
    • Early stopping on validation loss with checkpointing
    • Save loss/accuracy curves to logs/train_val_curves.png
    • Save best weights to logs/best_convnext.pth

Outputs:
    logs/best_convnext.pth          — best model weights by val loss
    logs/train_val_curves.png       — training curves (loss & accuracy)
Author: Minh Quoc Le (s4939494)
"""
import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import build_loaders
from modules import ConvNeXt
import numpy as np

print("PyTorch Version:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- Hyperparameters -----------------------------------------------------------
data_root = "/home/groups/comp3710/ADNI/AD_NC"  # ADNI preprocessed (AD vs NC)
batch_size = 256
num_epochs = 500
learning_rate = 4e-3
weight_decay  = 1e-2
label_smoothing = 0.1
warmup_epochs = 10                  # linear LR warm-up length
patience = 50                       # early stopping patience (epochs with no val improvement)
best_epoch = 1
ckpt_path = "logs/best_convnext.pth"
os.makedirs("logs", exist_ok=True)

train_loader, val_loader, test_loader, class_to_idx = build_loaders(
    root=data_root, batch_size=batch_size, img_size=256, val_split=0.2
)
num_classes = len(class_to_idx)
print("Class mapping:", class_to_idx)

# Model: ConvNeXt-Tiny variant. Grayscale in_chans=1.
model = ConvNeXt(num_classes=num_classes, in_chans=1, drop_path_rate=0.1).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Scheduler: cosine annealing after warm-up phase.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs - warmup_epochs
)

train_losses, val_losses, train_accs, val_accs = [], [], [], []
best_val_loss = float("inf")
epochs_no_improve = 0

print("Training ConvNeXt")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward + loss
        logits = model(images)
        loss = criterion(logits, labels)

        # Backprop
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Accumulate stats
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_seen += bs

    tr_loss = total_loss / max(1, total_seen)
    tr_acc  = total_correct / max(1, total_seen)

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_correct = 0
        val_seen = 0

        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            l = criterion(out, y)

            bs = x.size(0)
            val_loss += l.item() * bs
            val_correct += (out.argmax(1) == y).sum().item()
            val_seen += bs

    va_loss = val_loss / max(1, val_seen)
    va_acc  = val_correct / max(1, val_seen)

    # Record epoch metrics
    train_losses.append(tr_loss); val_losses.append(va_loss)
    train_accs.append(tr_acc);    val_accs.append(va_acc)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"train_loss: {tr_loss:.4f} train_acc: {tr_acc:.3f}  "
        f"val_loss: {va_loss:.4f}   val_acc: {va_acc:.3f}"
    )

    # Track best validation loss and early stop; save model state dict when improved.
    if va_loss < best_val_loss:
        best_val_loss = va_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), ckpt_path)
        epochs_no_improve = 0
        print(f"  ↑ New best (val_loss={va_loss:.4f}); saved to {ckpt_path}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"  Early stopping (no improvement {patience} epochs).")
            break

    # Warm-up: linearly scale LR from 0 → learning_rate over warmup_epochs.
    # After warm-up: use cosine annealing scheduler each epoch.
    if epoch < warmup_epochs:
        scale = float(epoch + 1) / warmup_epochs
        for g in optimizer.param_groups:
            g["lr"] = learning_rate * scale
    else:
        scheduler.step()


def draw_training_curves(train_loss, val_loss, train_acc, val_acc, save_dir="logs", show_plot=False):
    """
    Plot and save training/validation loss and accuracy curves.

    Args:
        train_loss (List[float]): Per-epoch training loss.
        val_loss   (List[float]): Per-epoch validation loss.
        train_acc  (List[float]): Per-epoch training accuracy.
        val_acc    (List[float]): Per-epoch validation accuracy.
        save_dir (str): Directory to write the figure.
        show_plot (bool): If True, display interactively (useful locally).
    """
    import matplotlib.pyplot as plt, os, numpy as np
    os.makedirs(save_dir, exist_ok=True)
    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    # ---- Loss subplot ----
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_loss, label="Val Loss", linewidth=2)
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)

    # ---- Accuracy subplot ----
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train Acc", linewidth=2)
    plt.plot(epochs, val_acc, label="Val Acc", linewidth=2)
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "train_val_curves.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved training curves → {save_path}")

    if show_plot:
        plt.show()
    plt.close()

# ---- Save curves & final summary ----------------------------------------------
draw_training_curves(train_losses, val_losses, train_accs, val_accs)
print(f"Done. Best val loss = {best_val_loss:.3f} at epoch {best_epoch}.")
print("Evaluate + generate confusion matrix via: predict.py")
