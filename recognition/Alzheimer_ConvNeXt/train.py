# train.py — minimal diff version
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

# ---- Hyperparameters ----
data_root = "/home/groups/comp3710/ADNI/AD_NC"
batch_size = 128
num_workers = 8
num_epochs = 300
learning_rate = 3e-4
weight_decay = 5e-4
label_smoothing = 0.1
patience = 50                                # early stopping
best_epoch = 1
ckpt_path = "logs/best_convnext.pth"
os.makedirs("logs", exist_ok=True)

# ---- Data (now returns train/val/test) ----
train_loader, val_loader, test_loader, class_to_idx = build_loaders(
    root=data_root, batch_size=batch_size, num_workers=num_workers, img_size=256, val_split=0.2
)
num_classes = len(class_to_idx)
print("Class mapping:", class_to_idx)

# ---- Model, loss, optimizer ----
model = ConvNeXt(num_classes=num_classes, in_chans=1, drop_path_rate=0.2).to(device)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.4, 1.0], device=device),label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

warmup_epochs = 5
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs - warmup_epochs
)

# ---- Tracking ----
train_losses, val_losses, train_accs, val_accs = [], [], [], []
best_val_acc = 0.0
epochs_no_improve = 0

print("Training ConvNeXt (scratch)...")
for epoch in range(num_epochs):
    # --- Train ---
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_seen += images.size(0)
    tr_loss = total_loss / max(1, total_seen)
    tr_acc  = total_correct / max(1, total_seen)

    # --- Validate ---
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_correct = 0
        val_seen = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            l = criterion(out, y)
            val_loss += l.item() * x.size(0)
            val_correct += (out.argmax(1) == y).sum().item()
            val_seen += x.size(0)
    va_loss = val_loss / max(1, val_seen)
    va_acc  = val_correct / max(1, val_seen)

    train_losses.append(tr_loss); val_losses.append(va_loss)
    train_accs.append(tr_acc);    val_accs.append(va_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"train_loss: {tr_loss:.4f} train_acc: {tr_acc:.3f}  "
          f"val_loss: {va_loss:.4f}   val_acc: {va_acc:.3f}")

    # --- Early stopping + checkpoint ---
    if va_acc > best_val_acc:
        best_val_acc = va_acc
        best_epoch = epoch + 1            # <-- track best epoch
        torch.save(model.state_dict(), ckpt_path)
        epochs_no_improve = 0
        print(f"  ↑ New best; saved to {ckpt_path}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"  Early stopping (no improvement {patience} epochs).")
            break
    # ---- Scheduler step  ----
    if epoch < warmup_epochs:
        for g in optimizer.param_groups:
            g["lr"] = learning_rate * float(epoch + 1) / warmup_epochs
    else:
        scheduler.step()


def draw_training_curves(train_loss, val_loss, train_acc, val_acc, save_dir="logs", show_plot=False):
    """
    Plot and save both training/validation loss and accuracy curves.
    """
    import matplotlib.pyplot as plt, os, numpy as np
    os.makedirs(save_dir, exist_ok=True)
    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    # ---- Loss plot ----
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_loss, label="Val Loss", linewidth=2)
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)

    # ---- Accuracy plot ----
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

# Save curves
draw_training_curves(train_losses, val_losses, train_accs, val_accs)
print(f"Done. Best val acc = {best_val_acc:.3f} at epoch {best_epoch}.")
print("Evaluate + make confusion matrix via: predict.py")