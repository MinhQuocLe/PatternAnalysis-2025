# train.py — minimal diff version
import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dataset import build_loaders
from modules import ConvNeXt
import numpy as np

print("PyTorch Version:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- Hyperparameters ----
data_root = "/home/groups/comp3710/ADNI/AD_NC"
batch_size = 32
num_workers = 0
num_epochs = 1
learning_rate = 1e-3
weight_decay = 5e-2
label_smoothing = 0.1
patience = 8                                 # early stopping
best_epoch = 1
ckpt_path = "logs/best_convnext.pth"
os.makedirs("logs", exist_ok=True)

# ---- Data (now returns train/val/test) ----
train_loader, val_loader, test_loader, class_to_idx = build_loaders(
    root=data_root, batch_size=batch_size, num_workers=num_workers, img_size=224, val_split=0.2
)
num_classes = len(class_to_idx)
print("Class mapping:", class_to_idx)

# ---- Model, loss, optimizer ----
model = ConvNeXt(num_classes=num_classes, in_chans=1, drop_path_rate=0.1).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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

def draw_training_curves(train_loss, val_loss, best_epoch, patience, save_dir="logs", show_plot=False):
    """
    Plot training and validation loss on one graph.
    Marks the epoch where early stopping was triggered.
    """
    import matplotlib.pyplot as plt, os, numpy as np
    os.makedirs(save_dir, exist_ok=True)

    epochs = np.arange(1, len(train_loss) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Training Loss", linewidth=2)
    plt.plot(epochs, val_loss, label="Validation Loss", linewidth=2)
    
    # mark early stopping line
    stop_epoch = best_epoch + patience if best_epoch + patience <= len(epochs) else len(epochs)
    plt.axvline(x=stop_epoch, color="r", linestyle="--", label="Early Stopping")
    
    plt.title("Training & Validation Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved training curve → {save_path}")

    if show_plot:
        plt.show()
    plt.close()
draw_training_curves(train_losses, val_losses, best_epoch, patience)
print(f"Done. Best val acc = {best_val_acc:.3f} at epoch {best_epoch}.")
print("Evaluate + make confusion matrix via: predict.py --ckpt logs/best_convnext.pth --split test")