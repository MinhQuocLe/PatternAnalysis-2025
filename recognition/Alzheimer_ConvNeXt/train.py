#containing the source code for training, validating, testing and saving your model. 
# The modelshould be imported from “modules.py” 
# and the data loader should be imported from “dataset.py”. 
# Make sure to plot the losses and metrics during training
import torch

# ---- Version + device ----
print("PyTorch Version:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

