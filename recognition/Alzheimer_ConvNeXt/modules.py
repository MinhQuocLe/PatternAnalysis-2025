"""
modules.py — ConvNeXt Tiny 
Implements a compact ConvNeXt-Tiny for grayscale MRI slices.
Architecture:
    Input  → PatchEmbed → 4 ConvNeXt stages → Global Avg Pool → Linear Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


# ----------------------------- Small helper modules -----------------------------

class PatchEmbed(nn.Sequential):
    """4x4 stride-4 stem (patch embedding) for 224→56.
    Converts 4x4 non-overlapping patches into tokens (feature vectors)."""
    def __init__(self, in_chans: int, out_chans: int):
        # Conv with stride=4 reduces H,W by 4; CFNorm normalises channels.
        super().__init__(
            nn.Conv2d(in_chans, out_chans, kernel_size=4, stride=4),
            CFNorm(out_chans)
        )

class Downsample(nn.Sequential):
    """2x2 stride-2 downsampling used between stages.
    Halves spatial size; increases channels in the next layer."""
    def __init__(self, in_chans: int, out_chans: int):
        # CFNorm before strided conv helps stability (norm → conv).
        super().__init__(
            CFNorm(in_chans),
            nn.Conv2d(in_chans, out_chans, kernel_size=2, stride=2)
        )

class Stage(nn.Sequential):
    """A stack of ConvNeXt blocks with per-block drop-path.
    depth = number of blocks; dp_rates supplies a per-block probability."""
    def __init__(self, dim: int, depth: int, dp_rates: list[float], layer_scale_init: float):
        super().__init__(*[
            ConvNeXtBlock(dim, drop_path=dp_rates[i], layer_scale_init=layer_scale_init)
            for i in range(depth)
        ])


# ------------------------ Channel-First LayerNorm (NCHW) ------------------------

class CFNorm(nn.Module):
    """
    Channel-first LayerNorm for (N, C, H, W) tensors.

    ConvNeXt often uses 'channels_last' (NHWC) for LayerNorm & Linear.
    This wrapper permutes to NHWC for LayerNorm, then back to NCHW.

    Args:
        num_channels (int): Number of feature channels (C) to normalise.
        eps (float, optional): Small value added to variance to avoid divide-by-zero. Defaults to 1e-6.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        # Learnable affine parameters per channel for LayerNorm
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        """
        Apply channel-wise LayerNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Normalised tensor with the same shape (N, C, H, W).
        """
        n, c, h, w = x.shape
        # Permute to NHWC → apply LN over C → permute back to NCHW
        y = x.permute(0, 2, 3, 1)                     # (N, H, W, C)
        y = F.layer_norm(y, (c,), self.weight, self.bias, self.eps)
        return y.permute(0, 3, 1, 2)                  # Back to (N, C, H, W)


# ------------------------------- ConvNeXt Block --------------------------------

class ConvNeXtBlock(nn.Module):
    """
    Core ConvNeXt block:
        Depthwise Conv (7x7) → LayerNorm → Linear → GELU → Linear → residual + stochastic depth

    Args:
        dim (int): Number of input and output channels.
        drop_path (float, optional): Probability for stochastic depth. Defaults to 0.0.
        layer_scale_init (float, optional): Initial scale value for layer scaling parameter γ. Defaults to 1e-6.
    """
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init: float = 1e-6):
        super().__init__()
        # Depthwise conv: per-channel spatial filtering (7×7) with padding to preserve size
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # LayerNorm (applied in channels_last layout)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # Channel mixing via 1×1 "MLP": expand → GELU → project back
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # Layer-scale γ (learnable) stabilises early training
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        # Stochastic depth (drop residual branch at random during training)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor):
        """
        Forward pass through one ConvNeXt block.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W), after residual addition.
        """
        residual = x
        # Depthwise conv in NCHW
        x = self.dwconv(x)
        # Switch to NHWC for LN + Linear ops
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # Layer scale if enabled
        if self.gamma is not None:
            x = self.gamma * x
        # Back to NCHW for later convs
        x = x.permute(0, 3, 1, 2)
        # Residual connection with optional stochastic depth
        return residual + self.drop_path(x)


# ------------------------------- ConvNeXt-Tiny Net ------------------------------

class ConvNeXt(nn.Module):
    """
     ConvNeXt-Tiny classifier for grayscale Alzheimer's MRI images.

    Consists of:
        • 4 downsampling stages (patch embedding + strided convs)
        • ConvNeXt blocks per stage
        • Global Average Pooling + Linear head for 2-class prediction

    Args:
        in_chans (int, optional): Number of input channels (1 for grayscale). Defaults to 1.
        num_classes (int, optional): Number of output classes. Defaults to 2 (AD vs NC).
        depths (tuple[int], optional): Number of blocks in each stage. Defaults to (3, 3, 9, 3).
        dims (tuple[int], optional): Channel dimensions per stage. Defaults to (96, 192, 384, 768).
        drop_path_rate (float, optional): Maximum stochastic depth rate. Defaults to 0.2.
        layer_scale_init (float, optional): Initial value for layer scaling γ. Defaults to 1e-6.
        head_init_scale (float, optional): Linear head scale multiplier. Defaults to 1.0.
    """
    def __init__(
        self,
        in_chans: int = 1,
        num_classes: int = 2,
        depths: tuple[int, ...] = (3, 3, 9, 3),
        dims: tuple[int, ...] = (96, 192, 384, 768),
        drop_path_rate: float = 0.2,
        layer_scale_init: float = 1e-6,
        head_init_scale: float = 1.0,
    ):
        super().__init__()

        # ---- 1) Downsampling stem (patch embed + 3 reductions) ----
        # 224→56 (PatchEmbed), then 56→28→14→7 via Downsample
        self.downsample_layers = nn.ModuleList([
            PatchEmbed(in_chans, dims[0]),
            Downsample(dims[0], dims[1]),
            Downsample(dims[1], dims[2]),
            Downsample(dims[2], dims[3]),
        ])

        # ---- 2) ConvNeXt stages ----
        # Create a list of drop-path rates distributed across all blocks (linear ramp)
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            # Build a sequential stack of ConvNeXtBlocks for stage i
            blocks = [
                ConvNeXtBlock(
                    dims[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init=layer_scale_init
                )
                for j in range(depths[i])
            ]
            self.stages.append(nn.Sequential(*blocks))
            cur += depths[i]

        # ---- 3) Classification head ----
        # LN on final channel dim then a Linear classifier (2 classes)
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        # ---- 4) Weight initialisation ----
        # Truncated normal init for Conv/Linear; scale head as per paper
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m: nn.Module):
        """
        Initialise Conv2D and Linear weights using truncated normal distribution.

        Args:
            m (nn.Module): A submodule within the network (Conv2d or Linear).
        """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor):
        """
        Forward pass up to the penultimate feature vector.

        Args:
            x (torch.Tensor): Input image batch of shape (N, C, H, W).

        Returns:
            torch.Tensor: Global average pooled feature vector of shape (N, C_last).
        """
        # Apply: PatchEmbed → Stage0 → Downsample → Stage1 → ... (4 times)
        for i in range(4):
            x = self.downsample_layers[i](x)  # resolution ↓, channels ↑
            x = self.stages[i](x)             # ConvNeXt blocks at this scale
        # Global average pooling over spatial dims (H,W) → (N, C)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass producing classification logits.

        Args:
            x (torch.Tensor): Input image batch of shape (N, 1, H, W).

        Returns:
            torch.Tensor: Logits of shape (N, num_classes).
        """
        return self.head(self.forward_features(x))


# ---------------------------- Factory for train.py ------------------------------

def build_model(in_chans=1, num_classes=2):
    """
    Factory function to build a ConvNeXt-Tiny classifier instance.

    Args:
        in_chans (int, optional): Number of input channels. Defaults to 1.
        num_classes (int, optional): Number of output classes. Defaults to 2.

    Returns:
        ConvNeXt: A ConvNeXt-Tiny model ready for training.
    """
    return ConvNeXt(in_chans=in_chans, num_classes=num_classes)
