"""
Loss functions used in HorusEye training.

ln_loss         : (L1 + sqrt(MSE)) / 2   — the primary reconstruction loss
sharp_loss      : gradient-based sharpness preservation (gradient + Sobel)
flow_loss       : slice-consistency loss for thickness reduction
ReconstructionLoss : weighted combination used during training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Fixed-weight gradient filters ─────────────────────────────────────────────

class _GradientFilter(nn.Module):
    """Shared base for fixed-kernel gradient filters."""

    def __init__(self, kernel_v: list, kernel_h: list):
        super().__init__()
        kv = torch.tensor(kernel_v, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kh = torch.tensor(kernel_h, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("weight_v", kv)
        self.register_buffer("weight_h", kh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for i in range(x.shape[1]):
            xi = x[:, i:i + 1]
            gv = F.conv2d(xi, self.weight_v, padding=1)
            gh = F.conv2d(xi, self.weight_h, padding=1)
            out.append(torch.sqrt(gv ** 2 + gh ** 2 + 1e-6))
        return torch.cat(out, dim=1)


class GradientFilter(_GradientFilter):
    def __init__(self):
        super().__init__(
            kernel_v=[[0, -1, 0], [0, 0, 0], [0, 1, 0]],
            kernel_h=[[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
        )


class SobelFilter(_GradientFilter):
    def __init__(self):
        super().__init__(
            kernel_v=[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            kernel_h=[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        )


# ── Loss functions ─────────────────────────────────────────────────────────────

def ln_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Combined L1 + sqrt(MSE) loss."""
    l1  = F.l1_loss(pred, gt)
    mse = F.mse_loss(pred, gt)
    return (l1 + torch.sqrt(mse)) / 2


def sharp_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Edge-preservation loss using gradient and Sobel responses."""
    device = pred.device
    grad  = GradientFilter().to(device)
    sobel = SobelFilter().to(device)
    loss  = F.l1_loss(grad(pred),  grad(gt))
    loss += F.l1_loss(sobel(pred), sobel(gt))
    return loss / 2


def flow_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Inter-slice consistency loss for thickness-reduction task.
    Penalises difference in consecutive-slice differences between pred and gt.
    Works on tensors of shape (B, C, H, W) where C is the slice dimension.
    """
    dp = pred[:, 1:] - pred[:, :-1]
    dg = gt[:, 1:]   - gt[:, :-1]
    return ln_loss(dp, dg)


def reconstruction_loss(pred: torch.Tensor,
                        gt: torch.Tensor,
                        lambda_sharp: float = 0.5) -> torch.Tensor:
    """Primary training loss: ln_loss + lambda * sharp_loss."""
    base = ln_loss(pred, gt)
    if lambda_sharp == 0:
        return base
    return base + lambda_sharp * sharp_loss(pred, gt)
