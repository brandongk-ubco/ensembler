from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["SoftBCELoss"]


class SoftBCELoss(nn.Module):

    __constants__ = [
        "weight", "pos_weight", "reduction", "ignore_index", "smooth_factor"
    ]

    def __init__(self,
                 weights=None,
                 ignore_index: Optional[int] = -100,
                 reduction: str = "mean",
                 smooth_factor: Optional[float] = None,
                 from_logits=False):
        """Drop-in replacement for torch.nn.SoftBCELoss with few additions: ignore_index and label_smoothing, configurable fromLogits
        
        Args:
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient. 
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 1] -> [0.9, 0.1, 0.9])
        
        Shape
             - **y_pred** - torch.Tensor of shape NxCxHxW
             - **y_true** - torch.Tensor of shape NxHxW or Nx1xHxW

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.from_logits = from_logits
        self.weights = weights

    def forward(self, y_pred: torch.Tensor,
                y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: torch.Tensor of shape (N, C, H, W)
            y_true: torch.Tensor of shape (N, H, W)  or (N, 1, H, W)
        
        Returns:
            loss: torch.Tensor
        """

        if self.smooth_factor is not None:
            soft_targets = (1 - y_true) * self.smooth_factor + y_true * (
                1 - self.smooth_factor)
        else:
            soft_targets = y_true

        if self.from_logits:
            loss = F.binary_cross_entropy_with_logits(y_pred,
                                                      soft_targets,
                                                      reduction="none")
        else:
            loss = F.binary_cross_entropy(y_pred,
                                          soft_targets,
                                          reduction="none")

        if self.weights is not None:
            for i, w in enumerate(self.weights):
                loss[:, i, ::] *= w

        if self.ignore_index is not None:
            not_ignored_mask = y_true != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss