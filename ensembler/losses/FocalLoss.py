import segmentation_models_pytorch as smp
import torch
from typing import Optional
from functools import partial
import torch.nn.functional as F

__all__ = ["FocalLoss"]


def focal_loss(output: torch.Tensor,
               target: torch.Tensor,
               alpha: Optional[float] = 0.5,
               reduction: str = "mean",
               normalized: bool = False,
               reduced_threshold: Optional[float] = None,
               eps: float = 1e-6,
               weights=None,
               from_logits=False) -> torch.Tensor:
    """Compute binary focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.
    Args:
        output: Tensor of arbitrary shape (predictions of the model)
        target: Tensor of the same shape as input
        gamma: Focal loss power factor
        alpha: Weight factor to balance positive and negative samples. Alpha must be in [0...1] range,
            high values will give more weight to positive class.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
    References:
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.type(output.type())

    if from_logits:
        logpt = F.binary_cross_entropy_with_logits(output,
                                                   target,
                                                   reduction="none")
    else:
        logpt = F.binary_cross_entropy(output, target, reduction="none")
    pt = torch.exp(-logpt)

    density = target.sum() / torch.numel(target)
    gamma = -torch.log(density) + 1
    gamma = gamma.clamp_max(3)

    # compute the loss
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss = focal_term * logpt

    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)

    if weights is not None:
        for i, w in enumerate(weights):
            loss[:, i, ::] *= w

    loss = loss.view(-1)

    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss /= norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


class FocalLoss(smp.losses.FocalLoss):
    def __init__(self,
                 mode: str,
                 alpha: Optional[float] = None,
                 ignore_index: Optional[int] = None,
                 reduction: Optional[str] = "mean",
                 normalized: bool = False,
                 reduced_threshold: Optional[float] = None,
                 weights=None,
                 from_logits=False):
        super().__init__(mode, alpha, ignore_index, reduction, normalized,
                         reduced_threshold)

        self.focal_loss_fn = partial(focal_loss,
                                     alpha=alpha,
                                     reduced_threshold=reduced_threshold,
                                     reduction=reduction,
                                     normalized=normalized,
                                     weights=weights,
                                     from_logits=from_logits)

    def forward(self, y_pred: torch.Tensor,
                y_true: torch.Tensor) -> torch.Tensor:

        if self.mode in {
                smp.losses.constants.BINARY_MODE,
                smp.losses.constants.MULTILABEL_MODE
        }:

            if self.ignore_index is not None:
                # Filter predictions with ignore label from loss computation
                not_ignored = y_true != self.ignore_index
                y_pred = y_pred[not_ignored]
                y_true = y_true[not_ignored]

            loss = self.focal_loss_fn(y_pred, y_true)

        elif self.mode == smp.losses.constants.MULTICLASS_MODE:

            num_classes = y_pred.size(1)
            loss = 0

            # Filter anchors with -1 label from loss computation
            if self.ignore_index is not None:
                not_ignored = y_true != self.ignore_index

            for cls in range(num_classes):
                cls_y_true = (y_true == cls).long()
                cls_y_pred = y_pred[:, cls, ...]

                if self.ignore_index is not None:
                    cls_y_true = cls_y_true[not_ignored]
                    cls_y_pred = cls_y_pred[not_ignored]

                loss += self.focal_loss_fn(cls_y_pred, cls_y_true)

        return loss
