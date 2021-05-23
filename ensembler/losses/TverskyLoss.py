import torch


class TverskyLoss(torch.nn.Module):
    def __init__(
        self,
        smooth=1e-6,
        alpha=0.5,
        from_logits=True,
        eps: float = 1e-6,
    ):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, inputs, targets):

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        density = targets.sum() / torch.numel(targets)
        gamma = -torch.log(density) + 1
        gamma = gamma.clamp_max(3)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets)
        FP = ((1 - targets) * inputs)
        FN = (targets * (1 - inputs))

        tversky = (TP + self.smooth) / (TP + self.alpha * FP +
                                        (1 - self.alpha) * FN + self.smooth)

        pt = 1.0 - tversky

        focal_term = pt.pow(gamma)

        loss = focal_term * pt
        norm_factor = focal_term.sum().clamp_min(self.eps)
        loss /= norm_factor

        return loss.sum()
