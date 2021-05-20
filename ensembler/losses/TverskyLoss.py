import torch


class TverskyLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6, from_logits=True):
        super(TverskyLoss, self).__init__()
        # self.alpha = alpha
        # self.beta = beta
        # self.gamma = gamma
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, inputs, targets):

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        density = targets.sum() / torch.numel(targets)
        gamma = -torch.log(density) + 1
        gamma = gamma.clamp_max(2)
        # kappa = 2 * (0.5 - density).pow(3) + 1
        # alpha = 2 * kappa
        # beta = 2 * (1 - alpha)
        # gamma = (3 * (density - 0.5)).pow(2) + 1

        alpha = 1
        beta = 1

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + alpha * FP + beta * FN +
                                        self.smooth)

        Tversky = 2**gamma * (1 - Tversky)**gamma

        return Tversky
