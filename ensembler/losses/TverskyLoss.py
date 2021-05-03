import torch


class TverskyLoss(torch.nn.Module):
    def __init__(self,
                 weight=None,
                 size_average=True,
                 smooth=1e-6,
                 alpha=0.5,
                 beta=0.5,
                 gamma=1,
                 from_logits=True):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, inputs, targets):

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        if self.from_logits:
            inputs = torch.sigmoid(inputs)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN +
                                        self.smooth)

        Tversky = (1 - Tversky)**self.gamma

        return Tversky
