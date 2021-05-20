import torch


class AdaptiveCrossEntropyLoss(torch.nn.Module):
    def __init__(self, from_logits=True):
        super(AdaptiveCrossEntropyLoss, self).__init__()
        self.from_logits = from_logits

    def forward(self, inputs, targets):

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        gamma = 2
        density = targets.sum() / torch.numel(targets)
        density = (density - 0.5).pow(3) + 0.5

        if self.from_logits:
            inputs = torch.sigmoid(inputs)

        positive_targets = targets[targets >= 0.5]
        negative_targets = targets[targets < 0.5]

        positive_inputs = inputs[targets >= 0.5]
        negative_inputs = inputs[targets < 0.5]

        positive_loss = -positive_targets * torch.log(
            positive_inputs.clamp(min=1e-7, max=1))
        negative_loss = -(1 - negative_targets) * torch.log(
            (1 - negative_inputs).clamp(min=1e-7, max=1))

        positive_pt = torch.exp(-positive_loss)
        positive_focal_term = (1.0 - positive_pt).pow(gamma)
        negative_pt = torch.exp(-negative_loss)
        negative_focal_term = (1.0 - negative_pt).pow(gamma)

        return 2 * ((1 - density) * positive_focal_term.mean() +
                    density * negative_focal_term.mean())
