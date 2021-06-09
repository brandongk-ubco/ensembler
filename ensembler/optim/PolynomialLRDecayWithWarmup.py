from torch.optim.lr_scheduler import _LRScheduler
import warnings


class PolynomialLRDecayWithWarmup(_LRScheduler):
    def __init__(self,
                 optimizer,
                 total_steps,
                 warmup_steps,
                 min_lr=1e-7,
                 power=2.0,
                 last_epoch: int = -1):
        assert total_steps > 1
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):

        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch <= self.warmup_steps:
            return [
                self.min_lr +
                (base_lr - self.min_lr) * self.last_epoch / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        elif self.last_epoch > self.total_steps:
            return [self.min_lr for _ in self.base_lrs]
        else:
            current_step = self.last_epoch - self.warmup_steps
            total_steps = self.total_steps - self.warmup_steps

            return [
                (base_lr - self.min_lr) *
                (1 - current_step / total_steps)**(self.power) + self.min_lr
                for base_lr in self.base_lrs
            ]
