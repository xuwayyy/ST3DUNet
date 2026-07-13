import torch
import torch.nn as nn
import numpy as np

class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=None):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

        if ignore_index == 6:
            freq = np.array([7.1, 10.3, 44.9, 0.7, 28, 8])
            freq = freq / freq.sum()

            class_weights = 1.0 / np.sqrt(freq + 1e-6)
            class_weights = class_weights / class_weights.sum() * len(freq)
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        elif ignore_index == 2:
            class_weights = torch.tensor([0.25, 0.75], dtype=torch.float32, device='cuda:2')

        self.register_buffer("class_weights", class_weights)

        self.loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=self.ignore_index)


    def forward(self, input, target):
        logits = input["logits"] if isinstance(input, dict) else input # [B, T, C, H, W]
        gt = target["gt"] if isinstance(target, dict) else target # [B, T, H, W]
        device = logits.device
        dtype = logits.dtype

        weights = self.class_weights.to(device=device, dtype=dtype) if self.class_weights is not None else None
        B, T, C, H, W = logits.shape
        logits = logits.reshape(B * T, C, H, W)
        gt = gt.reshape(B * T, H, W)

        return nn.functional.cross_entropy(logits, gt, weight=weights,
                               ignore_index=self.ignore_index if self.ignore_index is not None else -100)


# You can implement other Loss function and register here
LOSSES = {
    'ce': CrossEntropyLoss,
}
AVERAGE = {False: lambda x: x, True: lambda x: x.mean(dim=-1)}


class Losses(nn.Module):
    """The Losses meta-object that can take a mix of losses."""

    def __init__(self, mix={}, ignore_index=None):
        """Initializes the Losses object.
        Args:
            mix (dict): dictionary with keys "loss_name" and values weight
        """
        super(Losses, self).__init__()
        assert len(mix)
        self.ignore_index = ignore_index
        self.init_losses(mix)

    def init_losses(self, mix):
        """Initializes the losses.
        Args:
            mix (dict): dictionary with keys "loss_name" and values weight
        """
        self.loss = {}
        for m, v in mix.items():
            m = m.lower()
            try:
                self.loss[m] = (LOSSES[m](ignore_index=self.ignore_index), v)
            except KeyError:
                raise KeyError(f"Loss {m} not found in {LOSSES.keys()}")

    def forward(self, x, y, average=True):
        """Computes the losses.
        Args:
            x: dict that contains "logits": torch.Tensor BxTxKxHxW
            y: dict that contains "gt": torch.Tensor BxTxHxW
            average (bool): whether to average the losses or not
        Returns:
            dict: dictionary with losses
        """
        losses = {n: AVERAGE[average](f(x, y)) for n, (f, _) in self.loss.items()}
        losses["loss"] = sum([losses[n] * w for n, (_, w) in self.loss.items()])
        return losses

