"""Tversky Loss"""
import torch
import torch.nn as nn
import numpy as np


class TverskyCrossEntropyDiceWeightedLoss(nn.Module):
    """https://www.kaggle.com/endoruk1234/selfdriving-segmentation-with-pytorch"""

    def __init__(self, num_classes, device):
        super(TverskyCrossEntropyDiceWeightedLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device

    def tversky_loss(self, pred, target, alpha=0.5, beta=0.5):
        target_oh = torch.eye(self.num_classes)[target.squeeze(1)]
        target_oh = target_oh.permute(0, 3, 1, 2).float()
        probs = nn.functional.softmax(pred, dim=1)
        target_oh = target_oh.type(pred.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        inter = torch.sum(probs * target_oh, dims)
        fps = torch.sum(probs * (1 - target_oh), dims)
        fns = torch.sum((1 - probs) * target_oh, dims)
        t = (inter / (inter + (alpha * fps) + (beta * fns))).mean()
        return 1 - t

    def class_dice(self, pred, target, epsilon=1e-6):
        pred_class = torch.argmax(pred, dim=1)
        dice = np.ones(self.num_classes)
        for c in range(self.num_classes):
            p = pred_class == c
            t = target == c
            inter = (p * t).sum().float() + epsilon
            union = p.sum() + t.sum() + epsilon
            d = 2 * inter / union
            dice[c] = 1 - d
        return torch.from_numpy(dice).float()

    def forward(self, pred, target, cross_entropy_weight=0.5, tversky_weight=0.5):
        if cross_entropy_weight + tversky_weight != 1:
            raise ValueError(
                "Cross Entropy weight and Tversky weight should " "sum to 1"
            )
        ce = nn.functional.cross_entropy(
            pred, target, weight=self.class_dice(pred, target).to(self.device)
        )
        tv = self.tversky_loss(pred, target)
        loss = (cross_entropy_weight * ce) + (tversky_weight * tv)
        return loss

    @classmethod
    def dice(cls, pred, target, epsilon=1e-6):
        num_classes = pred.size(1)
        pred_class = torch.argmax(pred, dim=1)
        dice = np.ones(num_classes)
        for c in range(num_classes):
            p = pred_class == c
            t = target == c
            inter = (p * t).sum().float() + epsilon
            union = p.sum() + t.sum() + epsilon
            d = 2 * inter / union
            dice[c] = d
        return torch.from_numpy(dice).float()
