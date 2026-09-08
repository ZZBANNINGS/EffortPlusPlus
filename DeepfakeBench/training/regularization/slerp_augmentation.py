"""Same-class spherical interpolation used by Effort++."""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def slerp(first, second, amount):
    first = F.normalize(first, p=2, dim=-1)
    second = F.normalize(second, p=2, dim=-1)
    dot = (first * second).sum(dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    interpolated = (
        torch.sin((1 - amount) * theta) / sin_theta * first
        + torch.sin(amount * theta) / sin_theta * second
    )
    return F.normalize(interpolated, p=2, dim=-1)


class SLERPAugmentation(nn.Module):
    def __init__(self, t_range=(0.0, 1.0), probability=0.5, same_class_only=True):
        super().__init__()
        if not same_class_only:
            raise ValueError("Effort++ uses same-class SLERP only")
        self.t_range = t_range
        self.probability = probability

    def forward(self, features, labels):
        if not self.training or random.random() >= self.probability:
            return features

        normalized = F.normalize(features, p=2, dim=1)
        augmented = normalized.clone()
        for label in torch.unique(labels):
            mask = labels == label
            class_features = normalized[mask]
            if class_features.size(0) < 2:
                continue
            partners = class_features[torch.randperm(class_features.size(0), device=features.device)]
            low, high = self.t_range
            amount = torch.empty(
                class_features.size(0), 1, device=features.device, dtype=features.dtype
            ).uniform_(float(low), float(high))
            augmented[mask] = slerp(class_features, partners, amount)
        return augmented
