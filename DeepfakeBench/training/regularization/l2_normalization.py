"""L2 projection used by Effort++."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Normalization(nn.Module):
    def __init__(self, dim=-1, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, features):
        return F.normalize(features, p=2, dim=self.dim, eps=self.eps)


class HypersphereProjection(nn.Module):
    def __init__(self, feature_dim, learnable_scale=False, initial_scale=1.0, eps=1e-8):
        super().__init__()
        self.feature_dim = feature_dim
        self.learnable_scale = learnable_scale
        self.eps = eps
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1) * float(initial_scale))
        else:
            self.register_buffer("scale", torch.ones(1) * float(initial_scale))

    def forward(self, features):
        normalized = F.normalize(features, p=2, dim=-1, eps=self.eps)
        return normalized * self.scale if self.learnable_scale else normalized
