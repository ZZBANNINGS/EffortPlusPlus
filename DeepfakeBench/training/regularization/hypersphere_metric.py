"""Hyperspherical objectives used in Effort++."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def alignment_loss(embeddings, labels, alpha=2.0):
    if embeddings.size(0) < 2:
        return embeddings.sum() * 0.0
    positive_mask = (labels[:, None] == labels[None, :]).triu(diagonal=1)
    pairs = positive_mask.nonzero(as_tuple=False)
    if pairs.numel() == 0:
        return embeddings.sum() * 0.0
    first = embeddings[pairs[:, 0]]
    second = embeddings[pairs[:, 1]]
    return (first - second).norm(p=2, dim=1).pow(alpha).mean()


def uniformity_loss(embeddings, t=2.0, eps=1e-6):
    if embeddings.size(0) < 2:
        return embeddings.sum() * 0.0
    return torch.pdist(embeddings, p=2).square().mul(-t).exp().mean().clamp_min(eps).log()


class HypersphereMetricLearning(nn.Module):
    def __init__(
        self,
        alignment_weight=0.0,
        uniformity_weight=0.0,
        alignment_alpha=2.0,
        uniformity_t=2.0,
    ):
        super().__init__()
        self.alignment_weight = alignment_weight
        self.uniformity_weight = uniformity_weight
        self.alignment_alpha = alignment_alpha
        self.uniformity_t = uniformity_t

    def forward(self, embeddings, labels):
        norms = embeddings.norm(p=2, dim=1)
        if not torch.allclose(
            norms,
            torch.ones(embeddings.size(0), device=embeddings.device),
            atol=1e-4,
        ):
            embeddings = F.normalize(embeddings, p=2, dim=1)
        align = alignment_loss(embeddings, labels, self.alignment_alpha)
        uniform = uniformity_loss(embeddings, self.uniformity_t)
        total = self.alignment_weight * align + self.uniformity_weight * uniform
        return {
            "alignment_loss": align,
            "uniformity_loss": uniform,
            "total_loss": total,
        }


class CircleLoss(nn.Module):
    """Asymmetric Circle Loss: real-real positives and cross-class negatives."""

    def __init__(self, gamma=32.0, m=0.1, real_only_pos=True):
        super().__init__()
        if not real_only_pos:
            raise ValueError("The Effort++ Circle variant uses real-real positives only")
        self.gamma = gamma
        self.m = m

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity = embeddings @ embeddings.T
        count = similarity.size(0)
        eye = torch.eye(count, dtype=torch.bool, device=labels.device)
        real = labels == 0
        positive_mask = real[:, None] & real[None, :] & ~eye
        negative_mask = labels[:, None] != labels[None, :]

        alpha_p = torch.relu(1 + self.m - similarity.detach())
        alpha_n = torch.relu(similarity.detach() + self.m)
        positive_logits = -self.gamma * alpha_p * (similarity - (1 - self.m))
        negative_logits = self.gamma * alpha_n * (similarity - self.m)

        neg_inf = torch.finfo(similarity.dtype).min
        positive_lse = torch.logsumexp(positive_logits.masked_fill(~positive_mask, neg_inf), dim=1)
        negative_lse = torch.logsumexp(negative_logits.masked_fill(~negative_mask, neg_inf), dim=1)
        valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if not valid.any():
            return similarity.sum() * 0.0
        return F.softplus(positive_lse[valid] + negative_lse[valid]).mean()
