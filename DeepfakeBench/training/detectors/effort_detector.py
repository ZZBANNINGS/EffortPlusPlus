"""Effort++ detector from the accompanying paper.

This module intentionally contains only the paper model: CLIP ViT-L/14,
rank-one SVD residual adaptation, hyperspherical feature learning, same-class
SLERP, a cosine classifier, and the optional asymmetric Circle objective.
"""

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from metrics.base_metrics_class import calculate_metrics_for_train
from regularization import CircleLoss, HypersphereMetricLearning, HypersphereProjection, SLERPAugmentation
from utils.registry import DETECTOR


logger = logging.getLogger(__name__)


class CosineClassifier(nn.Module):
    """Bias-free cosine classifier with a fixed or learnable logit scale."""

    def __init__(self, input_dim, num_classes=2, scale=20.0, learnable_scale=False):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=False)
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, features):
        features = F.normalize(features, p=2, dim=1)
        prototypes = F.normalize(self.linear.weight, p=2, dim=1)
        return self.scale * F.linear(features, prototypes)


@DETECTOR.register_module(module_name="effort")
class EffortDetector(nn.Module):
    """Paper-aligned Effort++ binary deepfake detector."""

    feature_dim = 1024

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.backbone = self._build_backbone()

        self.use_l2_norm = bool(self.config.get("use_l2_norm", False))
        self.hypersphere_projection = HypersphereProjection(
            self.feature_dim,
            learnable_scale=bool(self.config.get("learnable_scale", False)),
            initial_scale=float(self.config.get("initial_scale", 1.0)),
        )

        head_type = str(self.config.get("head_type", "linear")).lower()
        if head_type == "cosine":
            self.head = CosineClassifier(
                self.feature_dim,
                scale=float(self.config.get("cosine_scale", 20.0)),
                learnable_scale=False,
            )
        elif head_type == "linear":
            self.head = nn.Linear(self.feature_dim, 2)
        else:
            raise ValueError("head_type must be 'linear' or 'cosine'")

        self.loss_ce = nn.CrossEntropyLoss()
        self.metric_learning = HypersphereMetricLearning(
            alignment_weight=float(self.config.get("alignment_weight", 0.0)),
            uniformity_weight=float(self.config.get("uniformity_weight", 0.0)),
            alignment_alpha=float(self.config.get("alignment_alpha", 2.0)),
            uniformity_t=float(self.config.get("uniformity_t", 2.0)),
        )
        self.slerp = SLERPAugmentation(
            t_range=tuple(self.config.get("slerp_t_range", [0.0, 1.0])),
            probability=float(self.config.get("slerp_probability", 0.5)),
            same_class_only=True,
        )
        self.slerp_enabled = bool(self.config.get("slerp_enabled", False))

        self.circle_loss_weight = float(self.config.get("circle_loss_weight", 0.0))
        self.circle_loss = CircleLoss(
            gamma=float(self.config.get("circle_scale", 32.0)),
            m=float(self.config.get("circle_margin", 0.1)),
            real_only_pos=True,
        )

    def _build_backbone(self):
        default_path = Path(__file__).resolve().parents[1] / "models--openai--clip-vit-large-patch14"
        clip_path = self.config.get("clip_path", str(default_path))
        logger.info("Loading CLIP ViT-L/14 from %s", clip_path)
        backbone = CLIPModel.from_pretrained(clip_path, local_files_only=True).vision_model

        if bool(self.config.get("use_svd", True)):
            rank = int(self.config.get("svd_trainable_rank", 1))
            if rank < 1:
                raise ValueError("svd_trainable_rank must be at least 1")
            apply_svd_residual_to_self_attn(backbone, trainable_rank=rank)
        return backbone

    def features(self, data_dict):
        features = self.backbone(data_dict["image"]).pooler_output
        if self.use_l2_norm:
            features = self.hypersphere_projection(features)
        return features

    def forward(self, data_dict, inference=False):
        labels = data_dict.get("label")
        features = self.features(data_dict)
        classifier_features = features
        if self.training and not inference and self.slerp_enabled and labels is not None:
            classifier_features = self.slerp(features, labels)

        logits = self.head(classifier_features)
        return {
            "cls": logits,
            "prob": torch.softmax(logits, dim=1)[:, 1],
            "feat": features,
            "loss_feat": classifier_features,
            "loss_label": labels,
        }

    def get_losses(self, data_dict, pred_dict):
        labels = pred_dict.get("loss_label")
        if labels is None:
            labels = data_dict["label"]
        logits = pred_dict["cls"]
        features = pred_dict["loss_feat"]

        ce_loss = self.loss_ce(logits, labels)
        metric_losses = self.metric_learning(features, labels)
        circle_loss = features.sum() * 0.0
        if self.circle_loss_weight > 0:
            circle_loss = self.circle_loss(features, labels)

        overall = ce_loss + metric_losses["total_loss"] + self.circle_loss_weight * circle_loss
        return {
            "overall": overall,
            "ce_loss": ce_loss,
            "alignment_loss": metric_losses["alignment_loss"],
            "uniformity_loss": metric_losses["uniformity_loss"],
            "hypersphere_loss": metric_losses["total_loss"],
            "circle_loss": circle_loss,
        }

    def get_train_metrics(self, data_dict, pred_dict):
        labels = pred_dict.get("loss_label")
        if labels is None:
            labels = data_dict["label"]
        auc, eer, acc, ap = calculate_metrics_for_train(
            labels.detach(), pred_dict["cls"].detach()
        )
        return {"acc": acc, "auc": auc, "eer": eer, "ap": ap}


class SVDResidualLinear(nn.Module):
    """Frozen dominant SVD reconstruction plus a trainable spectral tail."""

    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.weight_main = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        if init_weight is None:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))
        else:
            self.weight_main.data.copy_(init_weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def compute_current_weight(self):
        residual = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        return self.weight_main + residual

    @property
    def weight(self):
        return self.compute_current_weight()

    def forward(self, inputs):
        return F.linear(inputs, self.compute_current_weight(), self.bias)


def replace_with_svd_residual(module, trainable_rank):
    """Replace one Linear layer while preserving its initial function exactly."""
    if not isinstance(module, nn.Linear):
        raise TypeError("SVD residual replacement expects nn.Linear")

    decomposition_rank = min(module.in_features, module.out_features) - trainable_rank
    if decomposition_rank < 0:
        raise ValueError("trainable rank exceeds the linear layer rank")

    replacement = SVDResidualLinear(
        module.in_features,
        module.out_features,
        decomposition_rank,
        bias=module.bias is not None,
        init_weight=module.weight.detach().clone(),
    )
    if module.bias is not None:
        replacement.bias.data.copy_(module.bias.data)

    with torch.no_grad():
        U, S, Vh = torch.linalg.svd(module.weight.detach(), full_matrices=False)
        U_r = U[:, :decomposition_rank]
        S_r = S[:decomposition_rank]
        V_r = Vh[:decomposition_rank]
        replacement.weight_main.copy_(U_r @ torch.diag(S_r) @ V_r)

    replacement.U_r = nn.Parameter(U_r.clone(), requires_grad=False)
    replacement.S_r = nn.Parameter(S_r.clone(), requires_grad=False)
    replacement.V_r = nn.Parameter(V_r.clone(), requires_grad=False)
    replacement.U_residual = nn.Parameter(U[:, decomposition_rank:].clone())
    replacement.S_residual = nn.Parameter(S[decomposition_rank:].clone())
    replacement.V_residual = nn.Parameter(Vh[decomposition_rank:].clone())
    return replacement


def apply_svd_residual_to_self_attn(backbone, trainable_rank=1):
    """Apply SVD residual adaptation to Q/K/V/out in every CLIP attention block."""
    projections = ("q_proj", "k_proj", "v_proj", "out_proj")
    layers = backbone.encoder.layers
    for layer in layers:
        attention = layer.self_attn
        for name in projections:
            setattr(
                attention,
                name,
                replace_with_svd_residual(getattr(attention, name), trainable_rank),
            )

    for parameter in backbone.parameters():
        parameter.requires_grad = False
    for module in backbone.modules():
        if isinstance(module, SVDResidualLinear):
            module.U_residual.requires_grad = True
            module.S_residual.requires_grad = True
            module.V_residual.requires_grad = True

    expected = len(layers) * len(projections)
    actual = sum(isinstance(module, SVDResidualLinear) for module in backbone.modules())
    if actual != expected:
        raise RuntimeError(f"expected {expected} SVD layers, created {actual}")
    logger.info(
        "Applied rank-%d SVD residuals to %d attention projections",
        trainable_rank,
        actual,
    )
    return backbone
