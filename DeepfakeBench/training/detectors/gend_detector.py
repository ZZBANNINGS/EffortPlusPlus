"""Local GenD reimplementation. The paper uses the authors' released checkpoint."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from metrics.base_metrics_class import calculate_metrics_for_train
from regularization import alignment_loss, uniformity_loss
from utils.registry import DETECTOR


@DETECTOR.register_module(module_name="gend")
class GenDDetector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        default_path = Path(__file__).resolve().parents[1] / "models--openai--clip-vit-large-patch14"
        clip_path = self.config.get("clip_path", str(default_path))
        self.backbone = CLIPModel.from_pretrained(
            clip_path, local_files_only=True
        ).vision_model

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for name, parameter in self.backbone.named_parameters():
            if "layer_norm" in name or "layernorm" in name:
                parameter.requires_grad = True

        self.head = nn.Linear(1024, 2)
        self.lambda_align = float(self.config.get("lambda_align", 0.1))
        self.lambda_unif = float(self.config.get("lambda_unif", 0.5))
        self.loss_ce = nn.CrossEntropyLoss()

    def features(self, data_dict):
        return self.backbone(data_dict["image"]).pooler_output

    def forward(self, data_dict, inference=False):
        raw_features = self.features(data_dict)
        normalized_features = F.normalize(raw_features, p=2, dim=1)
        logits = self.head(normalized_features)
        return {
            "cls": logits,
            "prob": torch.softmax(logits, dim=1)[:, 1],
            "feat": raw_features,
            "feat_norm": normalized_features,
        }

    def get_losses(self, data_dict, pred_dict):
        labels = data_dict["label"]
        features = pred_dict["feat_norm"]
        ce = self.loss_ce(pred_dict["cls"], labels)
        align = alignment_loss(features, labels)
        uniform = uniformity_loss(features)
        return {
            "overall": ce + self.lambda_align * align + self.lambda_unif * uniform,
            "ce_loss": ce,
            "alignment_loss": align,
            "uniformity_loss": uniform,
        }

    def get_train_metrics(self, data_dict, pred_dict):
        auc, eer, acc, ap = calculate_metrics_for_train(
            data_dict["label"].detach(), pred_dict["cls"].detach()
        )
        return {"acc": acc, "auc": auc, "eer": eer, "ap": ap}
