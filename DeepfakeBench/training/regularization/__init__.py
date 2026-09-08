"""Paper regularization components."""

from .hypersphere_metric import CircleLoss, HypersphereMetricLearning, alignment_loss, uniformity_loss
from .l2_normalization import HypersphereProjection, L2Normalization
from .slerp_augmentation import SLERPAugmentation, slerp

__all__ = [
    "CircleLoss",
    "HypersphereMetricLearning",
    "HypersphereProjection",
    "L2Normalization",
    "SLERPAugmentation",
    "alignment_loss",
    "slerp",
    "uniformity_loss",
]
