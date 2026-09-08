"""Focused tests for the Effort++ mathematical components."""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectors.effort_detector import CosineClassifier, replace_with_svd_residual
from regularization import CircleLoss, HypersphereProjection, SLERPAugmentation, uniformity_loss


class EffortComponentsTest(unittest.TestCase):
    def test_hypersphere_projection_has_unit_norm(self):
        projection = HypersphereProjection(8)
        output = projection(torch.randn(6, 8))
        torch.testing.assert_close(output.norm(dim=1), torch.ones(6))

    def test_slerp_stays_on_sphere_and_within_class(self):
        torch.manual_seed(1)
        augmentation = SLERPAugmentation(probability=1.0)
        augmentation.train()
        features = F.normalize(torch.randn(8, 16), dim=1)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        output = augmentation(features, labels)
        torch.testing.assert_close(output.norm(dim=1), torch.ones(8), atol=1e-5, rtol=1e-5)

    def test_svd_replacement_preserves_linear_output(self):
        torch.manual_seed(2)
        linear = nn.Linear(12, 10)
        inputs = torch.randn(4, 12)
        expected = linear(inputs)
        replacement = replace_with_svd_residual(linear, trainable_rank=1)
        torch.testing.assert_close(replacement(inputs), expected, atol=1e-5, rtol=1e-5)
        trainable = sum(p.numel() for p in replacement.parameters() if p.requires_grad)
        self.assertEqual(trainable, 23)

    def test_cosine_head_uses_fixed_scale(self):
        head = CosineClassifier(8, scale=20.0)
        self.assertFalse(head.scale.requires_grad)
        self.assertEqual(head(torch.randn(3, 8)).shape, (3, 2))

    def test_paper_losses_are_finite(self):
        features = F.normalize(torch.randn(8, 16), dim=1).requires_grad_()
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        loss = 0.5 * uniformity_loss(features) + 0.02 * CircleLoss()(features, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(features.grad)


if __name__ == "__main__":
    unittest.main()
