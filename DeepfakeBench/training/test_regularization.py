"""Focused tests for the Effort++ mathematical components."""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectors.effort_detector import CosineClassifier, replace_with_svd_residual
from regularization import (
    CircleLoss,
    HypersphereMetricLearning,
    HypersphereProjection,
    SLERPAugmentation,
    slerp,
    uniformity_loss,
)


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

    def test_svd_regularizers_are_near_zero_at_initialization(self):
        torch.manual_seed(3)
        linear = nn.Linear(12, 10)
        replacement = replace_with_svd_residual(linear, trainable_rank=1)
        self.assertLess(float(replacement.compute_orthogonal_loss()), 1e-4)
        self.assertLess(float(replacement.compute_keepsv_loss()), 1e-4)

    def test_svd_reference_norm_is_a_nonpersistent_buffer(self):
        replacement = replace_with_svd_residual(nn.Linear(12, 10), trainable_rank=1)
        self.assertIn("weight_original_fnorm", dict(replacement.named_buffers()))
        self.assertNotIn("weight_original_fnorm", dict(replacement.named_parameters()))
        self.assertNotIn("weight_original_fnorm", list(replacement.state_dict()))
        replacement.to(dtype=torch.float64)
        self.assertEqual(replacement.weight_original_fnorm.dtype, torch.float64)

    def test_svd_loads_legacy_state_without_reference_norm_strictly(self):
        torch.manual_seed(4)
        linear = nn.Linear(12, 10)
        source = replace_with_svd_residual(linear, trainable_rank=1)
        target = replace_with_svd_residual(linear, trainable_rank=1)
        reference_norm = target.weight_original_fnorm.clone()
        with torch.no_grad():
            source.S_residual.add_(0.1)
        legacy_state = {
            key: value.clone()
            for key, value in source.state_dict().items()
            if key != "weight_original_fnorm"
        }
        result = target.load_state_dict(legacy_state, strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        torch.testing.assert_close(target.weight_original_fnorm, reference_norm)
        inputs = torch.randn(4, 12)
        torch.testing.assert_close(target(inputs), source(inputs))
        torch.testing.assert_close(target.compute_keepsv_loss(), source.compute_keepsv_loss())

    def test_cosine_head_uses_fixed_scale(self):
        head = CosineClassifier(8, scale=20.0)
        self.assertFalse(head.scale.requires_grad)
        self.assertEqual(head(torch.randn(3, 8)).shape, (3, 2))

    def test_cosine_head_matches_original_computation(self):
        torch.manual_seed(5)
        head = CosineClassifier(8, scale=20.0)
        base = F.normalize(torch.randn(3, 8), dim=1) * 1.00005
        actual_features = base.clone().requires_grad_()
        expected_features = base.clone().requires_grad_()
        actual = head(actual_features)
        expected = F.linear(
            expected_features,
            F.normalize(head.linear.weight, p=2, dim=1),
        ) * head.scale
        actual_gradient = torch.autograd.grad(actual.sum(), actual_features)[0]
        expected_gradient = torch.autograd.grad(expected.sum(), expected_features)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)

    def test_slerp_matches_original_computation(self):
        torch.manual_seed(6)
        first = torch.randn(4, 8, requires_grad=True)
        second = torch.randn(4, 8, requires_grad=True)
        amount = torch.rand(4, 1)
        expected_first = first.detach().clone().requires_grad_()
        expected_second = second.detach().clone().requires_grad_()
        first_norm = F.normalize(expected_first, dim=-1)
        second_norm = F.normalize(expected_second, dim=-1)
        dot = (first_norm * second_norm).sum(dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        amount_theta = amount * theta
        expected = (
            torch.sin(theta - amount_theta) / sin_theta * first_norm
            + torch.sin(amount_theta) / sin_theta * second_norm
        )
        actual = slerp(first, second, amount)
        actual_gradients = torch.autograd.grad(actual.sum(), (first, second))
        expected_gradients = torch.autograd.grad(expected.sum(), (expected_first, expected_second))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)

    def test_metric_learning_preserves_nearly_unit_features(self):
        torch.manual_seed(7)
        features = F.normalize(torch.randn(8, 16), dim=1) * 1.00005
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        metric = HypersphereMetricLearning(alignment_weight=0.0, uniformity_weight=1.0)
        result = metric(features, labels)
        expected = uniformity_loss(features)
        torch.testing.assert_close(result["uniformity_loss"], expected, rtol=0, atol=0)

    def test_paper_losses_are_finite(self):
        features = F.normalize(torch.randn(8, 16), dim=1).requires_grad_()
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        loss = 0.5 * uniformity_loss(features) + 0.02 * CircleLoss()(features, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(features.grad)


if __name__ == "__main__":
    unittest.main()
