import os
import sys
import unittest

import torch


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.loss.ua_asl import UncertaintyAwareASL


class TestUncertaintyAwareASL(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.criterion = UncertaintyAwareASL(
            gamma_pos=0.0,
            gamma_neg=4.0,
            margin=0.05,
            lambda_unc=0.5,
            alpha=0.5,
            reduction='mean',
        )
        self.batch_size = 8
        self.num_classes = 14

    def test_normal_batch_no_nan(self):
        logits = torch.randn(self.batch_size, self.num_classes, dtype=torch.float32)
        targets = torch.randint(
            low=-1,
            high=2,
            size=(self.batch_size, self.num_classes),
            dtype=torch.int64,
        ).float()

        loss = self.criterion(logits, targets)
        self.assertTrue(torch.isfinite(loss).item(), 'Loss should be finite for normal batch')

    def test_all_uncertain_finite(self):
        logits = torch.randn(self.batch_size, self.num_classes, dtype=torch.float32)
        targets = -torch.ones(self.batch_size, self.num_classes, dtype=torch.float32)

        loss = self.criterion(logits, targets)
        self.assertTrue(
            torch.isfinite(loss).item(),
            'Loss should be finite when all labels are uncertain',
        )

    def test_all_positive_better_prediction_has_lower_loss(self):
        targets = torch.ones(self.batch_size, self.num_classes, dtype=torch.float32)

        logits_good = torch.full((self.batch_size, self.num_classes), 5.0, dtype=torch.float32)
        logits_bad = torch.full((self.batch_size, self.num_classes), -5.0, dtype=torch.float32)

        loss_good = self.criterion(logits_good, targets)
        loss_bad = self.criterion(logits_bad, targets)

        self.assertLess(
            loss_good.item(),
            loss_bad.item(),
            'Loss should be lower when positive labels are predicted correctly',
        )

    def test_backward_gradient_no_nan(self):
        logits = torch.randn(
            self.batch_size,
            self.num_classes,
            dtype=torch.float32,
            requires_grad=True,
        )
        targets = torch.randint(
            low=-1,
            high=2,
            size=(self.batch_size, self.num_classes),
            dtype=torch.int64,
        ).float()

        loss = self.criterion(logits, targets)
        loss.backward()

        self.assertIsNotNone(logits.grad, 'Gradient should exist after backward pass')
        self.assertTrue(
            torch.isfinite(logits.grad).all().item(),
            'Gradient values should be finite (no NaN/Inf)',
        )


if __name__ == '__main__':
    unittest.main()
