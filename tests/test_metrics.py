"""
tests/test_metrics.py

Unit tests for evaluation metrics.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics import (
    compute_forgetting,
    compute_backward_transfer,
    compute_average_accuracy,
    evaluate_model,
    get_gpu_memory_mb,
)


class TestMetrics:
    def test_forgetting_positive(self):
        """Forgetting should be positive when accuracy drops."""
        f = compute_forgetting(acc_before=0.85, acc_after=0.70)
        assert f == pytest.approx(0.15, abs=1e-6)

    def test_forgetting_zero_when_same(self):
        f = compute_forgetting(acc_before=0.80, acc_after=0.80)
        assert f == pytest.approx(0.0)

    def test_forgetting_zero_when_improved(self):
        """If accuracy improved, forgetting = 0 (not negative)."""
        f = compute_forgetting(acc_before=0.70, acc_after=0.85)
        assert f == pytest.approx(0.0)

    def test_backward_transfer_negative_when_forgetting(self):
        bwt = compute_backward_transfer(acc_before=0.85, acc_after=0.70)
        assert bwt == pytest.approx(-0.15, abs=1e-6)

    def test_backward_transfer_positive_when_improved(self):
        bwt = compute_backward_transfer(acc_before=0.70, acc_after=0.85)
        assert bwt == pytest.approx(0.15, abs=1e-6)

    def test_average_accuracy_empty(self):
        assert compute_average_accuracy([]) == 0.0

    def test_average_accuracy_basic(self):
        avg = compute_average_accuracy([0.8, 0.6, 0.7])
        assert avg == pytest.approx(0.7, abs=1e-6)

    def test_gpu_memory_returns_float(self):
        mem = get_gpu_memory_mb()
        assert isinstance(mem, float)
        assert mem >= 0.0


class TestEvaluateModel:
    @pytest.fixture
    def tiny_model(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
            def forward(self, input_ids, attention_mask=None, **kwargs):
                logits = self.linear(input_ids)
                class Out:
                    pass
                out = Out()
                out.logits = logits
                return out
        return TinyModel()


    @pytest.fixture
    def loader(self):
        """Small loader with known correct answers."""
        n = 16
        x = torch.randn(n, 8)
        labels = torch.zeros(n, dtype=torch.long)  # all label 0
        ds = TensorDataset(x, labels)

        class CollateFn:
            def __call__(self, batch):
                inp, lab = zip(*batch)
                return {
                    "input_ids": torch.stack(inp),
                    "attention_mask": torch.ones(len(inp), 8, dtype=torch.long),
                    "labels": torch.stack(lab),
                }

        return DataLoader(ds, batch_size=4, collate_fn=CollateFn())

    def test_returns_loss_and_acc(self, tiny_model, loader):
        device = torch.device("cpu")
        loss, acc = evaluate_model(tiny_model, loader, device)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0.0
        assert 0.0 <= acc <= 1.0

    def test_perfect_model_high_acc(self, loader):
        """A model biased toward class 0 should get high accuracy on all-0 labels."""
        device = torch.device("cpu")

        class PerfectModel(nn.Module):
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                batch_size = input_ids.shape[0]
                logits = torch.zeros(batch_size, 4)
                logits[:, 0] = 100.0  # always predict class 0

                class Out:
                    pass
                out = Out()
                out.logits = logits
                return out

        loss, acc = evaluate_model(PerfectModel(), loader, device)
        assert acc == pytest.approx(1.0)
