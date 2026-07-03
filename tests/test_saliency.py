"""
tests/test_saliency.py

Unit tests for TAPSS saliency estimators.

These tests use a tiny 2-layer MLP instead of a real transformer to:
  - Run fast (no HuggingFace downloads needed in CI)
  - Test the scoring logic in isolation
  - Validate normalisation, ranking, and output shapes
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from saliency.base import ImportanceScores
from saliency.gradient import GradientMagnitudeEstimator
from saliency.activation import ActivationFrequencyEstimator
from saliency.perturbation import PerturbationSensitivityEstimator
from saliency.layer_contribution import LayerContributionEstimator
from saliency.tapss import TAPSSEstimator
from saliency.rankings import build_rankings, RankingResults


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_model():
    """A tiny 2-layer classification model for fast testing."""
    class TinyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.ModuleList([
                nn.Linear(32, 64),
                nn.Linear(64, 4),
            ])
            self.act = nn.ReLU()

        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            x = input_ids.float()
            x = self.act(self.layer[0](x))
            logits = self.layer[1](x)

            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits, labels)

            class Output:
                pass
            out = Output()
            out.logits = logits
            out.loss = loss
            out.hidden_states = None
            return out

    return TinyClassifier()


@pytest.fixture
def tiny_loader():
    """Small calibration DataLoader for testing."""
    n = 40
    inputs = torch.randint(0, 32, (n, 32))  # fake "input_ids"
    labels = torch.randint(0, 4, (n,))
    attention_mask = torch.ones(n, 32, dtype=torch.long)
    ds = TensorDataset(inputs, attention_mask, labels)

    class CollateFn:
        def __call__(self, batch):
            inp, mask, lab = zip(*batch)
            return {
                "input_ids": torch.stack(inp),
                "attention_mask": torch.stack(mask),
                "labels": torch.stack(lab),
            }

    return DataLoader(ds, batch_size=8, collate_fn=CollateFn())


@pytest.fixture
def device():
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# ImportanceScores tests
# ─────────────────────────────────────────────────────────────────────────────

class TestImportanceScores:
    def test_normalise_basic(self):
        raw = {"a": 1.0, "b": 2.0, "c": 3.0}
        normed = ImportanceScores.normalise(raw)
        assert normed["a"] == pytest.approx(0.0)
        assert normed["c"] == pytest.approx(1.0)
        assert 0.0 <= normed["b"] <= 1.0

    def test_normalise_uniform(self):
        raw = {"a": 0.5, "b": 0.5, "c": 0.5}
        normed = ImportanceScores.normalise(raw)
        # All uniform → all 0.5 (handled as edge case)
        assert all(v == pytest.approx(0.5) for v in normed.values())

    def test_normalise_empty(self):
        assert ImportanceScores.normalise({}) == {}

    def test_top_k(self):
        scores = ImportanceScores(
            scores={"a": 0.9, "b": 0.1, "c": 0.5},
            method="test",
        )
        top2 = scores.top_k(2)
        assert len(top2) == 2
        assert top2[0][0] == "a"
        assert top2[1][0] == "c"

    def test_combine_weights(self):
        s1 = {"a": 1.0, "b": 0.0}
        s2 = {"a": 0.0, "b": 1.0}
        combined = ImportanceScores.combine([s1, s2], weights=[0.5, 0.5])
        assert combined["a"] == pytest.approx(0.5)
        assert combined["b"] == pytest.approx(0.5)

    def test_layer_aggregated(self):
        scores = ImportanceScores(
            scores={
                "model.layer.0.weight": 0.8,
                "model.layer.0.bias": 0.4,
                "model.layer.1.weight": 0.2,
                "classifier.weight": 0.6,
            },
            method="test",
        )
        agg = scores.layer_aggregated()
        assert "layer_0" in agg
        assert "layer_1" in agg
        assert agg["layer_0"] == pytest.approx((0.8 + 0.4) / 2)


# ─────────────────────────────────────────────────────────────────────────────
# Estimator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientMagnitudeEstimator:
    def test_returns_importance_scores(self, tiny_model, tiny_loader, device):
        est = GradientMagnitudeEstimator(num_calibration_batches=3)
        scores = est.compute(tiny_model, tiny_loader, device)

        assert isinstance(scores, ImportanceScores)
        assert scores.method == "gradient_magnitude"
        assert len(scores.scores) > 0

    def test_scores_in_range(self, tiny_model, tiny_loader, device):
        est = GradientMagnitudeEstimator(num_calibration_batches=3)
        scores = est.compute(tiny_model, tiny_loader, device)

        for name, val in scores.scores.items():
            assert 0.0 <= val <= 1.0, f"Score out of range for {name}: {val}"

    def test_scores_not_all_equal(self, tiny_model, tiny_loader, device):
        est = GradientMagnitudeEstimator(num_calibration_batches=3)
        scores = est.compute(tiny_model, tiny_loader, device)
        values = list(scores.scores.values())
        # After normalisation, should have some variance (unless model is trivial)
        # Just check not all zero
        assert not all(v == 0.0 for v in values)


class TestActivationFrequencyEstimator:
    def test_returns_importance_scores(self, tiny_model, tiny_loader, device):
        est = ActivationFrequencyEstimator(num_calibration_batches=3)
        scores = est.compute(tiny_model, tiny_loader, device)
        assert isinstance(scores, ImportanceScores)
        assert len(scores.scores) > 0

    def test_scores_in_range(self, tiny_model, tiny_loader, device):
        est = ActivationFrequencyEstimator(num_calibration_batches=3)
        scores = est.compute(tiny_model, tiny_loader, device)
        for val in scores.scores.values():
            assert 0.0 <= val <= 1.0


class TestPerturbationSensitivityEstimator:
    def test_returns_importance_scores(self, tiny_model, tiny_loader, device):
        est = PerturbationSensitivityEstimator(
            perturbation_std=0.01, num_calibration_batches=2
        )
        scores = est.compute(tiny_model, tiny_loader, device)
        assert isinstance(scores, ImportanceScores)
        assert len(scores.scores) > 0

    def test_scores_in_range(self, tiny_model, tiny_loader, device):
        est = PerturbationSensitivityEstimator(num_calibration_batches=2)
        scores = est.compute(tiny_model, tiny_loader, device)
        for val in scores.scores.values():
            assert 0.0 <= val <= 1.0

    def test_model_restored(self, tiny_model, tiny_loader, device):
        """Parameter values must be identical before and after perturbation."""
        original_params = {
            n: p.data.clone() for n, p in tiny_model.named_parameters()
        }
        est = PerturbationSensitivityEstimator(num_calibration_batches=2)
        est.compute(tiny_model, tiny_loader, device)

        for name, param in tiny_model.named_parameters():
            assert torch.allclose(
                param.data, original_params[name]
            ), f"Parameter {name} was not restored after perturbation!"


class TestLayerContributionEstimator:
    def test_returns_importance_scores(self, tiny_model, tiny_loader, device):
        est = LayerContributionEstimator(num_calibration_batches=3)
        scores = est.compute(tiny_model, tiny_loader, device)
        assert isinstance(scores, ImportanceScores)

    def test_scores_in_range(self, tiny_model, tiny_loader, device):
        est = LayerContributionEstimator(num_calibration_batches=3)
        scores = est.compute(tiny_model, tiny_loader, device)
        for val in scores.scores.values():
            assert 0.0 <= val <= 1.0


class TestTAPSSEstimator:
    def test_combined_scores(self, tiny_model, tiny_loader, device):
        est = TAPSSEstimator(skip_perturbation=True)
        scores = est.compute(tiny_model, tiny_loader, device)
        assert isinstance(scores, ImportanceScores)
        assert scores.method == "tapss"
        assert len(scores.scores) > 0

    def test_scores_in_range(self, tiny_model, tiny_loader, device):
        est = TAPSSEstimator(skip_perturbation=True)
        scores = est.compute(tiny_model, tiny_loader, device)
        for val in scores.scores.values():
            assert 0.0 <= val <= 1.0, f"Out of range: {val}"

    def test_weight_renormalisation(self):
        """Weights should be renormalised to sum to 1."""
        weights = {"gradient": 2.0, "perturbation": 1.0, "activation": 1.0, "layer_contribution": 0.0}
        est = TAPSSEstimator(weights=weights, skip_perturbation=True)
        total = sum(est.weights.values())
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_component_scores_in_metadata(self, tiny_model, tiny_loader, device):
        est = TAPSSEstimator(skip_perturbation=True)
        scores = est.compute(tiny_model, tiny_loader, device)
        assert "component_scores" in scores.metadata
        assert "gradient" in scores.metadata["component_scores"]


# ─────────────────────────────────────────────────────────────────────────────
# Rankings tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRankings:
    def test_build_rankings(self, tiny_model):
        scores_dict = {
            name: float(i) / 10
            for i, (name, _) in enumerate(tiny_model.named_parameters())
        }
        scores = ImportanceScores(
            scores=ImportanceScores.normalise(scores_dict),
            method="test",
        )
        results = build_rankings(scores, tiny_model, protection_percent=25.0)
        assert isinstance(results, RankingResults)
        assert len(results.rankings) == len(scores_dict)
        # Ranks should be 1-indexed and contiguous
        ranks = sorted(r.rank for r in results.rankings)
        assert ranks == list(range(1, len(ranks) + 1))

    def test_top_k(self, tiny_model):
        scores_dict = {n: float(i) for i, (n, _) in enumerate(tiny_model.named_parameters())}
        scores = ImportanceScores(
            scores=ImportanceScores.normalise(scores_dict),
            method="test",
        )
        results = build_rankings(scores, tiny_model, protection_percent=50.0)
        top2 = results.top_k(2)
        assert len(top2) == 2
        assert top2[0].rank == 1

    def test_as_dataframe(self, tiny_model):
        scores_dict = {n: 0.5 for n, _ in tiny_model.named_parameters()}
        scores = ImportanceScores(scores=scores_dict, method="test")
        results = build_rankings(scores, tiny_model)
        df = results.as_dataframe()
        assert "param_name" in df.columns
        assert "importance_score" in df.columns
        assert "is_protected" in df.columns
        assert len(df) == len(scores_dict)
