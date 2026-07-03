"""
tests/test_protection.py

Unit tests for TAPSS protection policies.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from saliency.base import ImportanceScores
from peft_modules.protection import (
    FreezeTopKPolicy,
    LRScalingPolicy,
    RegularizationPolicy,
    SoftProtectionPolicy,
    AdaptiveProtectionPolicy,
    build_protection_policy,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_model():
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
    )
    return model


@pytest.fixture
def importance_scores(simple_model):
    scores = {}
    for i, (name, _) in enumerate(simple_model.named_parameters()):
        scores[name] = float(i + 1) / len(list(simple_model.named_parameters()))
    return ImportanceScores(
        scores=ImportanceScores.normalise(scores),
        method="test",
    )


@pytest.fixture
def optimizer(simple_model):
    return optim.AdamW(simple_model.parameters(), lr=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# FreezeTopKPolicy
# ─────────────────────────────────────────────────────────────────────────────

class TestFreezeTopKPolicy:
    def test_freezes_top_params(self, simple_model, importance_scores, optimizer):
        policy = FreezeTopKPolicy(importance_scores, topk_percent=50.0, min_protected=1)
        policy.on_train_begin(simple_model, optimizer)

        all_params = list(simple_model.named_parameters())
        frozen_params = [name for name, p in all_params if not p.requires_grad]
        trainable_params = [name for name, p in all_params if p.requires_grad]

        # At least some should be frozen
        assert len(frozen_params) > 0
        # At least some should remain trainable
        assert len(trainable_params) > 0

    def test_freeze_percent(self, simple_model, importance_scores, optimizer):
        policy = FreezeTopKPolicy(importance_scores, topk_percent=100.0)
        policy.on_train_begin(simple_model, optimizer)

        all_params = list(simple_model.named_parameters())
        frozen = sum(1 for _, p in all_params if not p.requires_grad)
        assert frozen == len(all_params)

    def test_no_additional_loss(self, simple_model, importance_scores):
        policy = FreezeTopKPolicy(importance_scores)
        loss = policy.compute_additional_loss(simple_model)
        assert float(loss.item()) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# LRScalingPolicy
# ─────────────────────────────────────────────────────────────────────────────

class TestLRScalingPolicy:
    def test_param_groups_built(self, simple_model, importance_scores):
        policy = LRScalingPolicy(importance_scores, scale_factor=0.9, min_lr_ratio=0.05)
        groups = policy.build_param_groups(simple_model, base_lr=1e-3)

        assert len(groups) == len(list(simple_model.parameters()))

        # All LRs should be in [min_lr_ratio * base_lr, base_lr]
        min_lr = 0.05 * 1e-3
        for g in groups:
            assert min_lr <= g["lr"] <= 1e-3 + 1e-9

    def test_high_importance_gets_low_lr(self, simple_model, importance_scores):
        policy = LRScalingPolicy(importance_scores, scale_factor=0.9, min_lr_ratio=0.05)
        groups = policy.build_param_groups(simple_model, base_lr=1.0)

        lr_by_name = {g["name"]: g["lr"] for g in groups}
        sorted_by_importance = sorted(
            importance_scores.scores.items(), key=lambda x: x[1], reverse=True
        )
        # The most important param should have the lowest LR
        if len(sorted_by_importance) >= 2:
            most_important = sorted_by_importance[0][0]
            least_important = sorted_by_importance[-1][0]
            if most_important in lr_by_name and least_important in lr_by_name:
                assert lr_by_name[most_important] <= lr_by_name[least_important]


# ─────────────────────────────────────────────────────────────────────────────
# RegularizationPolicy
# ─────────────────────────────────────────────────────────────────────────────

class TestRegularizationPolicy:
    def test_zero_loss_when_at_anchor(self, simple_model, importance_scores, optimizer):
        """If model is at Task A state, regularisation loss should be ~0."""
        policy = RegularizationPolicy(importance_scores, lambda_reg=1.0, topk_percent=50.0)
        policy.on_train_begin(simple_model, optimizer)

        # Task A state = current params
        task_a_state = {n: p.data.clone() for n, p in simple_model.named_parameters()}
        loss = policy.compute_additional_loss(simple_model, task_a_state)
        assert float(loss.item()) == pytest.approx(0.0, abs=1e-6)

    def test_nonzero_loss_when_diverged(self, simple_model, importance_scores, optimizer):
        """If params have changed, regularisation loss should be > 0."""
        policy = RegularizationPolicy(importance_scores, lambda_reg=1.0, topk_percent=100.0)
        policy.on_train_begin(simple_model, optimizer)

        task_a_state = {n: torch.zeros_like(p.data) for n, p in simple_model.named_parameters()}
        loss = policy.compute_additional_loss(simple_model, task_a_state)
        assert float(loss.item()) > 0.0

    def test_none_task_a_returns_zero(self, simple_model, importance_scores, optimizer):
        policy = RegularizationPolicy(importance_scores)
        policy.on_train_begin(simple_model, optimizer)
        loss = policy.compute_additional_loss(simple_model, task_a_state=None)
        assert float(loss.item()) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# SoftProtectionPolicy
# ─────────────────────────────────────────────────────────────────────────────

class TestSoftProtectionPolicy:
    def test_multipliers_in_range(self, simple_model, importance_scores, optimizer):
        policy = SoftProtectionPolicy(importance_scores, strength=0.8, min_grad_ratio=0.05)
        policy.on_train_begin(simple_model, optimizer)

        for name, mult in policy._grad_multipliers.items():
            assert 0.05 <= mult <= 1.0, f"Multiplier {mult} out of range for {name}"

    def test_dampening_reduces_grads(self, simple_model, importance_scores, optimizer):
        """Gradients of high-importance params should be reduced."""
        policy = SoftProtectionPolicy(importance_scores, strength=0.8, min_grad_ratio=0.05)
        policy.on_train_begin(simple_model, optimizer)

        # Create fake gradients
        x = torch.randn(4, 16)
        y = torch.randint(0, 4, (4,))
        logits = simple_model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        # Capture pre-dampening grads
        pre_grad_norms = {
            name: param.grad.norm().item()
            for name, param in simple_model.named_parameters()
            if param.grad is not None
        }

        policy.on_step_end(simple_model, optimizer, step=0)

        # Post-dampening norms should be <= pre-dampening norms
        for name, param in simple_model.named_parameters():
            if param.grad is not None and name in pre_grad_norms:
                post_norm = param.grad.norm().item()
                assert post_norm <= pre_grad_norms[name] + 1e-7, \
                    f"Grad increased for {name}: {pre_grad_norms[name]} → {post_norm}"


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildProtectionPolicy:
    def test_builds_all_policies(self, importance_scores):
        policy_names = [
            "freeze_topk", "lr_scaling", "regularization", "soft_protection", "adaptive", "none"
        ]
        for name in policy_names:
            policy = build_protection_policy(name, importance_scores)
            assert policy is not None
            assert hasattr(policy, "name")

    def test_unknown_policy_raises(self, importance_scores):
        with pytest.raises(ValueError, match="Unknown protection policy"):
            build_protection_policy("does_not_exist", importance_scores)
