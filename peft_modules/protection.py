"""
peft_modules/protection.py

Protection Policies for TAPSS continual learning.

Five distinct strategies for using parameter importance scores
to protect Task A knowledge during Task B LoRA fine-tuning.

Policy A — FreezeTopK:     Hard-freeze top-K% most important parameters.
Policy B — LRScaling:      Reduce learning rate for important parameters.
Policy C — Regularization: Add importance-weighted L2 penalty.
Policy D — SoftProtection: Dampen gradients of important parameters.
Policy E — Adaptive:       Combines B + C + D with schedule-based relaxation.

All policies share a common interface and are selected from YAML config.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from saliency.base import ImportanceScores

logger = logging.getLogger(__name__)


class ProtectionPolicy(ABC):
    """
    Abstract base class for parameter protection policies.

    A policy receives importance scores computed by a TAPSS estimator
    and uses them to constrain optimisation during Task B fine-tuning.
    """

    def __init__(self, importance_scores: ImportanceScores, cfg: Any = None):
        self.importance_scores = importance_scores
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def name(self) -> str:
        """Policy name for logging and reporting."""
        ...

    def on_train_begin(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        """Called once before Task B training begins."""
        pass

    def on_step_begin(
        self, model: nn.Module, optimizer: optim.Optimizer, step: int
    ) -> None:
        """Called before each optimiser step."""
        pass

    def on_step_end(
        self, model: nn.Module, optimizer: optim.Optimizer, step: int
    ) -> None:
        """Called after each optimiser step (and after backward pass)."""
        pass

    def compute_additional_loss(
        self,
        model: nn.Module,
        task_a_state: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute any additional loss term to add to the main task loss.
        Default: no additional loss.
        """
        return torch.tensor(0.0, requires_grad=False)

    def on_epoch_begin(self, epoch: int, total_epochs: int) -> None:
        """Called at the start of each epoch."""
        pass

    def protection_summary(self) -> dict:
        """Return a summary of this policy's configuration."""
        return {"policy": self.name}


# ─────────────────────────────────────────────────────────────────────────────
# Policy A: Freeze Top-K
# ─────────────────────────────────────────────────────────────────────────────

class FreezeTopKPolicy(ProtectionPolicy):
    """
    Policy A — Hard-freeze the top-K% most important parameters.

    Important parameters are excluded from the optimiser entirely.
    Only LoRA parameters and unprotected base model parameters are trained.

    This is the strictest protection: important weights cannot change at all.
    The trade-off is reduced Task B expressiveness.
    """

    def __init__(
        self,
        importance_scores: ImportanceScores,
        cfg: Any = None,
        topk_percent: float = 20.0,
        min_protected: int = 100,
    ):
        super().__init__(importance_scores, cfg)
        if cfg is not None and hasattr(cfg, "protection"):
            self.topk_percent = cfg.protection.get("topk_percent", topk_percent)
            self.min_protected = cfg.protection.get("min_protected", min_protected)
        else:
            self.topk_percent = topk_percent
            self.min_protected = min_protected

        self._protected_names: set[str] = set()

    @property
    def name(self) -> str:
        return "freeze_topk"

    def on_train_begin(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        """Freeze top-K% parameters before training starts."""
        items = sorted(
            self.importance_scores.scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        k = max(self.min_protected, int(len(items) * self.topk_percent / 100.0))
        self._protected_names = {name for name, _ in items[:k]}

        frozen_count = 0
        for name, param in model.named_parameters():
            if name in self._protected_names:
                param.requires_grad_(False)
                frozen_count += 1

        self.logger.info(
            f"[FreezeTopK] Froze {frozen_count} parameters "
            f"(top {self.topk_percent:.1f}% by TAPSS score)."
        )

    def protection_summary(self) -> dict:
        return {
            "policy": self.name,
            "topk_percent": self.topk_percent,
            "num_protected": len(self._protected_names),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Policy B: Learning Rate Scaling
# ─────────────────────────────────────────────────────────────────────────────

class LRScalingPolicy(ProtectionPolicy):
    """
    Policy B — Reduce learning rate proportionally to parameter importance.

    Creates per-parameter-group optimiser configuration where:
      lr_effective = base_lr × max(min_lr_ratio, 1 - importance × scale_factor)

    High-importance parameters learn slowly; low-importance parameters
    learn at the full rate. No parameters are frozen.
    """

    def __init__(
        self,
        importance_scores: ImportanceScores,
        cfg: Any = None,
        scale_factor: float = 0.9,
        min_lr_ratio: float = 0.05,
    ):
        super().__init__(importance_scores, cfg)
        if cfg is not None and hasattr(cfg, "protection"):
            self.scale_factor = cfg.protection.get("scale_factor", scale_factor)
            self.min_lr_ratio = cfg.protection.get("min_lr_ratio", min_lr_ratio)
        else:
            self.scale_factor = scale_factor
            self.min_lr_ratio = min_lr_ratio

    @property
    def name(self) -> str:
        return "lr_scaling"

    def build_param_groups(
        self, model: nn.Module, base_lr: float
    ) -> list[dict]:
        """
        Build optimiser parameter groups with per-parameter LR scaling.

        Parameters
        ----------
        model : nn.Module
        base_lr : float
            The global base learning rate.

        Returns
        -------
        list[dict] — parameter groups for torch.optim
        """
        groups: list[dict] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            importance = self.importance_scores.scores.get(name, 0.0)
            lr_ratio = max(self.min_lr_ratio, 1.0 - importance * self.scale_factor)
            groups.append({"params": [param], "lr": base_lr * lr_ratio, "name": name})

        self.logger.info(
            f"[LRScaling] Built {len(groups)} parameter groups. "
            f"LR range: [{self.min_lr_ratio * base_lr:.2e}, {base_lr:.2e}]"
        )
        return groups

    def protection_summary(self) -> dict:
        return {
            "policy": self.name,
            "scale_factor": self.scale_factor,
            "min_lr_ratio": self.min_lr_ratio,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Policy C: Regularization
# ─────────────────────────────────────────────────────────────────────────────

class RegularizationPolicy(ProtectionPolicy):
    """
    Policy C — Importance-weighted L2 regularisation toward Task A weights.

    Adds an auxiliary loss term:
      L_reg = λ × Σ_i [importance_i × ||θ_i - θ_i^A||²]

    This is the TAPSS version of Elastic Weight Consolidation (EWC),
    with TAPSS scores as the importance weights instead of Fisher information.

    Parameters anchor important weights close to their Task A values
    without completely freezing them.
    """

    def __init__(
        self,
        importance_scores: ImportanceScores,
        cfg: Any = None,
        lambda_reg: float = 1.0,
        topk_percent: float = 30.0,
    ):
        super().__init__(importance_scores, cfg)
        if cfg is not None and hasattr(cfg, "protection"):
            self.lambda_reg = cfg.protection.get("lambda_reg", lambda_reg)
            self.topk_percent = cfg.protection.get("topk_percent", topk_percent)
        else:
            self.lambda_reg = lambda_reg
            self.topk_percent = topk_percent

        self._protected_names: set[str] = set()
        self._importance_tensor: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "regularization"

    def on_train_begin(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        items = sorted(
            self.importance_scores.scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        k = max(1, int(len(items) * self.topk_percent / 100.0))
        self._protected_names = {name for name, _ in items[:k]}
        self._importance_tensor = {name: score for name, score in items[:k]}
        self.logger.info(
            f"[Regularization] Will regularise {len(self._protected_names)} "
            f"important parameters (top {self.topk_percent:.1f}%)."
        )

    def compute_additional_loss(
        self,
        model: nn.Module,
        task_a_state: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute importance-weighted L2 distance from Task A weights.

        Returns
        -------
        torch.Tensor
            Scalar regularisation loss term.
        """
        if task_a_state is None:
            return torch.tensor(0.0)

        reg_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        count = 0

        for name, param in model.named_parameters():
            if name not in self._protected_names:
                continue
            if name not in task_a_state:
                continue

            importance = self._importance_tensor.get(name, 0.0)
            anchor = task_a_state[name].to(param.device)
            reg_loss = reg_loss + importance * (param - anchor).pow(2).sum()
            count += 1

        return self.lambda_reg * reg_loss / max(count, 1)

    def protection_summary(self) -> dict:
        return {
            "policy": self.name,
            "lambda_reg": self.lambda_reg,
            "topk_percent": self.topk_percent,
            "num_regularized": len(self._protected_names),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Policy D: Soft Protection (Gradient Dampening)
# ─────────────────────────────────────────────────────────────────────────────

class SoftProtectionPolicy(ProtectionPolicy):
    """
    Policy D — Dampen gradients of important parameters after backward pass.

    For each parameter with importance score s:
      grad_effective = grad × max(min_grad_ratio, 1 - s × strength)

    High-importance parameters have their gradients dampened —
    they can still change, but at a much slower rate.
    This is softer than freezing and allows the model to adapt
    to Task B while resisting large deviations.
    """

    def __init__(
        self,
        importance_scores: ImportanceScores,
        cfg: Any = None,
        strength: float = 0.8,
        min_grad_ratio: float = 0.05,
    ):
        super().__init__(importance_scores, cfg)
        if cfg is not None and hasattr(cfg, "protection"):
            self.strength = cfg.protection.get("strength", strength)
            self.min_grad_ratio = cfg.protection.get("min_grad_ratio", min_grad_ratio)
        else:
            self.strength = strength
            self.min_grad_ratio = min_grad_ratio

        # Precompute gradient multipliers
        self._grad_multipliers: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "soft_protection"

    def on_train_begin(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        """Precompute gradient dampening multipliers."""
        self._grad_multipliers = {}
        for name in self.importance_scores.scores:
            importance = self.importance_scores.scores[name]
            multiplier = max(self.min_grad_ratio, 1.0 - importance * self.strength)
            self._grad_multipliers[name] = multiplier

        self.logger.info(
            f"[SoftProtection] Gradient dampening configured for "
            f"{len(self._grad_multipliers)} parameters. "
            f"Multiplier range: [{self.min_grad_ratio:.2f}, 1.0]"
        )

    def on_step_end(
        self, model: nn.Module, optimizer: optim.Optimizer, step: int
    ) -> None:
        """Apply gradient dampening after backward, before optimiser step."""
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            mult = self._grad_multipliers.get(name, 1.0)
            if mult < 1.0:
                param.grad.data.mul_(mult)

    def protection_summary(self) -> dict:
        return {
            "policy": self.name,
            "strength": self.strength,
            "min_grad_ratio": self.min_grad_ratio,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Policy E: Adaptive (Combined)
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveProtectionPolicy(ProtectionPolicy):
    """
    Policy E — Adaptive combination of regularisation and soft protection.

    Combines:
    - Importance-weighted L2 regularisation (like Policy C)
    - Gradient dampening (like Policy D)
    - Scheduled relaxation: protection decreases as training progresses,
      allowing the model to adapt more freely to Task B over time.

    The intuition: early in Task B training, protect aggressively to
    prevent catastrophic forgetting. Later, relax protection to allow
    better Task B performance.
    """

    def __init__(
        self,
        importance_scores: ImportanceScores,
        cfg: Any = None,
        initial_topk_percent: float = 10.0,
        final_topk_percent: float = 25.0,
        regularization_lambda: float = 0.5,
        grad_dampening_strength: float = 0.6,
        warmup_epochs: int = 1,
    ):
        super().__init__(importance_scores, cfg)
        if cfg is not None and hasattr(cfg, "protection"):
            self.initial_topk_percent = cfg.protection.get(
                "initial_topk_percent", initial_topk_percent
            )
            self.final_topk_percent = cfg.protection.get(
                "final_topk_percent", final_topk_percent
            )
            self.regularization_lambda = cfg.protection.get(
                "regularization_lambda", regularization_lambda
            )
            self.grad_dampening_strength = cfg.protection.get(
                "grad_dampening_strength", grad_dampening_strength
            )
            self.warmup_epochs = cfg.protection.get("warmup_epochs", warmup_epochs)
        else:
            self.initial_topk_percent = initial_topk_percent
            self.final_topk_percent = final_topk_percent
            self.regularization_lambda = regularization_lambda
            self.grad_dampening_strength = grad_dampening_strength
            self.warmup_epochs = warmup_epochs

        self._current_strength: float = grad_dampening_strength
        self._current_lambda: float = regularization_lambda
        self._protected_names: set[str] = set()
        self._grad_multipliers: dict[str, float] = {}
        self._total_epochs: int = 1

    @property
    def name(self) -> str:
        return "adaptive"

    def on_train_begin(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        """Initialise with maximum protection."""
        self._update_protection(0, self._total_epochs)
        self.logger.info(
            f"[Adaptive] Initialised. Lambda={self._current_lambda:.3f}, "
            f"strength={self._current_strength:.3f}"
        )

    def on_epoch_begin(self, epoch: int, total_epochs: int) -> None:
        """Relax protection as training progresses."""
        self._total_epochs = total_epochs
        self._update_protection(epoch, total_epochs)

    def _update_protection(self, epoch: int, total_epochs: int) -> None:
        """Linearly interpolate protection from initial → final."""
        if total_epochs <= 1:
            progress = 1.0
        else:
            progress = min(1.0, epoch / (total_epochs - 1))

        # Relax: move from initial (strict) to final (relaxed) over training
        topk = self.initial_topk_percent + progress * (
            self.final_topk_percent - self.initial_topk_percent
        )

        # Lambda decreases (less regularisation) as training progresses
        self._current_lambda = self.regularization_lambda * (1.0 - 0.5 * progress)
        # Strength decreases (less dampening) as training progresses
        self._current_strength = self.grad_dampening_strength * (1.0 - 0.4 * progress)

        # Update protected set
        items = sorted(
            self.importance_scores.scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        k = max(1, int(len(items) * topk / 100.0))
        self._protected_names = {name for name, _ in items[:k]}

        # Update gradient multipliers
        self._grad_multipliers = {
            name: max(0.05, 1.0 - self.importance_scores.scores.get(name, 0.0) * self._current_strength)
            for name in self.importance_scores.scores
        }

        self.logger.debug(
            f"[Adaptive] Epoch {epoch}/{total_epochs}: "
            f"topk={topk:.1f}%, lambda={self._current_lambda:.3f}, "
            f"strength={self._current_strength:.3f}"
        )

    def on_step_end(
        self, model: nn.Module, optimizer: optim.Optimizer, step: int
    ) -> None:
        """Apply gradient dampening."""
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            mult = self._grad_multipliers.get(name, 1.0)
            if mult < 1.0:
                param.grad.data.mul_(mult)

    def compute_additional_loss(
        self,
        model: nn.Module,
        task_a_state: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Importance-weighted L2 regularisation toward Task A weights."""
        if task_a_state is None or self._current_lambda == 0:
            return torch.tensor(0.0)

        device = next(model.parameters()).device
        reg_loss = torch.tensor(0.0, device=device)
        count = 0

        for name, param in model.named_parameters():
            if name not in self._protected_names:
                continue
            if name not in task_a_state:
                continue
            importance = self.importance_scores.scores.get(name, 0.0)
            anchor = task_a_state[name].to(device)
            reg_loss = reg_loss + importance * (param - anchor).pow(2).sum()
            count += 1

        return self._current_lambda * reg_loss / max(count, 1)

    def protection_summary(self) -> dict:
        return {
            "policy": self.name,
            "initial_topk_percent": self.initial_topk_percent,
            "final_topk_percent": self.final_topk_percent,
            "regularization_lambda": self.regularization_lambda,
            "grad_dampening_strength": self.grad_dampening_strength,
            "warmup_epochs": self.warmup_epochs,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_protection_policy(
    policy_name: str,
    importance_scores: ImportanceScores,
    cfg: Any = None,
    **kwargs,
) -> ProtectionPolicy:
    """
    Factory function: instantiate a protection policy by name.

    Parameters
    ----------
    policy_name : str
        One of: "freeze_topk", "lr_scaling", "regularization",
                "soft_protection", "adaptive", "none".
    importance_scores : ImportanceScores
        Pre-computed importance scores from a TAPSS estimator.
    cfg : Any
        Hydra config (optional; used for parameter overrides).
    **kwargs :
        Additional keyword arguments forwarded to the policy constructor.

    Returns
    -------
    ProtectionPolicy
    """
    registry: dict[str, type] = {
        "freeze_topk": FreezeTopKPolicy,
        "lr_scaling": LRScalingPolicy,
        "regularization": RegularizationPolicy,
        "soft_protection": SoftProtectionPolicy,
        "soft": SoftProtectionPolicy,
        "adaptive": AdaptiveProtectionPolicy,
    }

    if policy_name == "none":
        # Return a no-op policy
        class NoOpPolicy(ProtectionPolicy):
            @property
            def name(self):
                return "none"
        return NoOpPolicy(importance_scores, cfg)

    if policy_name not in registry:
        raise ValueError(
            f"Unknown protection policy: {policy_name!r}. "
            f"Available: {list(registry.keys())}"
        )

    cls = registry[policy_name]
    return cls(importance_scores, cfg, **kwargs)
