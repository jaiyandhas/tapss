"""
continual_learning/ewc.py

Elastic Weight Consolidation (EWC) Baseline.

EWC (Kirkpatrick et al., 2017) is a landmark continual learning method.
It prevents catastrophic forgetting by penalising changes to parameters
that were important for Task A, using the Fisher Information Matrix (FIM)
as the importance measure.

This implementation uses the *empirical* (diagonal) Fisher, computed as:
  F_i = E[(∂L/∂θ_i)²]

The EWC penalty term is:
  L_EWC = λ/2 × Σ_i F_i × (θ_i - θ_i^A)²

Differences from TAPSS
-----------------------
- EWC uses squared gradients (Fisher ≈ E[grad²]) as importance.
- TAPSS combines gradient magnitude, perturbation, activation, and layer.
- EWC importance is computed on the *full* training set (or a subset);
  TAPSS uses a small calibration set.
- EWC applies regularisation only; TAPSS supports 5 protection strategies.

Approximation notes
-------------------
Computing the full Fisher matrix is O(n²) in parameters. We use the
diagonal Fisher approximation, which is O(n) and sufficient for comparison.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from task_datasets.base import TaskDataset
from peft_modules.lora_trainer import LoRATrainer, TrainingHistory
from continual_learning.baselines import BaselineMethod
from saliency.base import ImportanceScores

logger = logging.getLogger(__name__)


class EWCBaseline(BaselineMethod):
    """
    Elastic Weight Consolidation continual learning baseline.

    Computes the diagonal Fisher Information Matrix on Task A data,
    then adds an EWC regularisation term during Task B training.

    Parameters
    ----------
    cfg : DictConfig
        Experiment config.
    device : torch.device
    ewc_lambda : float
        EWC regularisation coefficient (default: 100.0).
    fisher_batches : int
        Number of batches to use for Fisher estimation (default: 50).
    """

    def __init__(
        self,
        cfg: Any,
        device: torch.device,
        ewc_lambda: float = 100.0,
        fisher_batches: int = 50,
    ):
        super().__init__(cfg, device)
        self.ewc_lambda = ewc_lambda
        self.fisher_batches = fisher_batches
        self._fisher: Optional[dict[str, torch.Tensor]] = None
        self._task_a_params: Optional[dict[str, torch.Tensor]] = None

    @property
    def name(self) -> str:
        return "ewc"

    def compute_fisher(
        self,
        model: nn.Module,
        task_a_loader: DataLoader,
    ) -> dict[str, torch.Tensor]:
        """
        Compute diagonal empirical Fisher Information Matrix.

        F_i = (1/N) Σ_n (∂L_n/∂θ_i)²

        Parameters
        ----------
        model : nn.Module
            Model trained on Task A.
        task_a_loader : DataLoader
            Task A data for Fisher estimation.

        Returns
        -------
        dict[str, torch.Tensor]
            Per-parameter Fisher diagonal values.
        """
        self.logger.info(
            f"[EWC] Computing Fisher Information Matrix "
            f"({self.fisher_batches} batches)..."
        )
        t0 = time.time()

        model.eval()

        # Initialise Fisher accumulators
        fisher: dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        num_batches = 0
        for i, batch in enumerate(task_a_loader):
            if i >= self.fisher_batches:
                break

            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            model.zero_grad()

            outputs = model(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
            )

            # Sample from predicted distribution (more principled than using true labels)
            # This is the "online" EWC Fisher approximation
            log_probs = F.log_softmax(outputs.logits, dim=-1)
            sampled_labels = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)
            loss = F.nll_loss(log_probs, sampled_labels)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)

            num_batches += 1

        # Average over batches
        for name in fisher:
            fisher[name] /= max(num_batches, 1)

        model.zero_grad()

        elapsed = time.time() - t0
        self.logger.info(
            f"[EWC] Fisher computed in {elapsed:.1f}s. "
            f"Mean Fisher magnitude: "
            f"{sum(f.mean().item() for f in fisher.values()) / len(fisher):.6f}"
        )

        return fisher

    def prepare(self, model: nn.Module, task_a_loader: DataLoader) -> None:
        """
        Prepare EWC for Task B training.

        Must be called after Task A training and before Task B training.

        Parameters
        ----------
        model : nn.Module
            Model trained on Task A.
        task_a_loader : DataLoader
            Task A data for Fisher estimation.
        """
        self._fisher = self.compute_fisher(model, task_a_loader)
        self._task_a_params = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.logger.info("[EWC] Prepared: Fisher + Task A parameters cached.")

    def _ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute the EWC regularisation loss."""
        if self._fisher is None or self._task_a_params is None:
            return torch.tensor(0.0)

        device = next(model.parameters()).device
        ewc_loss = torch.tensor(0.0, device=device)
        count = 0

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self._fisher or name not in self._task_a_params:
                continue

            fisher_i = self._fisher[name].to(device)
            anchor_i = self._task_a_params[name].to(device)

            ewc_loss = ewc_loss + (fisher_i * (param - anchor_i).pow(2)).sum()
            count += 1

        return (self.ewc_lambda / 2.0) * ewc_loss / max(count, 1)

    def run_task_b(
        self,
        model: nn.Module,
        task_b: TaskDataset,
        task_a_state: dict[str, torch.Tensor],
        model_name: str,
    ) -> TrainingHistory:
        """
        Fine-tune on Task B with EWC regularisation.

        Note: EWC.prepare() must have been called before this method.
        If not, it will train without EWC protection.
        """
        if self._fisher is None:
            self.logger.warning(
                "[EWC] Fisher not computed. Call prepare() first. "
                "Running without EWC protection."
            )

        self.logger.info(
            f"[EWC] Task B fine-tuning with EWC (lambda={self.ewc_lambda})."
        )

        # Apply LoRA
        peft_model = self.trainer.apply_lora(model, model_name, task_b.num_labels)
        peft_model.to(self.device)

        # Build a lightweight wrapper that injects EWC loss
        ewc_policy = _EWCProtectionPolicy(self)

        history = self.trainer.train(
            peft_model,
            task_b.train_loader,
            task_b.val_loader,
            policy=ewc_policy,
            task_a_state=task_a_state,
            run_name=f"{self.name}_task_b",
        )
        return history

    def fisher_as_importance_scores(self) -> Optional[ImportanceScores]:
        """Convert Fisher diagonal to ImportanceScores for visualisation."""
        if self._fisher is None:
            return None
        raw = {name: float(f.mean().item()) for name, f in self._fisher.items()}
        return ImportanceScores(
            scores=ImportanceScores.normalise(raw),
            method="ewc_fisher",
            metadata={"ewc_lambda": self.ewc_lambda},
        )


class _EWCProtectionPolicy:
    """
    Minimal ProtectionPolicy-compatible wrapper for EWC loss injection.
    Used internally by EWCBaseline to plug into the LoRATrainer loop.
    """

    def __init__(self, ewc: EWCBaseline):
        self._ewc = ewc

    @property
    def name(self) -> str:
        return "ewc"

    def on_train_begin(self, model, optimizer) -> None: ...
    def on_step_begin(self, model, optimizer, step) -> None: ...
    def on_step_end(self, model, optimizer, step) -> None: ...
    def on_epoch_begin(self, epoch, total_epochs) -> None: ...

    def compute_additional_loss(self, model, task_a_state=None) -> torch.Tensor:
        return self._ewc._ewc_loss(model)

    def protection_summary(self) -> dict:
        return {"policy": "ewc", "lambda": self._ewc.ewc_lambda}
