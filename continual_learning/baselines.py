"""
continual_learning/baselines.py

Baseline methods for comparison with TAPSS.

All baselines implement the same interface as the TAPSS pipeline,
enabling fair, apples-to-apples comparison on the same task sequence.

Baselines
---------
1. VanillaLoRABaseline       — LoRA fine-tuning, no protection at all.
2. NaiveFinetuningBaseline   — Full fine-tuning, no LoRA, no protection.
3. RandomProtectionBaseline  — Randomly select parameters to protect.
                               Tests whether *any* protection helps (vs TAPSS).

EWC is in ewc.py (separate due to its complexity).
"""
from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from task_datasets.base import TaskDataset
from peft_modules.lora_trainer import LoRATrainer, TrainingHistory
from saliency.base import ImportanceScores

logger = logging.getLogger(__name__)


class BaselineMethod(ABC):
    """Abstract base class for all baseline methods."""

    def __init__(self, cfg: Any, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.trainer = LoRATrainer(cfg, device)
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def name(self) -> str:
        """Method name for reporting."""
        ...

    @abstractmethod
    def run_task_b(
        self,
        model: nn.Module,
        task_b: TaskDataset,
        task_a_state: dict[str, torch.Tensor],
        model_name: str,
    ) -> TrainingHistory:
        """
        Fine-tune on Task B according to this baseline's strategy.

        Parameters
        ----------
        model : nn.Module
            Model already trained on Task A.
        task_b : TaskDataset
            Task B dataset (new task).
        task_a_state : dict
            Frozen Task A parameter values.
        model_name : str
            Model identifier for LoRA target module detection.

        Returns
        -------
        TrainingHistory
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1: Vanilla LoRA
# ─────────────────────────────────────────────────────────────────────────────

class VanillaLoRABaseline(BaselineMethod):
    """
    Vanilla LoRA fine-tuning with no parameter protection.

    Applies LoRA adapters and trains all LoRA parameters freely.
    This is the most common PEFT approach and serves as the
    upper bound on Task B performance / lower bound on Task A forgetting.
    """

    @property
    def name(self) -> str:
        return "vanilla_lora"

    def run_task_b(
        self,
        model: nn.Module,
        task_b: TaskDataset,
        task_a_state: dict[str, torch.Tensor],
        model_name: str,
    ) -> TrainingHistory:
        """Apply LoRA and fine-tune without any protection."""
        self.logger.info("[VanillaLoRA] Applying LoRA adapters (no protection).")
        peft_model = self.trainer.apply_lora(model, model_name, task_b.num_labels)
        peft_model.to(self.device)

        history = self.trainer.train(
            peft_model,
            task_b.train_loader,
            task_b.val_loader,
            policy=None,
            task_a_state=None,
            run_name=f"{self.name}_task_b",
        )
        return history


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2: Naive Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

class NaiveFinetuningBaseline(BaselineMethod):
    """
    Naive full fine-tuning: no LoRA, no protection.

    All parameters (including the full pre-trained model) are updated
    freely on Task B. Maximally catastrophically forgetful — sets the
    lower bound on forgetting prevention.
    """

    @property
    def name(self) -> str:
        return "naive_finetuning"

    def run_task_b(
        self,
        model: nn.Module,
        task_b: TaskDataset,
        task_a_state: dict[str, torch.Tensor],
        model_name: str,
    ) -> TrainingHistory:
        """Full fine-tuning on Task B, no LoRA, no protection."""
        self.logger.info("[NaiveFinetune] Full fine-tuning (no LoRA, no protection).")

        # Ensure all parameters are trainable
        for param in model.parameters():
            param.requires_grad_(True)

        history = self.trainer.train(
            model,
            task_b.train_loader,
            task_b.val_loader,
            policy=None,
            task_a_state=None,
            run_name=f"{self.name}_task_b",
        )
        return history


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 3: Random Protection
# ─────────────────────────────────────────────────────────────────────────────

class RandomProtectionBaseline(BaselineMethod):
    """
    Random parameter protection baseline.

    Randomly selects the same number of parameters to protect as TAPSS
    (by default, the same topk_percent from the config), then applies
    the same FreezeTopK policy.

    This ablation tests whether the *selection criterion* matters:
    if random protection performs as well as TAPSS, the importance scores
    are not providing useful signal.
    """

    def __init__(self, cfg: Any, device: torch.device, seed: int = 42):
        super().__init__(cfg, device)
        self.seed = seed
        topk = cfg.protection.topk_percent if hasattr(cfg, "protection") else 20.0
        self.topk_percent = topk

    @property
    def name(self) -> str:
        return "random_protection"

    def run_task_b(
        self,
        model: nn.Module,
        task_b: TaskDataset,
        task_a_state: dict[str, torch.Tensor],
        model_name: str,
    ) -> TrainingHistory:
        """Apply LoRA + randomly chosen frozen parameters."""
        from peft_modules.protection import FreezeTopKPolicy

        self.logger.info(
            f"[RandomProtection] Randomly protecting {self.topk_percent:.1f}% of parameters."
        )

        # Build a fake importance score dict with random values
        rng = random.Random(self.seed)
        all_params = [
            name for name, p in model.named_parameters() if p.requires_grad
        ]
        random_scores = {name: rng.random() for name in all_params}
        random_importance = ImportanceScores(
            scores=ImportanceScores.normalise(random_scores),
            method="random",
        )

        # Apply LoRA
        peft_model = self.trainer.apply_lora(model, model_name, task_b.num_labels)
        peft_model.to(self.device)

        # Apply same freeze policy but with random scores
        policy = FreezeTopKPolicy(
            random_importance,
            cfg=self.cfg,
            topk_percent=self.topk_percent,
        )

        history = self.trainer.train(
            peft_model,
            task_b.train_loader,
            task_b.val_loader,
            policy=policy,
            task_a_state=task_a_state,
            run_name=f"{self.name}_task_b",
        )
        return history
