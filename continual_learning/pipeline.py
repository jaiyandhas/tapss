"""
continual_learning/pipeline.py

TAPSS Continual Learning Pipeline.

Orchestrates the full experimental sequence:

  Step 1: Train on Task A (classification)
           → Save Task A checkpoint + accuracy
  Step 2: Compute TAPSS importance scores on Task A data
           → Rank parameters + apply protection policy
  Step 3: Fine-tune on Task B with LoRA + protection
           → Save Task B checkpoint + accuracy
  Step 4: Re-evaluate on Task A
           → Measure catastrophic forgetting

This pipeline is shared between TAPSS and all baseline methods,
ensuring a fair comparison on the same data splits and model.
"""
from __future__ import annotations

import copy
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

from task_datasets.base import TaskDataset
from evaluation.metrics import (
    CLMetrics,
    compute_forgetting,
    compute_backward_transfer,
    compute_average_accuracy,
)
from peft_modules.lora_trainer import LoRATrainer, TrainingHistory
from peft_modules.protection import ProtectionPolicy
from saliency.base import ImportanceScores
from saliency.rankings import RankingResults

logger = logging.getLogger(__name__)


@dataclass
class CLResult:
    """
    Complete results from a single TAPSS continual learning experiment.

    Contains all metrics, histories, and metadata needed for comparison
    and visualisation.
    """

    method_name: str
    model_name: str

    # Task A
    task_a_name: str
    task_a_train_history: Optional[TrainingHistory]
    task_a_pre_accuracy: float   # Accuracy on Task A before Task B fine-tuning
    task_a_post_accuracy: float  # Accuracy on Task A AFTER Task B fine-tuning

    # Task B
    task_b_name: str
    task_b_train_history: Optional[TrainingHistory]
    task_b_accuracy: float

    # Importance scores (TAPSS methods only)
    importance_scores: Optional[ImportanceScores] = None
    rankings: Optional[RankingResults] = None

    # Derived CL metrics
    forgetting: float = 0.0           # task_a_pre - task_a_post
    backward_transfer: float = 0.0
    average_accuracy: float = 0.0

    # Resource usage
    total_time_seconds: float = 0.0
    num_trainable_params_task_b: int = 0

    # Extra metadata
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.forgetting = compute_forgetting(
            self.task_a_pre_accuracy, self.task_a_post_accuracy
        )
        self.backward_transfer = compute_backward_transfer(
            self.task_a_pre_accuracy, self.task_a_post_accuracy
        )
        self.average_accuracy = compute_average_accuracy(
            [self.task_a_post_accuracy, self.task_b_accuracy]
        )

    def to_dict(self) -> dict:
        return {
            "method": self.method_name,
            "model": self.model_name,
            "task_a": self.task_a_name,
            "task_b": self.task_b_name,
            "task_a_pre_acc": self.task_a_pre_accuracy,
            "task_a_post_acc": self.task_a_post_accuracy,
            "task_b_acc": self.task_b_accuracy,
            "forgetting": self.forgetting,
            "backward_transfer": self.backward_transfer,
            "average_accuracy": self.average_accuracy,
            "total_time_s": self.total_time_seconds,
            "trainable_params": self.num_trainable_params_task_b,
        }


class ContinualLearningPipeline:
    """
    TAPSS Continual Learning Pipeline.

    Shared infrastructure for all methods (TAPSS + baselines).
    Each method provides its own importance scores and protection policy;
    the pipeline handles training, evaluation, and metric collection.

    Parameters
    ----------
    cfg : DictConfig
        Hydra experiment configuration.
    device : torch.device
        Computation device.
    """

    def __init__(self, cfg: Any, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.trainer = LoRATrainer(cfg, device)
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(
        self,
        model_factory,           # Callable[[], (nn.Module, tokenizer)]
        task_a: TaskDataset,
        task_b: TaskDataset,
        method_name: str,
        importance_scores: Optional[ImportanceScores] = None,
        protection_policy: Optional[ProtectionPolicy] = None,
        rankings: Optional[RankingResults] = None,
        apply_lora: bool = True,
        skip_task_a_training: bool = False,
        pretrained_task_a_checkpoint: Optional[str] = None,
    ) -> CLResult:
        """
        Execute the full CL experiment sequence.

        Parameters
        ----------
        model_factory : Callable
            Returns a fresh (model, tokenizer) pair. Called once per method.
        task_a, task_b : TaskDataset
            Task A (original) and Task B (new task) datasets.
        method_name : str
            Human-readable name for this experimental run.
        importance_scores : ImportanceScores | None
            Pre-computed parameter importance. If None, no protection.
        protection_policy : ProtectionPolicy | None
            Protection policy to apply during Task B training.
        rankings : RankingResults | None
            Pre-computed parameter rankings (for saving/reporting).
        apply_lora : bool
            Whether to apply LoRA adapters for Task B training.
        skip_task_a_training : bool
            If True, skip Task A training and load from checkpoint.
        pretrained_task_a_checkpoint : str | None
            Path to a saved Task A checkpoint (used if skip_task_a_training=True).

        Returns
        -------
        CLResult
        """
        t_total = time.time()
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"[Pipeline] Running: {method_name}")
        self.logger.info(f"[Pipeline] Task A: {task_a.name} → Task B: {task_b.name}")
        self.logger.info(f"{'='*60}")

        # ── Step 1: Task A Training ──
        model, tokenizer = model_factory()
        model.to(self.device)

        if skip_task_a_training and pretrained_task_a_checkpoint:
            self.logger.info(f"[Pipeline] Loading Task A checkpoint: {pretrained_task_a_checkpoint}")
            state = torch.load(pretrained_task_a_checkpoint, map_location=self.device)
            model.load_state_dict(state["model_state_dict"], strict=False)
            task_a_history = None
        else:
            self.logger.info("[Pipeline] Step 1: Training on Task A...")
            task_a_history = self.trainer.train(
                model,
                task_a.train_loader,
                task_a.val_loader,
                policy=None,
                run_name=f"{method_name}_task_a",
            )

        # Evaluate on Task A (this is the "pre" accuracy)
        task_a_loss, task_a_pre_acc = self.trainer.evaluate(model, task_a.test_loader)
        self.logger.info(
            f"[Pipeline] Task A accuracy (pre-B): {task_a_pre_acc:.4f} (loss={task_a_loss:.4f})"
        )

        # Snapshot Task A parameters for regularisation-based policies
        task_a_state = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
        }

        # Save Task A checkpoint
        task_a_ckpt_path = os.path.join(
            self.cfg.experiment.output_dir, "checkpoints",
            f"{method_name}_task_a.pt"
        )
        os.makedirs(os.path.dirname(task_a_ckpt_path), exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, task_a_ckpt_path)

        # ── Step 2: Apply LoRA ──
        self.logger.info("[Pipeline] Step 2: Applying LoRA adapters for Task B...")
        model_name = self.cfg.model.name if hasattr(self.cfg, "model") else "distilbert-base-uncased"
        num_labels_b = task_b.num_labels

        # We need to update the classifier head for Task B
        # (re-init classification head with correct num_labels)
        model = self._adapt_classifier(model, num_labels_b, model_name)
        model.to(self.device)

        if apply_lora:
            model = self.trainer.apply_lora(model, model_name, num_labels_b)
            model.to(self.device)

        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"[Pipeline] Trainable params for Task B: {num_trainable:,}")

        # ── Step 3: Task B Fine-tuning with Protection ──
        self.logger.info("[Pipeline] Step 3: Fine-tuning on Task B...")
        task_b_history = self.trainer.train(
            model,
            task_b.train_loader,
            task_b.val_loader,
            policy=protection_policy,
            task_a_state=task_a_state,
            run_name=f"{method_name}_task_b",
        )

        _, task_b_acc = self.trainer.evaluate(model, task_b.test_loader)
        self.logger.info(f"[Pipeline] Task B accuracy: {task_b_acc:.4f}")

        # ── Step 4: Re-evaluate on Task A ──
        self.logger.info("[Pipeline] Step 4: Re-evaluating on Task A (forgetting check)...")

        # For LoRA models, we need to evaluate with the Task A head
        # (in a simplified CL setup, we swap the final layer back for Task A eval)
        # Here we use a fresh copy of the Task A head weights for evaluation
        task_a_post_model = self._make_task_a_eval_model(
            model_factory, task_a_state, task_a.num_labels, model_name
        )
        _, task_a_post_acc = self.trainer.evaluate(task_a_post_model, task_a.test_loader)
        self.logger.info(
            f"[Pipeline] Task A accuracy (post-B): {task_a_post_acc:.4f} "
            f"| Forgetting: {task_a_pre_acc - task_a_post_acc:.4f}"
        )

        total_time = time.time() - t_total
        self.logger.info(f"[Pipeline] {method_name} complete in {total_time:.1f}s")

        return CLResult(
            method_name=method_name,
            model_name=model_name,
            task_a_name=task_a.name,
            task_a_train_history=task_a_history,
            task_a_pre_accuracy=task_a_pre_acc,
            task_a_post_accuracy=task_a_post_acc,
            task_b_name=task_b.name,
            task_b_train_history=task_b_history,
            task_b_accuracy=task_b_acc,
            importance_scores=importance_scores,
            rankings=rankings,
            total_time_seconds=total_time,
            num_trainable_params_task_b=num_trainable,
            metadata={
                "protection": protection_policy.protection_summary()
                if protection_policy else {"policy": "none"},
                "task_a_pre_loss": task_a_loss,
            },
        )

    def _adapt_classifier(
        self, model: nn.Module, num_labels: int, model_name: str
    ) -> nn.Module:
        """Replace the classification head for a different number of labels."""
        model_copy = copy.deepcopy(model)

        # Try common classifier attribute paths
        for attr in ["classifier", "pre_classifier"]:
            if hasattr(model_copy, attr):
                head = getattr(model_copy, attr)
                if hasattr(head, "out_features") and head.out_features != num_labels:
                    in_features = head.in_features
                    setattr(
                        model_copy, attr,
                        nn.Linear(in_features, num_labels)
                    )
                    self.logger.info(
                        f"Replaced {attr}: {head.out_features} → {num_labels} labels"
                    )

        return model_copy

    def _make_task_a_eval_model(
        self,
        model_factory,
        task_a_state: dict,
        num_labels_a: int,
        model_name: str,
    ) -> nn.Module:
        """
        Create a model for Task A evaluation by restoring Task A weights.

        This reconstructs the Task A model state for forgetting measurement
        without needing to re-train.
        """
        model, _ = model_factory()
        model.to(self.device)

        # Load Task A state (strict=False to handle classifier mismatch)
        model.load_state_dict(task_a_state, strict=False)
        return model
