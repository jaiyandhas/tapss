"""
peft_modules/lora_trainer.py

LoRA fine-tuning trainer for TAPSS continual learning experiments.

Wraps HuggingFace PEFT LoRA configuration + a training loop that
integrates protection policies into the optimisation process.

Supports:
  - Configurable LoRA rank, alpha, dropout, target_modules
  - All 5 protection policies via the ProtectionPolicy interface
  - Automatic task-A checkpoint restoration
  - TensorBoard logging
  - Training history for downstream visualisation
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from peft_modules.protection import ProtectionPolicy, LRScalingPolicy
from saliency.base import ImportanceScores

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """Records per-epoch training and evaluation metrics."""

    train_losses: list[float] = field(default_factory=list)
    train_accuracies: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)
    additional_losses: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "train_loss": self.train_losses,
            "train_acc": self.train_accuracies,
            "val_loss": self.val_losses,
            "val_acc": self.val_accuracies,
            "epoch_times": self.epoch_times,
            "additional_losses": self.additional_losses,
        }


class LoRATrainer:
    """
    TAPSS LoRA fine-tuning trainer.

    Applies a PEFT LoRA configuration to a model and trains it on a task,
    optionally applying a protection policy to preserve important parameters.

    Usage
    -----
    >>> trainer = LoRATrainer(cfg, device)
    >>> trainer.apply_lora(model)
    >>> history = trainer.train(model, train_loader, val_loader, policy)
    """

    def __init__(self, cfg: Any, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)

        # LoRA hyperparameters
        lora_cfg = cfg.lora if hasattr(cfg, "lora") else cfg
        self.lora_r = int(lora_cfg.get("r", 8))
        self.lora_alpha = int(lora_cfg.get("lora_alpha", 16))
        self.lora_dropout = float(lora_cfg.get("lora_dropout", 0.05))
        self.bias = str(lora_cfg.get("bias", "none"))
        self.modules_to_save = list(lora_cfg.get("modules_to_save", ["classifier"]))
        self.target_modules = lora_cfg.get("target_modules", None)
        if self.target_modules is not None:
            self.target_modules = list(self.target_modules)

        # Training hyperparameters
        train_cfg = cfg.training if hasattr(cfg, "training") else cfg
        self.num_epochs = int(train_cfg.get("num_epochs", 3))
        self.learning_rate = float(train_cfg.get("learning_rate", 2e-4))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.01))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        self.warmup_ratio = float(train_cfg.get("warmup_ratio", 0.06))

        self.output_dir = str(cfg.experiment.output_dir) if hasattr(cfg, "experiment") else "outputs"

    # ─────────────────────────────────────────────────────────────
    # LoRA Application
    # ─────────────────────────────────────────────────────────────

    def apply_lora(
        self,
        model: nn.Module,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
    ) -> nn.Module:
        """
        Apply LoRA adapters to the model using PEFT.

        Auto-detects target modules if not specified in config.

        Parameters
        ----------
        model : nn.Module
            Base model to apply LoRA to.
        model_name : str
            Model name for auto-detecting target modules.
        num_labels : int
            Number of task labels (for task_type inference).

        Returns
        -------
        nn.Module
            PEFT model with LoRA adapters.
        """
        from models.registry import LORA_TARGET_MODULES

        # Auto-detect target modules if not configured
        target_modules = self.target_modules
        if target_modules is None:
            for key, modules in LORA_TARGET_MODULES.items():
                if key in model_name.lower() or model_name.lower() in key:
                    target_modules = modules
                    break
            if target_modules is None:
                target_modules = ["query", "value"]
                self.logger.warning(
                    f"Could not auto-detect LoRA target modules for {model_name!r}. "
                    f"Defaulting to {target_modules}."
                )

        self.logger.info(
            f"Applying LoRA: r={self.lora_r}, alpha={self.lora_alpha}, "
            f"dropout={self.lora_dropout}, target_modules={target_modules}"
        )

        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=TaskType.SEQ_CLS,
            target_modules=target_modules,
            modules_to_save=self.modules_to_save,
        )

        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()
        return peft_model

    # ─────────────────────────────────────────────────────────────
    # Training Loop
    # ─────────────────────────────────────────────────────────────

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        policy: Optional[ProtectionPolicy] = None,
        task_a_state: Optional[dict[str, torch.Tensor]] = None,
        run_name: str = "lora_run",
        use_tensorboard: bool = True,
    ) -> TrainingHistory:
        """
        Run LoRA fine-tuning with optional protection policy.

        Parameters
        ----------
        model : nn.Module
            PEFT model with LoRA adapters applied.
        train_loader : DataLoader
            Training data for the current task.
        val_loader : DataLoader
            Validation data for evaluation.
        policy : ProtectionPolicy | None
            Protection policy to apply. None = no protection.
        task_a_state : dict | None
            Frozen Task A parameter state (needed for regularisation policies).
        run_name : str
            Name for this training run (used in TensorBoard and file names).
        use_tensorboard : bool
            Whether to log to TensorBoard.

        Returns
        -------
        TrainingHistory
        """
        history = TrainingHistory()
        tb_writer = None

        if use_tensorboard:
            tb_dir = os.path.join(self.output_dir, "tb_logs", run_name)
            os.makedirs(tb_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=tb_dir)

        model.to(self.device)

        # Build optimiser (with LR scaling if using LRScalingPolicy)
        if isinstance(policy, LRScalingPolicy):
            param_groups = policy.build_param_groups(model, self.learning_rate)
            optimizer = optim.AdamW(param_groups, weight_decay=self.weight_decay)
        else:
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        # LR scheduler (linear warmup → linear decay)
        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = self._build_scheduler(optimizer, warmup_steps, total_steps)

        # Notify policy
        if policy is not None:
            policy.on_train_begin(model, optimizer)

        loss_fn = nn.CrossEntropyLoss()
        global_step = 0

        for epoch in range(self.num_epochs):
            if policy is not None:
                policy.on_epoch_begin(epoch, self.num_epochs)

            # ── Training ──
            model.train()
            epoch_train_loss = 0.0
            epoch_additional_loss = 0.0
            correct = 0
            total = 0
            epoch_t0 = time.time()

            pbar = tqdm(
                train_loader,
                desc=f"[{run_name}] Epoch {epoch + 1}/{self.num_epochs}",
                leave=False,
            )

            for batch in pbar:
                batch = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }

                if policy is not None:
                    policy.on_step_begin(model, optimizer, global_step)

                optimizer.zero_grad()

                outputs = model(
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels"),
                )

                # Primary task loss
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    task_loss = outputs.loss
                else:
                    task_loss = loss_fn(outputs.logits, batch["labels"])

                # Additional loss from protection policy
                additional_loss = torch.tensor(0.0, device=self.device)
                if policy is not None:
                    additional_loss = policy.compute_additional_loss(model, task_a_state)
                    additional_loss = additional_loss.to(self.device)

                total_loss = task_loss + additional_loss
                total_loss.backward()

                # Gradient dampening (for soft/adaptive policies)
                if policy is not None:
                    policy.on_step_end(model, optimizer, global_step)

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    self.max_grad_norm,
                )

                optimizer.step()
                scheduler.step()

                # Metrics
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
                epoch_train_loss += task_loss.item()
                epoch_additional_loss += additional_loss.item()

                pbar.set_postfix(
                    loss=f"{task_loss.item():.4f}",
                    acc=f"{correct / max(total, 1):.3f}",
                )
                global_step += 1

            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_train_acc = correct / max(total, 1)
            avg_additional = epoch_additional_loss / len(train_loader)

            history.train_losses.append(avg_train_loss)
            history.train_accuracies.append(avg_train_acc)
            history.additional_losses.append(avg_additional)
            history.epoch_times.append(time.time() - epoch_t0)

            # ── Validation ──
            val_loss, val_acc = self._evaluate(model, val_loader, loss_fn)
            history.val_losses.append(val_loss)
            history.val_accuracies.append(val_acc)

            self.logger.info(
                f"[{run_name}] Epoch {epoch + 1}/{self.num_epochs} — "
                f"train_loss={avg_train_loss:.4f}, train_acc={avg_train_acc:.3f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}, "
                f"reg_loss={avg_additional:.4f}"
            )

            if tb_writer:
                tb_writer.add_scalar("train/loss", avg_train_loss, epoch)
                tb_writer.add_scalar("train/acc", avg_train_acc, epoch)
                tb_writer.add_scalar("val/loss", val_loss, epoch)
                tb_writer.add_scalar("val/acc", val_acc, epoch)
                tb_writer.add_scalar("train/reg_loss", avg_additional, epoch)

        if tb_writer:
            tb_writer.close()

        return history

    def _evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
    ) -> tuple[float, float]:
        """Run evaluation and return (avg_loss, accuracy)."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                outputs = model(
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                )
                loss = loss_fn(outputs.logits, batch["labels"])
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
                total_loss += loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)
        accuracy = correct / max(total, 1)
        model.train()
        return avg_loss, accuracy

    def evaluate(
        self, model: nn.Module, dataloader: DataLoader
    ) -> tuple[float, float]:
        """Public evaluation method."""
        return self._evaluate(model, dataloader, nn.CrossEntropyLoss())

    def _build_scheduler(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
    ):
        """Build a linear warmup + linear decay scheduler."""
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / max(warmup_steps, 1)
            progress = float(current_step - warmup_steps) / max(
                total_steps - warmup_steps, 1
            )
            return max(0.0, 1.0 - progress)

        return LambdaLR(optimizer, lr_lambda)
