"""
evaluation/metrics.py

TAPSS Evaluation Metrics.

Computes all standard continual learning metrics:
  - Task accuracy and loss
  - Catastrophic Forgetting (CF)
  - Average Accuracy (AA)
  - Backward Transfer (BWT)
  - Forward Transfer (FWT) [for completeness]
  - Training time and GPU memory usage
  - Trainable parameter count and parameter distribution

Metric Definitions
------------------
Let R[i,j] = accuracy on Task i after training on Task j.
  n = number of tasks (here, 2)

  Average Accuracy (AA) = (1/n) Σ_i R[i,n]
  Backward Transfer (BWT) = (1/(n-1)) Σ_i [R[i,n] - R[i,i]]   (i < n)
  Forgetting (F) = max_j R[i,j] - R[i,n]   for Task i (simplified: R[i,A] - R[i,B])

References
----------
Lopez-Paz & Ranzato, "Gradient Episodic Memory for Continual Learning", NeurIPS 2017.
Chaudhry et al., "Riemannian Walk for Incremental Learning", ECCV 2018.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class CLMetrics:
    """
    Container for all continual learning evaluation metrics.

    Attributes
    ----------
    task_a_accuracy : float
        Accuracy on Task A after Task B training (primary forgetting signal).
    task_b_accuracy : float
        Accuracy on Task B (new task performance).
    task_a_loss : float
        Loss on Task A after Task B training.
    task_b_loss : float
        Loss on Task B.
    forgetting : float
        CF = accuracy_before - accuracy_after (positive = forgetting occurred).
    average_accuracy : float
        Mean accuracy across all tasks.
    backward_transfer : float
        BWT = R[A,B] - R[A,A] (negative BWT = forgetting).
    training_time_seconds : float
        Total training time.
    gpu_memory_mb : float
        Peak GPU memory usage in MB.
    num_trainable_params : int
        Number of trainable parameters during Task B training.
    param_distribution : dict
        Breakdown of parameters by layer.
    """

    task_a_accuracy: float
    task_b_accuracy: float
    task_a_loss: float = 0.0
    task_b_loss: float = 0.0
    forgetting: float = 0.0
    average_accuracy: float = 0.0
    backward_transfer: float = 0.0
    training_time_seconds: float = 0.0
    gpu_memory_mb: float = 0.0
    num_trainable_params: int = 0
    param_distribution: dict = field(default_factory=dict)
    method_name: str = ""

    def to_dict(self) -> dict:
        return {
            "method": self.method_name,
            "task_a_acc": self.task_a_accuracy,
            "task_b_acc": self.task_b_accuracy,
            "task_a_loss": self.task_a_loss,
            "task_b_loss": self.task_b_loss,
            "forgetting": self.forgetting,
            "avg_acc": self.average_accuracy,
            "backward_transfer": self.backward_transfer,
            "train_time_s": self.training_time_seconds,
            "gpu_mem_mb": self.gpu_memory_mb,
            "trainable_params": self.num_trainable_params,
        }

    def __str__(self) -> str:
        return (
            f"CLMetrics({self.method_name}):\n"
            f"  Task A acc (post-B): {self.task_a_accuracy:.4f}\n"
            f"  Task B acc:          {self.task_b_accuracy:.4f}\n"
            f"  Forgetting:          {self.forgetting:.4f}\n"
            f"  Average accuracy:    {self.average_accuracy:.4f}\n"
            f"  Backward transfer:   {self.backward_transfer:.4f}\n"
            f"  Training time:       {self.training_time_seconds:.1f}s\n"
            f"  GPU memory:          {self.gpu_memory_mb:.1f} MB\n"
            f"  Trainable params:    {self.num_trainable_params:,}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Core Metric Functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_forgetting(acc_before: float, acc_after: float) -> float:
    """
    Catastrophic forgetting = drop in Task A accuracy after Task B.

    CF = max(0, acc_before - acc_after)
    A positive value means the model forgot Task A.
    A negative value means the model actually improved on Task A (unlikely).
    """
    return max(0.0, acc_before - acc_after)


def compute_backward_transfer(acc_before: float, acc_after: float) -> float:
    """
    Backward Transfer (BWT).

    BWT = acc_after - acc_before
    Negative BWT indicates catastrophic forgetting.
    Positive BWT indicates the model improved on Task A after Task B training.
    """
    return acc_after - acc_before


def compute_average_accuracy(accuracies: list[float]) -> float:
    """Average accuracy across a list of tasks."""
    if not accuracies:
        return 0.0
    return sum(accuracies) / len(accuracies)


def get_gpu_memory_mb() -> float:
    """Return current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
) -> tuple[float, float]:
    """
    Evaluate a model on a DataLoader.

    Parameters
    ----------
    model : nn.Module
    dataloader : DataLoader
    device : torch.device
    loss_fn : nn.Module | None
        Loss function. Defaults to CrossEntropyLoss.

    Returns
    -------
    (avg_loss, accuracy) : tuple[float, float]
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    model.eval()
    model.to(device)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
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
    return avg_loss, accuracy


def compute_param_distribution(model: nn.Module) -> dict[str, int]:
    """
    Compute parameter count by layer group.

    Returns
    -------
    dict[str, int]
        Mapping from layer group name to parameter count.
    """
    distribution: dict[str, int] = {}
    for name, param in model.named_parameters():
        parts = name.split(".")
        layer_key = "other"
        for i, part in enumerate(parts):
            if part == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_key = f"layer_{parts[i + 1]}"
                break
        distribution[layer_key] = distribution.get(layer_key, 0) + param.numel()
    return distribution
