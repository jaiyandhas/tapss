"""
models/wrappers.py

TAPSSModel: a thin wrapper around HuggingFace models that adds:
  - Named parameter iteration with layer-index awareness
  - Checkpoint save/restore for Task A → Task B CL transitions
  - Parameter statistics reporting
"""
from __future__ import annotations

import copy
import logging
import os
from typing import Any, Iterator

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TAPSSModel(nn.Module):
    """
    Wrapper around a HuggingFace model that adds continual-learning utilities.

    Attributes
    ----------
    model : nn.Module
        The underlying HuggingFace model.
    model_name : str
        Canonical model name string (e.g. "distilbert-base-uncased").
    _task_a_state : dict | None
        Frozen copy of parameters after Task A training.
        Used to compute forgetting and apply regularisation-based protection.
    """

    def __init__(self, model: nn.Module, model_name: str = "unknown"):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self._task_a_state: dict[str, torch.Tensor] | None = None

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, **kwargs) -> Any:
        return self.model(**kwargs)

    # ------------------------------------------------------------------
    # Parameter utilities
    # ------------------------------------------------------------------

    def named_trainable_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Iterate only over parameters that require gradients."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                yield name, param

    def num_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def num_total_parameters(self) -> int:
        """Count all parameters."""
        return sum(p.numel() for p in self.model.parameters())

    def parameter_summary(self) -> dict[str, int]:
        """Return a summary of parameter counts."""
        total = self.num_total_parameters()
        trainable = self.num_trainable_parameters()
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_pct": round(100.0 * trainable / max(total, 1), 2),
        }

    def layer_parameter_groups(self) -> dict[str, list[tuple[str, nn.Parameter]]]:
        """
        Group parameters by transformer layer index.

        Returns a dict mapping layer-key → list of (name, param) pairs.
        Ungrouped parameters (embeddings, pooler, classifier) are placed
        under key "other".
        """
        groups: dict[str, list] = {}
        for name, param in self.model.named_parameters():
            # Extract layer index from typical HF naming patterns:
            # e.g. "distilbert.transformer.layer.3.attention.q_lin.weight"
            parts = name.split(".")
            layer_key = "other"
            for i, part in enumerate(parts):
                if part == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_key = f"layer_{parts[i + 1]}"
                    break
            groups.setdefault(layer_key, []).append((name, param))
        return groups

    # ------------------------------------------------------------------
    # Checkpoint utilities for CL
    # ------------------------------------------------------------------

    def snapshot_task_a(self) -> None:
        """
        Store a deep copy of the current parameters.
        Called after Task A training to enable forgetting measurement
        and regularisation-based protection during Task B fine-tuning.
        """
        self._task_a_state = {
            name: param.data.clone().detach()
            for name, param in self.model.named_parameters()
        }
        logger.info(
            f"Snapshot saved: {len(self._task_a_state)} parameter tensors "
            f"from Task A checkpoint."
        )

    def get_task_a_state(self) -> dict[str, torch.Tensor]:
        """Return the stored Task A parameter state."""
        if self._task_a_state is None:
            raise RuntimeError(
                "No Task A snapshot found. Call snapshot_task_a() after Task A training."
            )
        return self._task_a_state

    def parameter_delta_from_task_a(self) -> dict[str, torch.Tensor]:
        """
        Compute per-parameter change from Task A → current state.
        Used for EWC regularisation and forgetting analysis.
        """
        task_a = self.get_task_a_state()
        return {
            name: (param.data - task_a[name]).abs()
            for name, param in self.model.named_parameters()
            if name in task_a
        }

    def save_checkpoint(self, path: str, metadata: dict | None = None) -> None:
        """Save model weights + metadata to a .pt checkpoint file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "model_state_dict": self.model.state_dict(),
            "model_name": self.model_name,
            "metadata": metadata or {},
        }
        torch.save(state, path)
        logger.info(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path: str, model: nn.Module) -> "TAPSSModel":
        """Load model weights from a checkpoint file."""
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        wrapped = cls(model, model_name=state.get("model_name", "unknown"))
        logger.info(f"Checkpoint loaded from {path}")
        return wrapped
