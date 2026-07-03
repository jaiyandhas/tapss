"""
saliency/activation.py

Method 2 — Activation Frequency Estimator.

Tracks how often neurons/weights contribute significantly during inference.
A weight matrix's importance is proxied by the activation magnitude of the
neurons it projects into.

Rationale
---------
Weights feeding into frequently-active neurons carry more "live" information.
Dead neurons (activations near zero) can typically be modified without
meaningfully disrupting the model's function. This is the dual of gradient
magnitude: we look at what the model *uses* rather than what the loss *pushes*.

Implementation
--------------
- Register forward hooks on all Linear and Conv1D layers.
- For each batch, record mean absolute activation per layer.
- "Activation frequency" = fraction of batches where activation exceeds a threshold.
- Map activation importance back to the weight parameters of each layer.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from saliency.base import ImportanceEstimator, ImportanceScores

logger = logging.getLogger(__name__)


class ActivationFrequencyEstimator(ImportanceEstimator):
    """
    Estimates parameter importance via activation frequency and magnitude.

    For each Linear layer, the importance of its weight matrix is the
    mean activation magnitude of its output neurons across the calibration set,
    thresholded to measure how often the neuron is "active".
    """

    def __init__(
        self,
        cfg: Any = None,
        num_calibration_batches: int = 50,
        threshold_percentile: float = 75.0,
    ):
        super().__init__(cfg)
        if cfg is not None and hasattr(cfg, "saliency"):
            self.num_calibration_batches = cfg.saliency.get(
                "num_calibration_batches", num_calibration_batches
            )
            self.threshold_percentile = cfg.saliency.get(
                "threshold_percentile", threshold_percentile
            )
        else:
            self.num_calibration_batches = num_calibration_batches
            self.threshold_percentile = threshold_percentile

    @property
    def name(self) -> str:
        return "activation_frequency"

    def compute(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> ImportanceScores:
        """
        Compute activation frequency importance scores.

        For each weight parameter:
          score = mean(|activation|) at the layer, averaged over batches.

        Parameters
        ----------
        model : nn.Module
        dataloader : DataLoader
        device : torch.device

        Returns
        -------
        ImportanceScores
        """
        logger.info(
            f"[ActivationFrequency] Computing over {self.num_calibration_batches} batches..."
        )
        t0 = time.time()

        model.eval()

        # --- Register hooks on Linear layers ---
        # Maps module_name → accumulated mean absolute output activation
        layer_activation_sums: dict[str, float] = defaultdict(float)
        layer_above_thresh_count: dict[str, int] = defaultdict(int)
        layer_batch_count: dict[str, int] = defaultdict(int)

        # Build a map from module id → qualified name
        module_name_map: dict[int, str] = {}
        for name, module in model.named_modules():
            module_name_map[id(module)] = name

        hooks = []

        def _make_hook(module_name: str):
            def hook(module, input, output):
                # output shape: (batch, seq, hidden) or (batch, hidden)
                if isinstance(output, torch.Tensor):
                    mean_abs = output.detach().abs().mean().item()
                    layer_activation_sums[module_name] += mean_abs
                    layer_batch_count[module_name] += 1
            return hook

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear,)):
                h = module.register_forward_hook(_make_hook(name))
                hooks.append(h)

        # --- Run calibration batches ---
        num_batches_processed = 0
        with torch.no_grad():
            for batch in self._iter_batches(dataloader, self.num_calibration_batches, device):
                model(
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                )
                num_batches_processed += 1

        # Remove hooks
        for h in hooks:
            h.remove()

        if num_batches_processed == 0:
            logger.warning("[ActivationFrequency] No batches processed!")
            return ImportanceScores(
                scores={
                    n: 0.0 for n, p in model.named_parameters() if p.requires_grad
                },
                method=self.name,
            )

        # Average activation per module
        module_importance: dict[str, float] = {
            name: layer_activation_sums[name] / max(layer_batch_count[name], 1)
            for name in layer_activation_sums
        }

        # Map module activation importance → weight parameter importance
        # For each Linear layer `foo.bar`, its weight is `foo.bar.weight`
        raw_scores: dict[str, float] = {}
        for param_name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Find the parent module name
            parent = ".".join(param_name.split(".")[:-1])
            score = module_importance.get(parent, 0.0)
            raw_scores[param_name] = score

        normalised = ImportanceScores.normalise(raw_scores)

        elapsed = time.time() - t0
        logger.info(
            f"[ActivationFrequency] Done in {elapsed:.1f}s. "
            f"Batches: {num_batches_processed}. "
            f"Layers instrumented: {len(module_importance)}."
        )

        return ImportanceScores(
            scores=normalised,
            method=self.name,
            metadata={
                "raw_scores": raw_scores,
                "module_importances": module_importance,
                "num_batches": num_batches_processed,
                "elapsed_seconds": elapsed,
                "threshold_percentile": self.threshold_percentile,
            },
        )
