"""
saliency/gradient.py

Gradient magnitude importance estimator.

Computes E[|∂L/∂θ|] over a calibration dataset.
Parameters that consistently attract large gradients are treated as
important — the model is actively relying on their current values.

Note: this is noisy and sensitive to calibration set size. It's most
useful as one signal among several, not on its own.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from saliency.base import ImportanceEstimator, ImportanceScores

logger = logging.getLogger(__name__)


class GradientMagnitudeEstimator(ImportanceEstimator):
    """
    Estimates parameter importance via average absolute gradient magnitude.

    For each calibration batch:
      1. Run forward pass (model in eval mode, but gradients enabled).
      2. Compute cross-entropy loss.
      3. Backpropagate.
      4. Accumulate |grad| per parameter.

    Final score = mean |grad| over all calibration batches, normalised to [0, 1].
    """

    def __init__(self, cfg: Any = None, num_calibration_batches: int = 50):
        super().__init__(cfg)
        if cfg is not None and hasattr(cfg, "saliency") and cfg.saliency is not None:
            self.num_calibration_batches = cfg.saliency.get("num_calibration_batches", num_calibration_batches)
        else:
            self.num_calibration_batches = num_calibration_batches

    @property
    def name(self) -> str:
        return "gradient_magnitude"

    def compute(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> ImportanceScores:
        """
        Compute gradient magnitude importance scores.

        Parameters
        ----------
        model : nn.Module
            Model to analyse (parameters must require grad).
        dataloader : DataLoader
            Calibration data.
        device : torch.device
            Computation device.

        Returns
        -------
        ImportanceScores
            Min-max normalised gradient magnitude scores.
        """
        logger.info(
            f"[GradientMagnitude] Computing over {self.num_calibration_batches} batches..."
        )
        t0 = time.time()

        model.eval()  # eval mode (no dropout) but we still need gradients

        # Accumulate |grad| per named parameter
        grad_accum: dict[str, torch.Tensor] = {}
        param_names = [
            name
            for name, p in model.named_parameters()
            if p.requires_grad
        ]

        # Initialise accumulators
        for name, p in model.named_parameters():
            if p.requires_grad:
                grad_accum[name] = torch.zeros_like(p.data, dtype=torch.float32)

        num_batches_processed = 0
        loss_fn = nn.CrossEntropyLoss()

        for batch in self._iter_batches(dataloader, self.num_calibration_batches, device):
            model.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )

            # Use model-computed loss if available, else compute manually
            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                loss = loss_fn(outputs.logits, batch["labels"])

            loss.backward()

            # Accumulate absolute gradients
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    grad_accum[name] += p.grad.data.abs().float()

            num_batches_processed += 1

        model.zero_grad()  # clean up

        if num_batches_processed == 0:
            logger.warning("[GradientMagnitude] No batches processed! Returning zero scores.")
            return ImportanceScores(
                scores={name: 0.0 for name in param_names},
                method=self.name,
            )

        # Average over batches, then collapse to scalar per parameter (mean over elements)
        raw_scores: dict[str, float] = {}
        for name in param_names:
            mean_abs_grad = grad_accum[name] / num_batches_processed
            raw_scores[name] = float(mean_abs_grad.mean().item())

        # Normalise to [0, 1]
        normalised = ImportanceScores.normalise(raw_scores)

        elapsed = time.time() - t0
        logger.info(
            f"[GradientMagnitude] Done in {elapsed:.1f}s. "
            f"Batches: {num_batches_processed}. "
            f"Params scored: {len(normalised)}."
        )

        return ImportanceScores(
            scores=normalised,
            method=self.name,
            metadata={
                "raw_scores": raw_scores,
                "num_batches": num_batches_processed,
                "elapsed_seconds": elapsed,
            },
        )
