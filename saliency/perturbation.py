"""
saliency/perturbation.py

Method 3 — Perturbation Sensitivity Estimator.

Measures parameter importance by perturbing each parameter (block) and
observing the resulting shift in the model's output distribution.

Rationale
---------
A parameter is important if perturbing it causes large output changes.
This is model-agnostic and does not require gradients — it probes the
functional sensitivity of the model's output to each weight.

This resembles occlusion sensitivity in computer vision interpretability:
"What happens to the output if this component is slightly corrupted?"

Mathematically, for parameter θ_i:
  sensitivity_i = E_x[D(f(x; θ), f(x; θ + ε_i))]

where D is an output divergence measure (KL divergence or L2) and ε_i
is Gaussian noise applied only to θ_i.

Implementation details
----------------------
- We perturb parameter by parameter (or by parameter group).
- For efficiency, perturbation is applied in-place and then restored.
- The KL divergence between original and perturbed logits (softmax) is used.
- A smaller calibration set is used here (this is slow for large models).
"""
from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from saliency.base import ImportanceEstimator, ImportanceScores

logger = logging.getLogger(__name__)


def _kl_divergence(logits_orig: torch.Tensor, logits_perturbed: torch.Tensor) -> float:
    """KL(original || perturbed) averaged over batch."""
    p = F.softmax(logits_orig, dim=-1).clamp(min=1e-9)
    q = F.softmax(logits_perturbed, dim=-1).clamp(min=1e-9)
    kl = (p * (p / q).log()).sum(dim=-1)  # (batch,)
    return float(kl.mean().item())


def _l2_divergence(logits_orig: torch.Tensor, logits_perturbed: torch.Tensor) -> float:
    """L2 distance between softmax distributions, averaged over batch."""
    p = F.softmax(logits_orig, dim=-1)
    q = F.softmax(logits_perturbed, dim=-1)
    return float((p - q).norm(dim=-1).mean().item())


class PerturbationSensitivityEstimator(ImportanceEstimator):
    """
    Estimates parameter importance via output sensitivity to perturbation.

    This is the most computationally expensive estimator.
    For large models, set num_calibration_batches low (10–20).
    """

    def __init__(
        self,
        cfg: Any = None,
        perturbation_std: float = 0.01,
        num_calibration_batches: int = 20,
        divergence: str = "kl",
    ):
        super().__init__(cfg)
        if cfg is not None and hasattr(cfg, "saliency"):
            self.perturbation_std = cfg.saliency.get("perturbation_std", perturbation_std)
            self.num_calibration_batches = cfg.saliency.get(
                "num_calibration_batches", num_calibration_batches
            )
            self.divergence = cfg.saliency.get("divergence", divergence)
        else:
            self.perturbation_std = perturbation_std
            self.num_calibration_batches = num_calibration_batches
            self.divergence = divergence

        self._divergence_fn = _kl_divergence if self.divergence == "kl" else _l2_divergence

    @property
    def name(self) -> str:
        return "perturbation_sensitivity"

    def compute(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> ImportanceScores:
        """
        Compute perturbation sensitivity scores.

        For each named parameter (weight tensors only, bias excluded for speed):
          1. Cache the original outputs on the calibration set.
          2. Perturb the parameter with Gaussian noise.
          3. Recompute outputs.
          4. Measure divergence.
          5. Restore original parameter values.

        Returns
        -------
        ImportanceScores
        """
        logger.info(
            f"[PerturbationSensitivity] Computing (divergence={self.divergence}, "
            f"std={self.perturbation_std}, batches={self.num_calibration_batches})..."
        )
        t0 = time.time()

        model.eval()

        # Collect calibration batches once (reused for each parameter perturbation)
        calibration_batches = []
        with torch.no_grad():
            for batch in self._iter_batches(dataloader, self.num_calibration_batches, device):
                calibration_batches.append(batch)

        if not calibration_batches:
            logger.warning("[PerturbationSensitivity] No calibration batches!")
            return ImportanceScores(
                scores={n: 0.0 for n, p in model.named_parameters() if p.requires_grad},
                method=self.name,
            )

        # Compute baseline logits on the calibration set
        baseline_logits: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in calibration_batches:
                out = model(
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                )
                baseline_logits.append(out.logits.detach())

        # --- Perturb each parameter and measure divergence ---
        raw_scores: dict[str, float] = {}
        all_params = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]

        logger.info(f"[PerturbationSensitivity] Scoring {len(all_params)} parameter tensors...")

        for idx, (name, param) in enumerate(all_params):
            if idx % 20 == 0:
                logger.debug(
                    f"[PerturbationSensitivity] {idx}/{len(all_params)}: {name}"
                )

            # Save original
            original_data = param.data.clone()

            # Apply Gaussian noise perturbation
            noise = torch.randn_like(param.data) * self.perturbation_std
            param.data.add_(noise)

            # Compute perturbed logits and measure divergence
            total_divergence = 0.0
            with torch.no_grad():
                for batch, base_logits in zip(calibration_batches, baseline_logits):
                    out = model(
                        input_ids=batch.get("input_ids"),
                        attention_mask=batch.get("attention_mask"),
                    )
                    div = self._divergence_fn(base_logits, out.logits)
                    total_divergence += div

            raw_scores[name] = total_divergence / len(calibration_batches)

            # Restore original parameter
            param.data.copy_(original_data)

        normalised = ImportanceScores.normalise(raw_scores)

        elapsed = time.time() - t0
        logger.info(
            f"[PerturbationSensitivity] Done in {elapsed:.1f}s. "
            f"Parameters scored: {len(normalised)}."
        )

        return ImportanceScores(
            scores=normalised,
            method=self.name,
            metadata={
                "raw_scores": raw_scores,
                "perturbation_std": self.perturbation_std,
                "divergence": self.divergence,
                "num_batches": len(calibration_batches),
                "elapsed_seconds": elapsed,
            },
        )
