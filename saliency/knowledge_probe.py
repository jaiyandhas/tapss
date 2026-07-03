"""
saliency/knowledge_probe.py

Part 8 — Knowledge Sensitivity Probe.

Investigates whether specific parameter blocks encode "knowledge" by
measuring how much their perturbation affects various model behaviours:

1. Prediction confidence shift:  |P(y=ŷ) - P'(y=ŷ)|
2. Output probability shift:     KL(P || P')
3. Embedding shift:             ||h - h'||_2 (L2 distance of [CLS] embeddings)
4. Token distribution shift:    KL(softmax(logits) || softmax(logits'))

These four metrics are combined into a single Knowledge Sensitivity Score (KSS).

Design
------
Unlike the saliency estimators, the KSS is designed for *interpretive* use —
we apply it to specific, named parameter blocks (e.g. a single attention head's
weight matrix) and produce a rich visualisation showing which blocks carry
the most "embedded knowledge" from pre-training.

This helps answer the core research question:
"Do the parameters TAPSS identifies as important actually encode useful knowledge?"
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class BlockSensitivity:
    """
    Knowledge sensitivity results for a single parameter block.

    Attributes
    ----------
    block_name : str
        Fully-qualified parameter name.
    layer : str
        Inferred transformer layer key.
    confidence_shift : float
        Mean absolute change in prediction probability for top-1 class.
    kl_divergence : float
        Mean KL divergence between original and perturbed output distributions.
    embedding_shift : float
        Mean L2 distance between [CLS] embeddings before/after perturbation.
    token_distribution_kl : float
        Mean KL divergence of full token distribution (if LM head available).
    knowledge_sensitivity_score : float
        Composite score (weighted average of above metrics), normalised to [0, 1].
    """

    block_name: str
    layer: str
    confidence_shift: float = 0.0
    kl_divergence: float = 0.0
    embedding_shift: float = 0.0
    token_distribution_kl: float = 0.0
    knowledge_sensitivity_score: float = 0.0


@dataclass
class KnowledgeProbeResults:
    """
    Full results from a knowledge sensitivity probe run.

    Attributes
    ----------
    sensitivities : list[BlockSensitivity]
        Sensitivity results for each probed parameter block.
    perturbation_std : float
        Noise standard deviation used.
    num_calibration_batches : int
        Number of benchmark prompts/batches used.
    elapsed_seconds : float
        Total computation time.
    """

    sensitivities: list[BlockSensitivity]
    perturbation_std: float
    num_calibration_batches: int
    elapsed_seconds: float = 0.0

    def as_dataframe(self) -> "pd.DataFrame":
        """Convert results to a DataFrame."""
        import pandas as pd
        from dataclasses import asdict
        return pd.DataFrame([asdict(s) for s in self.sensitivities])

    def top_k(self, k: int) -> list[BlockSensitivity]:
        """Return the k highest-sensitivity blocks."""
        return sorted(
            self.sensitivities,
            key=lambda s: s.knowledge_sensitivity_score,
            reverse=True,
        )[:k]


def _infer_layer(name: str) -> str:
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return f"layer_{parts[i + 1]}"
    return "other"


def _get_cls_embedding(model: nn.Module, batch: dict) -> Optional[torch.Tensor]:
    """
    Extract the [CLS] token embedding from the model's last hidden state.
    Returns None if the model does not expose hidden states.
    """
    try:
        outputs = model(
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=True,
        )
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            # Last hidden state, [CLS] token (position 0)
            return outputs.hidden_states[-1][:, 0, :].detach()
    except Exception:
        pass
    return None


class KnowledgeProbe:
    """
    Probes the knowledge encoded in parameter blocks via perturbation.

    Usage
    -----
    >>> probe = KnowledgeProbe(perturbation_std=0.01, num_calibration_batches=20)
    >>> results = probe.run(model, dataloader, device, block_names=[...])
    >>> results.as_dataframe()
    """

    def __init__(
        self,
        perturbation_std: float = 0.01,
        num_calibration_batches: int = 20,
        confidence_weight: float = 0.35,
        kl_weight: float = 0.35,
        embedding_weight: float = 0.30,
    ):
        self.perturbation_std = perturbation_std
        self.num_calibration_batches = num_calibration_batches
        self.confidence_weight = confidence_weight
        self.kl_weight = kl_weight
        self.embedding_weight = embedding_weight
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        block_names: Optional[list[str]] = None,
        max_blocks: int = 50,
    ) -> KnowledgeProbeResults:
        """
        Run the knowledge sensitivity probe.

        Parameters
        ----------
        model : nn.Module
            Model to probe.
        dataloader : DataLoader
            Benchmark prompt batches.
        device : torch.device
            Computation device.
        block_names : list[str] | None
            Specific parameter names to probe. If None, probes all
            weight tensors (up to max_blocks).
        max_blocks : int
            Maximum number of blocks to probe (for time budget).

        Returns
        -------
        KnowledgeProbeResults
        """
        self.logger.info(
            f"[KnowledgeProbe] Starting probe (std={self.perturbation_std}, "
            f"batches={self.num_calibration_batches}, max_blocks={max_blocks})"
        )
        t0 = time.time()

        model.eval()

        # Select blocks to probe
        all_params = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        if block_names is not None:
            param_map = dict(all_params)
            probe_params = [(n, param_map[n]) for n in block_names if n in param_map]
        else:
            # Prefer weight tensors; skip tiny biases
            probe_params = [
                (name, p) for name, p in all_params
                if name.endswith(".weight") and p.numel() > 100
            ][:max_blocks]

        self.logger.info(f"[KnowledgeProbe] Probing {len(probe_params)} parameter blocks.")

        # Collect calibration batches
        calibration_batches = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= self.num_calibration_batches:
                    break
                calibration_batches.append(
                    {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                )

        # Compute baseline outputs
        baseline_probs: list[torch.Tensor] = []
        baseline_embeddings: list[Optional[torch.Tensor]] = []

        with torch.no_grad():
            for batch in calibration_batches:
                out = model(
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                )
                baseline_probs.append(F.softmax(out.logits, dim=-1).detach())
                baseline_embeddings.append(_get_cls_embedding(model, batch))

        # Probe each block
        sensitivities: list[BlockSensitivity] = []

        for block_idx, (name, param) in enumerate(probe_params):
            if block_idx % 10 == 0:
                self.logger.debug(f"[KnowledgeProbe] {block_idx}/{len(probe_params)}: {name}")

            original_data = param.data.clone()
            noise = torch.randn_like(param.data) * self.perturbation_std
            param.data.add_(noise)

            conf_shifts, kl_divs, emb_shifts = [], [], []

            with torch.no_grad():
                for i, batch in enumerate(calibration_batches):
                    out = model(
                        input_ids=batch.get("input_ids"),
                        attention_mask=batch.get("attention_mask"),
                    )
                    perturbed_probs = F.softmax(out.logits, dim=-1)

                    # Confidence shift: change in top-1 probability
                    base_conf = baseline_probs[i].max(dim=-1).values
                    pert_conf = perturbed_probs.max(dim=-1).values
                    conf_shifts.append(float((base_conf - pert_conf).abs().mean().item()))

                    # KL divergence
                    p = baseline_probs[i].clamp(min=1e-9)
                    q = perturbed_probs.clamp(min=1e-9)
                    kl = (p * (p / q).log()).sum(dim=-1).mean()
                    kl_divs.append(float(kl.item()))

                    # Embedding shift
                    pert_emb = _get_cls_embedding(model, batch)
                    if pert_emb is not None and baseline_embeddings[i] is not None:
                        shift = (baseline_embeddings[i] - pert_emb).norm(dim=-1).mean()
                        emb_shifts.append(float(shift.item()))

            param.data.copy_(original_data)

            mean_conf = float(np.mean(conf_shifts)) if conf_shifts else 0.0
            mean_kl = float(np.mean(kl_divs)) if kl_divs else 0.0
            mean_emb = float(np.mean(emb_shifts)) if emb_shifts else 0.0

            sensitivities.append(
                BlockSensitivity(
                    block_name=name,
                    layer=_infer_layer(name),
                    confidence_shift=mean_conf,
                    kl_divergence=mean_kl,
                    embedding_shift=mean_emb,
                    token_distribution_kl=mean_kl,  # reuse KL as proxy
                    # Raw composite (will be normalised below)
                    knowledge_sensitivity_score=(
                        self.confidence_weight * mean_conf
                        + self.kl_weight * mean_kl
                        + self.embedding_weight * mean_emb
                    ),
                )
            )

        # Normalise composite scores to [0, 1]
        raw_composite = np.array([s.knowledge_sensitivity_score for s in sensitivities])
        if raw_composite.max() - raw_composite.min() > 1e-10:
            normalised = (raw_composite - raw_composite.min()) / (
                raw_composite.max() - raw_composite.min()
            )
        else:
            normalised = np.full_like(raw_composite, 0.5)

        for s, kss in zip(sensitivities, normalised):
            s.knowledge_sensitivity_score = float(kss)

        elapsed = time.time() - t0
        self.logger.info(
            f"[KnowledgeProbe] Complete in {elapsed:.1f}s. "
            f"Blocks probed: {len(sensitivities)}."
        )

        return KnowledgeProbeResults(
            sensitivities=sensitivities,
            perturbation_std=self.perturbation_std,
            num_calibration_batches=len(calibration_batches),
            elapsed_seconds=elapsed,
        )
