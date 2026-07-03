"""
scripts/compute_saliency.py

Compute TAPSS parameter importance scores for a pre-trained or fine-tuned model.

Usage
-----
  python scripts/compute_saliency.py \\
      --model distilbert-base-uncased \\
      --dataset ag_news \\
      --method tapss \\
      --output-dir outputs/saliency_run

  python scripts/compute_saliency.py \\
      --model distilbert-base-uncased \\
      --dataset ag_news \\
      --method gradient \\
      --num-batches 30 \\
      --skip-perturbation
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from task_datasets.loaders import load_task_dataset
from models.registry import ModelRegistry
from saliency import (
    GradientMagnitudeEstimator,
    ActivationFrequencyEstimator,
    PerturbationSensitivityEstimator,
    LayerContributionEstimator,
    TAPSSEstimator,
    build_rankings,
)
from visualization.heatmaps import plot_layer_importance_heatmap, plot_multi_method_heatmap
from visualization.distributions import plot_importance_histogram
from visualization.interactive import interactive_importance_scatter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

ESTIMATOR_MAP = {
    "gradient": GradientMagnitudeEstimator,
    "activation": ActivationFrequencyEstimator,
    "perturbation": PerturbationSensitivityEstimator,
    "layer": LayerContributionEstimator,
    "tapss": TAPSSEstimator,
    "all": None,  # special: run all methods
}


@click.command()
@click.option("--model", "-m", default="distilbert-base-uncased", help="Model name or alias.")
@click.option("--dataset", "-d", default="ag_news", help="Dataset name (registry key).")
@click.option("--method", default="tapss", type=click.Choice(list(ESTIMATOR_MAP.keys())), help="Saliency method.")
@click.option("--num-batches", default=50, help="Number of calibration batches.")
@click.option("--batch-size", default=16, help="Batch size for calibration loader.")
@click.option("--protection-percent", default=20.0, help="Top-K% parameters to flag as protected.")
@click.option("--output-dir", "-o", default="outputs/saliency", help="Output directory.")
@click.option("--seed", default=42, help="Random seed.")
@click.option("--skip-perturbation", is_flag=True, help="Skip slow perturbation estimator.")
@click.option("--show-plots", is_flag=True, help="Display plots interactively.")
def main(
    model: str,
    dataset: str,
    method: str,
    num_batches: int,
    batch_size: int,
    protection_percent: float,
    output_dir: str,
    seed: int,
    skip_perturbation: bool,
    show_plots: bool,
) -> None:
    """Compute and visualise parameter importance scores."""
    from rich.console import Console
    console = Console()
    console.print(f"[bold cyan]TAPSS Saliency Estimator[/bold cyan]")
    console.print(f"  Model:   {model}")
    console.print(f"  Dataset: {dataset}")
    console.print(f"  Method:  {method}")
    console.print(f"  Output:  {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    console.print("\n[dim]Loading model...[/dim]")
    from transformers import AutoTokenizer
    from transformers import AutoModelForSequenceClassification

    resolved_name = ModelRegistry.resolve_model_name(model)
    tokenizer = AutoTokenizer.from_pretrained(resolved_name)

    # Load dataset to determine num_labels
    raw_dataset = load_task_dataset(dataset, tokenizer, batch_size=batch_size, seed=seed)
    hf_model = AutoModelForSequenceClassification.from_pretrained(
        resolved_name, num_labels=raw_dataset.num_labels, ignore_mismatched_sizes=True
    )
    hf_model.to(device)

    console.print(f"  Model parameters: {sum(p.numel() for p in hf_model.parameters()):,}")

    # Build calibration loader (small subset for speed)
    from torch.utils.data import DataLoader, Subset
    calib_size = min(num_batches * batch_size, len(raw_dataset.train_dataset))
    calib_subset = Subset(raw_dataset.train_dataset, list(range(calib_size)))
    calib_loader = DataLoader(calib_subset, batch_size=batch_size, shuffle=False)

    # Run selected estimators
    all_scores = {}

    if method == "all":
        methods_to_run = ["gradient", "activation", "layer", "tapss"]
        if not skip_perturbation:
            methods_to_run.insert(2, "perturbation")
    else:
        methods_to_run = [method]

    for m in methods_to_run:
        console.print(f"\n[cyan]Running: {m}[/cyan]")
        if m == "gradient":
            est = GradientMagnitudeEstimator(num_calibration_batches=num_batches)
        elif m == "activation":
            est = ActivationFrequencyEstimator(num_calibration_batches=num_batches)
        elif m == "perturbation":
            est = PerturbationSensitivityEstimator(num_calibration_batches=min(num_batches, 20))
        elif m == "layer":
            est = LayerContributionEstimator(num_calibration_batches=num_batches)
        else:  # tapss
            est = TAPSSEstimator(skip_perturbation=skip_perturbation)

        scores = est.compute(hf_model, calib_loader, device)
        all_scores[m] = scores
        console.print(f"  Scored {len(scores)} parameters.")

        # Save rankings for this method
        rankings = build_rankings(scores, hf_model, protection_percent=protection_percent)
        method_dir = os.path.join(output_dir, m)
        rankings.save_csv(os.path.join(method_dir, "rankings.csv"))
        rankings.save_json(os.path.join(method_dir, "rankings.json"))
        rankings.save_html(os.path.join(method_dir, "rankings.html"))

        # Save plots
        plot_layer_importance_heatmap(
            scores,
            title=f"Layer Importance — {m}",
            save_path=os.path.join(method_dir, "layer_heatmap.png"),
            show=show_plots,
        )
        plot_importance_histogram(
            scores,
            save_path=os.path.join(method_dir, "score_histogram.png"),
            show=show_plots,
        )
        fig = interactive_importance_scatter(
            rankings,
            title=f"Parameter Importance — {m}",
            save_path=os.path.join(method_dir, "importance_scatter.html"),
        )
        console.print(f"  Outputs saved to {method_dir}/")

    # Multi-method heatmap if more than one method
    if len(all_scores) > 1:
        from visualization.heatmaps import plot_multi_method_heatmap
        plot_multi_method_heatmap(
            all_scores,
            save_path=os.path.join(output_dir, "multi_method_heatmap.png"),
            show=show_plots,
        )
        from visualization.interactive import interactive_layer_heatmap
        interactive_layer_heatmap(
            all_scores,
            save_path=os.path.join(output_dir, "interactive_heatmap.html"),
        )

    console.print(f"\n[bold green]✓ Saliency computation complete![/bold green]")
    console.print(f"  Output: {output_dir}/")


if __name__ == "__main__":
    main()
