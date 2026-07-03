"""
scripts/run_experiment.py

Main CLI entry point for a full TAPSS continual learning experiment.

Usage
-----
  python scripts/run_experiment.py --config configs/experiment/cl_ag_sst2.yaml
  python scripts/run_experiment.py --config configs/experiment/cl_ag_sst2.yaml \
      training.num_epochs=5 protection.topk_percent=30

All config values can be overridden via CLI key=value syntax (Hydra-style).
"""
from __future__ import annotations

import copy
import logging
import os
import sys
import time
from pathlib import Path

import click
import torch
import yaml
from omegaconf import OmegaConf

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from task_datasets.loaders import load_task_dataset
from models.registry import get_model_and_tokenizer, ModelRegistry
from saliency.tapss import TAPSSEstimator
from saliency.rankings import build_rankings
from peft_modules.protection import build_protection_policy
from continual_learning.pipeline import ContinualLearningPipeline
from evaluation.tables import build_comparison_table, format_results_table
from evaluation.tracker import ExperimentTracker
from visualization.report import generate_html_report
from visualization.heatmaps import plot_layer_importance_heatmap, plot_protection_map
from visualization.distributions import plot_importance_histogram
from visualization.continual import (
    plot_forgetting_over_methods, plot_task_accuracy_comparison,
    plot_training_history, plot_radar_chart,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_config(config_path: str, overrides: tuple[str]) -> OmegaConf:
    """Load YAML config, resolve defaults list, and apply CLI overrides."""
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    config_dir = Path(__file__).resolve().parent.parent / "configs"
    composed = {}

    defaults = raw.get("defaults", [])
    for item in defaults:
        if isinstance(item, dict):
            for k, v in item.items():
                if k == "_self_":
                    continue
                sub_path = config_dir / k / f"{v}.yaml"
                if sub_path.exists():
                    with open(sub_path) as sf:
                        sub_raw = yaml.safe_load(sf) or {}
                        composed.update(sub_raw)
        elif isinstance(item, str):
            if item == "_self_":
                continue
            if item.startswith("/"):
                sub_path = config_dir / f"{item[1:]}.yaml"
                if sub_path.exists():
                    with open(sub_path) as sf:
                        sub_raw = yaml.safe_load(sf) or {}
                        composed.update(sub_raw)

    composed.update({k: v for k, v in raw.items() if k != "defaults"})
    cfg = OmegaConf.create(composed)

    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        OmegaConf.update(cfg, key, yaml.safe_load(value), merge=True)

    return cfg



def _set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@click.command()
@click.option(
    "--config", "-c",
    default="configs/experiment/cl_ag_sst2.yaml",
    help="Path to experiment YAML config file.",
    show_default=True,
)
@click.option(
    "--override", "-o",
    multiple=True,
    help="Config overrides in key=value format (repeatable).",
)
@click.option("--skip-perturbation", is_flag=True, help="Skip slow perturbation estimator.")
@click.option("--dry-run", is_flag=True, help="Validate config but don't run training.")
def main(config: str, override: tuple, skip_perturbation: bool, dry_run: bool) -> None:
    """
    Run a full TAPSS continual learning experiment.

    Executes Task A training, TAPSS importance estimation,
    protection policy application, Task B fine-tuning, and evaluation.
    Saves results, plots, and a self-contained HTML report.
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    console.print(Panel.fit(
        "[bold cyan]TAPSS — Task-Adaptive Parameter Saliency Score[/bold cyan]\n"
        "[dim]Continual Learning Experiment Runner[/dim]",
        border_style="cyan",
    ))

    # ── Load config ──
    cfg = _load_config(config, override)
    seed = cfg.get("experiment", {}).get("seed", 42)
    output_dir = cfg.get("experiment", {}).get("output_dir", "outputs/run")

    console.print(f"\n[bold]Config:[/bold] {config}")
    console.print(f"[bold]Output:[/bold] {output_dir}")
    console.print(f"[bold]Seed:[/bold] {seed}")

    if dry_run:
        console.print("\n[yellow]Dry run — config validated. Exiting.[/yellow]")
        console.print(OmegaConf.to_yaml(cfg))
        return

    _set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold]Device:[/bold] {device}")

    os.makedirs(output_dir, exist_ok=True)

    # ── Save config snapshot ──
    config_snapshot_path = os.path.join(output_dir, "config.yaml")
    with open(config_snapshot_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # ── Load Task A dataset ──
    task_a_name = cfg.get("continual_learning", {}).get("task_a_dataset", "ag_news")
    task_b_name = cfg.get("continual_learning", {}).get("task_b_dataset", "sst2")
    model_name = cfg.model.name if hasattr(cfg, "model") else "distilbert-base-uncased"
    batch_size = cfg.training.batch_size if hasattr(cfg, "training") else 16

    console.print(f"\n[bold]Task A:[/bold] {task_a_name} → [bold]Task B:[/bold] {task_b_name}")

    # Load tokenizer (using task A's model name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        ModelRegistry.resolve_model_name(model_name),
        cache_dir=cfg.model.get("cache_dir", ".cache/models") if hasattr(cfg, "model") else ".cache/models",
    )

    train_size = cfg.get("dataset", {}).get("train_size", None)
    val_size = cfg.get("dataset", {}).get("val_size", None)
    test_size = cfg.get("dataset", {}).get("test_size", None)

    console.print("[dim]Loading Task A dataset...[/dim]")
    task_a = load_task_dataset(
        task_a_name, tokenizer, batch_size=batch_size, seed=seed,
        train_size=train_size, val_size=val_size, test_size=test_size
    )

    console.print("[dim]Loading Task B dataset...[/dim]")
    task_b = load_task_dataset(
        task_b_name, tokenizer, batch_size=batch_size, seed=seed,
        train_size=train_size, val_size=val_size, test_size=test_size
    )

    console.print(f"  Task A: {task_a}")
    console.print(f"  Task B: {task_b}")

    # ── Model factory ──
    def model_factory():
        return get_model_and_tokenizer(cfg, task_a.num_labels, device)

    # ── Train Task A and compute TAPSS scores ──
    console.print("\n[bold cyan]Step 1: Training on Task A...[/bold cyan]")
    t0 = time.time()

    pipeline = ContinualLearningPipeline(cfg, device)

    # Get a model for saliency computation
    tapss_model, _ = model_factory()

    # Quick Task A training pass to get a calibrated model
    trainer_for_saliency = pipeline.trainer
    task_a_history_for_saliency = trainer_for_saliency.train(
        tapss_model.model,
        task_a.train_loader,
        task_a.val_loader,
        run_name="saliency_task_a",
    )

    # ── Compute TAPSS scores ──
    console.print("\n[bold cyan]Step 2: Computing TAPSS importance scores...[/bold cyan]")
    estimator = TAPSSEstimator(cfg, skip_perturbation=skip_perturbation)

    # Use a subset of Task A for calibration
    from torch.utils.data import DataLoader, Subset
    calib_size = min(
        cfg.get("calibration", {}).get("num_batches", 50) * batch_size,
        len(task_a.train_dataset),
    )
    calib_dataset = Subset(task_a.train_dataset, list(range(calib_size)))
    calib_loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False)

    importance_scores = estimator.compute(tapss_model.model, calib_loader, device)
    console.print(f"  Scored {len(importance_scores)} parameter tensors.")

    # ── Build rankings ──
    topk_pct = cfg.protection.topk_percent if hasattr(cfg, "protection") else 20.0
    rankings = build_rankings(importance_scores, tapss_model.model, protection_percent=topk_pct)

    # Save rankings
    rankings_dir = os.path.join(output_dir, "rankings")
    rankings.save_csv(os.path.join(rankings_dir, "parameter_rankings.csv"))
    rankings.save_json(os.path.join(rankings_dir, "parameter_rankings.json"))
    rankings.save_html(os.path.join(rankings_dir, "parameter_rankings.html"))
    console.print(f"  Rankings saved to {rankings_dir}/")

    # ── Build protection policy ──
    policy_name = cfg.protection.policy if hasattr(cfg, "protection") else "freeze_topk"
    protection = build_protection_policy(policy_name, importance_scores, cfg)
    console.print(f"  Protection policy: {protection.name}")

    # ── Run TAPSS CL experiment ──
    console.print("\n[bold cyan]Step 3: TAPSS Task B fine-tuning...[/bold cyan]")
    result = pipeline.run(
        model_factory=model_factory,
        task_a=task_a,
        task_b=task_b,
        method_name="tapss",
        importance_scores=importance_scores,
        protection_policy=protection,
        rankings=rankings,
    )

    # ── Report metrics ──
    console.print("\n[bold green]Results:[/bold green]")
    console.print(f"  Task A accuracy (pre-B):  {result.task_a_pre_accuracy:.4f}")
    console.print(f"  Task A accuracy (post-B): {result.task_a_post_accuracy:.4f}")
    console.print(f"  Task B accuracy:           {result.task_b_accuracy:.4f}")
    console.print(f"  Catastrophic Forgetting:   [bold red]{result.forgetting:.4f}[/bold red]")
    console.print(f"  Average Accuracy:          {result.average_accuracy:.4f}")
    console.print(f"  Total time:                {result.total_time_seconds:.1f}s")

    # ── Save visualisations ──
    console.print("\n[dim]Generating visualisations...[/dim]")
    plots_dir = os.path.join(output_dir, "plots")

    plot_layer_importance_heatmap(
        importance_scores,
        save_path=os.path.join(plots_dir, "layer_heatmap.png"),
    )
    plot_importance_histogram(
        importance_scores,
        save_path=os.path.join(plots_dir, "importance_histogram.png"),
    )
    plot_protection_map(
        rankings,
        save_path=os.path.join(plots_dir, "protection_map.png"),
    )
    if result.task_b_train_history:
        plot_training_history(
            result.task_b_train_history,
            method_name="TAPSS",
            task="Task B",
            save_path=os.path.join(plots_dir, "task_b_training.png"),
        )

    # ── Log to tracker ──
    tracker = ExperimentTracker(os.path.join(output_dir, "experiments.db"))
    tracker.log(result, cfg, seed=seed)

    # ── Generate HTML report ──
    report_path = os.path.join(output_dir, "report.html")
    generate_html_report([result], report_path, experiment_name=cfg.get("experiment", {}).get("name", "TAPSS"))

    console.print(f"\n[bold green]✓ Done![/bold green]")
    console.print(f"  Report:  [link=file://{os.path.abspath(report_path)}]{report_path}[/link]")
    console.print(f"  Outputs: {output_dir}/")


if __name__ == "__main__":
    main()
