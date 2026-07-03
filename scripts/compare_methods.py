"""
scripts/compare_methods.py

Run all baseline methods plus TAPSS on the same CL task sequence
and produce a comprehensive comparison report.

Usage
-----
  python scripts/compare_methods.py \\
      --config configs/experiment/baseline_comparison.yaml

  python scripts/compare_methods.py \\
      --config configs/experiment/baseline_comparison.yaml \\
      --methods tapss vanilla_lora ewc
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
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from task_datasets.loaders import load_task_dataset
from models.registry import get_model_and_tokenizer, ModelRegistry
from saliency.tapss import TAPSSEstimator
from saliency.rankings import build_rankings
from peft_modules.protection import build_protection_policy
from continual_learning.pipeline import ContinualLearningPipeline, CLResult
from continual_learning.baselines import (
    VanillaLoRABaseline,
    NaiveFinetuningBaseline,
    RandomProtectionBaseline,
)
from continual_learning.ewc import EWCBaseline
from evaluation.tables import build_comparison_table, format_results_table, to_latex
from evaluation.tracker import ExperimentTracker
from visualization.continual import (
    plot_forgetting_over_methods, plot_task_accuracy_comparison, plot_radar_chart
)
from visualization.interactive import interactive_comparison_bar, interactive_radar
from visualization.report import generate_html_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

AVAILABLE_METHODS = ["tapss", "vanilla_lora", "naive_finetuning", "ewc", "random_protection"]


def _resolve_defaults(config_dir: Path, defaults_list: list) -> dict:
    composed = {}
    for item in defaults_list:
        if isinstance(item, dict):
            for k, v in item.items():
                if k == "_self_":
                    continue
                sub_path = config_dir / k / f"{v}.yaml"
                if sub_path.exists():
                    with open(sub_path) as sf:
                        sub_raw = yaml.safe_load(sf) or {}
                    nested = _resolve_defaults(config_dir, sub_raw.get("defaults", []))
                    composed.update(nested)
                    composed.update({nk: nv for nk, nv in sub_raw.items() if nk != "defaults"})
        elif isinstance(item, str):
            if item == "_self_":
                continue
            if item.startswith("/"):
                sub_path = config_dir / f"{item[1:]}.yaml"
                if sub_path.exists():
                    with open(sub_path) as sf:
                        sub_raw = yaml.safe_load(sf) or {}
                    nested = _resolve_defaults(config_dir, sub_raw.get("defaults", []))
                    composed.update(nested)
                    composed.update({nk: nv for nk, nv in sub_raw.items() if nk != "defaults"})
    return composed


def _load_cfg(config_path: str, overrides: tuple[str]) -> OmegaConf:
    """Load YAML config, recursively resolve defaults, and apply overrides."""
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    config_dir = Path(__file__).resolve().parent.parent / "configs"
    composed = _resolve_defaults(config_dir, raw.get("defaults", []))
    composed.update({k: v for k, v in raw.items() if k != "defaults"})
    
    cfg = OmegaConf.create(composed)
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        OmegaConf.update(cfg, key, yaml.safe_load(value), merge=True)
    return cfg



def _set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


@click.command()
@click.option("--config", "-c", default="configs/experiment/baseline_comparison.yaml")
@click.option("--methods", "-m", multiple=True, default=AVAILABLE_METHODS, help="Methods to run.")
@click.option("--output-dir", "-o", default=None, help="Override output directory.")
@click.option("--override", "-v", multiple=True, help="Config overrides in key=value format.")
@click.option("--skip-perturbation", is_flag=True)
def main(config: str, methods: tuple, output_dir: str | None, override: tuple, skip_perturbation: bool) -> None:
    """
    Run all continual learning methods and produce a comparison report.
    """
    console = Console()
    console.print("[bold cyan]TAPSS Method Comparison[/bold cyan]")

    cfg = _load_cfg(config, override)
    seed = cfg.get("experiment", {}).get("seed", 42)
    out_dir = output_dir or cfg.get("experiment", {}).get("output_dir", "outputs/comparison")

    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    console.print(f"Device: {device} | Seed: {seed} | Output: {out_dir}")

    # Load tokenizer
    model_name = cfg.model.name if hasattr(cfg, "model") else "distilbert-base-uncased"
    resolved_name = ModelRegistry.resolve_model_name(model_name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(resolved_name)

    # Load datasets
    task_a_name = cfg.get("continual_learning", {}).get("task_a_dataset", "ag_news")
    task_b_name = cfg.get("continual_learning", {}).get("task_b_dataset", "sst2")
    batch_size = cfg.training.batch_size if hasattr(cfg, "training") else 16

    train_size = cfg.get("dataset", {}).get("train_size", None)
    val_size = cfg.get("dataset", {}).get("val_size", None)
    test_size = cfg.get("dataset", {}).get("test_size", None)

    console.print(f"[dim]Loading datasets: {task_a_name} → {task_b_name}[/dim]")
    task_a = load_task_dataset(
        task_a_name, tokenizer, batch_size=batch_size, seed=seed,
        train_size=train_size, val_size=val_size, test_size=test_size
    )
    task_b = load_task_dataset(
        task_b_name, tokenizer, batch_size=batch_size, seed=seed,
        train_size=train_size, val_size=val_size, test_size=test_size
    )

    def model_factory():
        return get_model_and_tokenizer(cfg, task_a.num_labels, device)

    pipeline = ContinualLearningPipeline(cfg, device)
    tracker = ExperimentTracker(os.path.join(out_dir, "experiments.db"))

    # Pre-compute TAPSS scores (shared between TAPSS + random_protection)
    console.print("\n[dim]Computing TAPSS importance scores (pre-computation)...[/dim]")
    temp_model, _ = model_factory()

    from torch.utils.data import DataLoader, Subset
    calib_size = min(50 * batch_size, len(task_a.train_dataset))
    calib_loader = DataLoader(
        Subset(task_a.train_dataset, list(range(calib_size))), batch_size=batch_size
    )

    estimator = TAPSSEstimator(cfg, skip_perturbation=skip_perturbation)
    tapss_scores = estimator.compute(temp_model.model, calib_loader, device)
    topk_pct = cfg.protection.topk_percent if hasattr(cfg, "protection") else 20.0
    tapss_rankings = build_rankings(tapss_scores, temp_model.model, protection_percent=topk_pct)

    # EWC Fisher (also pre-computed on Task A data)
    ewc_baseline = EWCBaseline(cfg, device) if "ewc" in methods else None

    all_results: list[CLResult] = []

    for method_name in methods:
        console.rule(f"[bold]{method_name.upper()}[/bold]")

        try:
            if method_name == "tapss":
                policy_name = cfg.protection.policy if hasattr(cfg, "protection") else "freeze_topk"
                protection = build_protection_policy(policy_name, tapss_scores, cfg)
                result = pipeline.run(
                    model_factory, task_a, task_b,
                    method_name="tapss",
                    importance_scores=tapss_scores,
                    protection_policy=protection,
                    rankings=tapss_rankings,
                )

            elif method_name == "vanilla_lora":
                baseline = VanillaLoRABaseline(cfg, device)
                result = pipeline.run(
                    model_factory, task_a, task_b,
                    method_name="vanilla_lora",
                )

            elif method_name == "naive_finetuning":
                result = pipeline.run(
                    model_factory, task_a, task_b,
                    method_name="naive_finetuning",
                    apply_lora=False,
                )

            elif method_name == "ewc":
                ewc = EWCBaseline(cfg, device)
                # Train Task A first, then prepare EWC
                from transformers import AutoModelForSequenceClassification
                ewc_model = AutoModelForSequenceClassification.from_pretrained(
                    resolved_name, num_labels=task_a.num_labels, ignore_mismatched_sizes=True
                ).to(device)
                pipeline.trainer.train(
                    ewc_model, task_a.train_loader, task_a.val_loader, run_name="ewc_task_a"
                )
                ewc.prepare(ewc_model, calib_loader)
                result = pipeline.run(
                    model_factory, task_a, task_b,
                    method_name="ewc",
                    importance_scores=ewc.fisher_as_importance_scores(),
                )

            elif method_name == "random_protection":
                baseline = RandomProtectionBaseline(cfg, device, seed=seed)
                from saliency.base import ImportanceScores
                import random
                rng = random.Random(seed)
                rand_scores = ImportanceScores(
                    scores=ImportanceScores.normalise(
                        {n: rng.random() for n in tapss_scores.scores}
                    ),
                    method="random",
                )
                rand_policy = build_protection_policy("freeze_topk", rand_scores, cfg)
                result = pipeline.run(
                    model_factory, task_a, task_b,
                    method_name="random_protection",
                    importance_scores=rand_scores,
                    protection_policy=rand_policy,
                )

            else:
                console.print(f"[yellow]Unknown method: {method_name}[/yellow]")
                continue

            all_results.append(result)
            tracker.log(result, cfg, seed=seed)

            console.print(
                f"  Forgetting: [bold red]{result.forgetting:.4f}[/bold red]  "
                f"Task B: [bold green]{result.task_b_accuracy:.4f}[/bold green]"
            )

        except Exception as e:
            console.print(f"[red]ERROR in {method_name}: {e}[/red]")
            logger.exception(f"Error running {method_name}")

    if not all_results:
        console.print("[red]No results to compare.[/red]")
        return

    # ── Generate comparison outputs ──
    console.rule("[bold cyan]Comparison Report[/bold cyan]")

    df = build_comparison_table(all_results)
    console.print(format_results_table(df))

    # Save CSV
    df.to_csv(os.path.join(out_dir, "comparison_table.csv"), index=False)

    # Save LaTeX
    with open(os.path.join(out_dir, "comparison_table.tex"), "w") as f:
        f.write(to_latex(df))

    # Static plots
    plots_dir = os.path.join(out_dir, "plots")
    plot_forgetting_over_methods(all_results, save_path=os.path.join(plots_dir, "forgetting.png"))
    plot_task_accuracy_comparison(all_results, save_path=os.path.join(plots_dir, "task_accuracy.png"))
    plot_radar_chart(all_results, save_path=os.path.join(plots_dir, "radar.png"))

    # Interactive Plotly
    interactive_comparison_bar(all_results, save_path=os.path.join(out_dir, "comparison_bar.html"))
    interactive_radar(all_results, save_path=os.path.join(out_dir, "radar_chart.html"))

    # Full HTML report
    report_path = generate_html_report(
        all_results,
        os.path.join(out_dir, "comparison_report.html"),
        experiment_name="Baseline Comparison",
        cfg=cfg,
    )

    console.print(f"\n[bold green]✓ Comparison complete![/bold green]")
    console.print(f"  Report: {report_path}")
    console.print(f"  Table:  {os.path.join(out_dir, 'comparison_table.csv')}")


if __name__ == "__main__":
    main()
