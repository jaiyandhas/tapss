"""
scripts/evaluate.py

Evaluate a trained model on a dataset and print metrics.

Usage
-----
  python scripts/evaluate.py \\
      --checkpoint outputs/cl_ag_sst2/checkpoints/tapss_task_b.pt \\
      --dataset sst2 \\
      --model distilbert-base-uncased
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
from evaluation.metrics import evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--checkpoint", "-c", required=True, help="Path to model checkpoint (.pt file).")
@click.option("--dataset", "-d", default="sst2", help="Dataset to evaluate on.")
@click.option("--model", "-m", default="distilbert-base-uncased", help="Base model name.")
@click.option("--batch-size", default=32, help="Evaluation batch size.")
@click.option("--seed", default=42)
@click.option("--split", default="test", type=click.Choice(["train", "val", "test"]))
def main(checkpoint: str, dataset: str, model: str, batch_size: int, seed: int, split: str) -> None:
    """Evaluate a TAPSS checkpoint on a given dataset split."""
    from rich.console import Console
    console = Console()
    console.print(f"[bold cyan]TAPSS Evaluator[/bold cyan]")
    console.print(f"  Checkpoint: {checkpoint}")
    console.print(f"  Dataset:    {dataset}")
    console.print(f"  Split:      {split}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = ModelRegistry.resolve_model_name(model)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(resolved)
    task_data = load_task_dataset(dataset, tokenizer, batch_size=batch_size, seed=seed)

    # Load model
    hf_model = AutoModelForSequenceClassification.from_pretrained(
        resolved, num_labels=task_data.num_labels, ignore_mismatched_sizes=True
    )

    # Load checkpoint
    if not os.path.exists(checkpoint):
        console.print(f"[red]Checkpoint not found: {checkpoint}[/red]")
        return

    state = torch.load(checkpoint, map_location=device)
    if "model_state_dict" in state:
        hf_model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        hf_model.load_state_dict(state, strict=False)

    hf_model.to(device)

    loader_map = {
        "train": task_data.train_loader,
        "val": task_data.val_loader,
        "test": task_data.test_loader,
    }
    loader = loader_map[split]

    loss, acc = evaluate_model(hf_model, loader, device)

    console.print(f"\n[bold green]Results:[/bold green]")
    console.print(f"  Loss:     {loss:.6f}")
    console.print(f"  Accuracy: {acc:.4f} ({acc * 100:.2f}%)")


if __name__ == "__main__":
    main()
