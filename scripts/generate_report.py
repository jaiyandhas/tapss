"""
scripts/generate_report.py

Generate an HTML research report from saved experiment results.

Usage
-----
  python scripts/generate_report.py \\
      --results-dir outputs/baseline_comparison \\
      --title "TAPSS vs Baselines"
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option("--results-dir", "-r", required=True, help="Experiment output directory.")
@click.option("--title", "-t", default="TAPSS Research Report", help="Report title.")
@click.option("--output", "-o", default=None, help="Output HTML path.")
def main(results_dir: str, title: str, output: str | None) -> None:
    """Generate a standalone HTML report from saved experiment results."""
    from rich.console import Console
    console = Console()
    console.print(f"[bold cyan]TAPSS Report Generator[/bold cyan]")

    # Load comparison table CSV if present
    csv_path = os.path.join(results_dir, "comparison_table.csv")
    if not os.path.exists(csv_path):
        console.print(f"[red]No comparison_table.csv found in {results_dir}[/red]")
        return

    df = pd.read_csv(csv_path)
    console.print(f"Loaded {len(df)} results from {csv_path}")

    out_path = output or os.path.join(results_dir, "report.html")

    from visualization.report import generate_html_report

    # Reconstruct minimal CLResult-like objects from DataFrame
    results = []
    for _, row in df.iterrows():
        class _FakeResult:
            def __init__(self, r):
                self.method_name = r.get("Method", "unknown")
                self.model_name = "distilbert-base-uncased"
                self.task_a_name = "ag_news"
                self.task_b_name = "sst2"
                self.task_a_pre_accuracy = float(r.get("Task A (Pre)", 0))
                self.task_a_post_accuracy = float(r.get("Task A (Post)", 0))
                self.task_b_accuracy = float(r.get("Task B", 0))
                self.forgetting = float(r.get("Forgetting ↓", 0))
                self.average_accuracy = float(r.get("Avg Acc ↑", 0))
                self.backward_transfer = float(r.get("BWT", 0))
                self.total_time_seconds = float(r.get("Time (s)", 0))
                self.num_trainable_params_task_b = int(r.get("Trainable Params", 0))

        results.append(_FakeResult(row))

    generate_html_report(results, out_path, experiment_name=title)
    console.print(f"[bold green]✓ Report generated: {out_path}[/bold green]")


if __name__ == "__main__":
    main()
