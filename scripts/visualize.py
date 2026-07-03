"""
scripts/visualize.py

Standalone visualisation script.
Reads saved rankings and experiment results, generates all plots.

Usage
-----
  python scripts/visualize.py --results-dir outputs/cl_ag_sst2 --all
  python scripts/visualize.py --rankings outputs/cl_ag_sst2/rankings/parameter_rankings.csv
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
logger = logging.getLogger(__name__)


@click.command()
@click.option("--results-dir", "-r", default=None, help="Experiment output directory.")
@click.option("--rankings", default=None, help="Path to parameter rankings CSV.")
@click.option("--output-dir", "-o", default=None, help="Where to save plots.")
@click.option("--all", "all_plots", is_flag=True, help="Generate all plot types.")
@click.option("--show", is_flag=True, help="Display plots interactively.")
def main(results_dir: str | None, rankings: str | None, output_dir: str | None, all_plots: bool, show: bool) -> None:
    """Generate TAPSS visualisations from saved experiment results."""
    from rich.console import Console
    console = Console()
    console.print("[bold cyan]TAPSS Visualisation Generator[/bold cyan]")

    out_dir = output_dir or (os.path.join(results_dir, "plots") if results_dir else "outputs/plots")
    os.makedirs(out_dir, exist_ok=True)

    # ── Rankings plots ──
    if rankings or (results_dir and os.path.exists(
        os.path.join(results_dir, "rankings", "parameter_rankings.csv")
    )):
        rankings_path = rankings or os.path.join(results_dir, "rankings", "parameter_rankings.csv")
        console.print(f"[dim]Loading rankings from {rankings_path}...[/dim]")

        from saliency.base import ImportanceScores
        from saliency.rankings import ParameterRanking, RankingResults

        df = pd.read_csv(rankings_path)
        # Reconstruct ImportanceScores from CSV
        method = df["method"].iloc[0] if "method" in df.columns else "tapss"
        scores_dict = dict(zip(df["param_name"], df["importance_score"]))
        scores = ImportanceScores(scores=scores_dict, method=method)

        from visualization.heatmaps import plot_layer_importance_heatmap
        from visualization.distributions import plot_importance_histogram, plot_gradient_distribution

        console.print("[dim]Generating layer heatmap...[/dim]")
        plot_layer_importance_heatmap(
            scores, save_path=os.path.join(out_dir, "layer_heatmap.png"), show=show
        )

        console.print("[dim]Generating importance histogram...[/dim]")
        plot_importance_histogram(
            scores, save_path=os.path.join(out_dir, "score_histogram.png"), show=show
        )

        console.print("[dim]Generating gradient distribution...[/dim]")
        plot_gradient_distribution(
            scores, save_path=os.path.join(out_dir, "gradient_distribution.png"), show=show
        )

        # Interactive scatter
        from visualization.interactive import interactive_importance_scatter
        rankings_list = []
        for _, row in df.iterrows():
            rankings_list.append(ParameterRanking(
                param_name=row.get("param_name", ""),
                layer=row.get("layer", "other"),
                importance_score=float(row.get("importance_score", 0.0)),
                rank=int(row.get("rank", 0)),
                num_elements=int(row.get("num_elements", 0)),
                is_protected=bool(row.get("is_protected", False)),
                method=method,
            ))
        ranking_results = RankingResults(rankings=rankings_list, method=method)
        interactive_importance_scatter(
            ranking_results,
            save_path=os.path.join(out_dir, "importance_scatter.html"),
        )

        console.print(f"[green]✓ Rankings plots saved to {out_dir}/[/green]")

    if not rankings and not results_dir:
        console.print("[yellow]Provide --results-dir or --rankings to generate plots.[/yellow]")
        return

    console.print(f"\n[bold green]✓ Plots saved to {out_dir}/[/bold green]")


if __name__ == "__main__":
    main()
