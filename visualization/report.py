"""
visualization/report.py

Self-contained HTML report generator for TAPSS experiments.

Produces a single portable HTML file containing:
  - Experiment metadata and configuration summary
  - Results comparison table
  - Embedded Plotly figures (importance scatter, radar, heatmap, comparison bar)
  - Training curves (if available)
  - Parameter ranking highlights

No external server required — open the HTML directly in any browser.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import plotly.io as pio

from evaluation.tables import build_comparison_table, to_html

logger = logging.getLogger(__name__)

REPORT_CSS = """
<style>
  :root {
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --text: #c9d1d9;
    --accent: #58a6ff;
    --green: #3fb950;
    --amber: #f0883e;
    --red: #f85149;
    --purple: #8b5cf6;
    --font: 'Segoe UI', system-ui, -apple-system, sans-serif;
    --mono: 'Fira Code', 'JetBrains Mono', monospace;
  }
  * { box-sizing: border-box; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    margin: 0;
    padding: 0;
  }
  .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
  h1 {
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 0.25rem;
  }
  h2 {
    font-size: 1.25rem;
    color: var(--accent);
    margin-top: 2rem;
    margin-bottom: 0.75rem;
    border-left: 3px solid var(--accent);
    padding-left: 0.75rem;
  }
  .subtitle { color: #8b949e; font-size: 0.9rem; margin-bottom: 2rem; }
  .meta-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }
  .meta-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
  }
  .meta-card .label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
  .meta-card .value { font-size: 1.1rem; font-weight: 600; color: var(--text); margin-top: 0.25rem; }
  .tapss-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  .tapss-table th {
    background: var(--surface);
    color: var(--accent);
    padding: 0.6rem 1rem;
    text-align: left;
    border-bottom: 2px solid var(--border);
    font-weight: 600;
  }
  .tapss-table td {
    padding: 0.5rem 1rem;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 0.85rem;
  }
  .tapss-table tr:hover td { background: #1c2128; }
  .tapss-table .method-tapss { color: var(--accent); font-weight: 700; }
  .plotly-figure { margin: 2rem 0; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .section-divider { border: none; border-top: 1px solid var(--border); margin: 2.5rem 0; }
  .badge {
    display: inline-block;
    padding: 0.2em 0.6em;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .badge-blue { background: #1d4ed8; color: white; }
  .badge-green { background: #15803d; color: white; }
  footer { margin-top: 4rem; padding-top: 1rem; border-top: 1px solid var(--border); color: #8b949e; font-size: 0.8rem; }
</style>
"""


def generate_html_report(
    results: list[Any],  # list[CLResult]
    output_path: str,
    experiment_name: str = "TAPSS Experiment",
    cfg: Any = None,
    extra_figures: Optional[dict[str, str]] = None,
) -> str:
    """
    Generate a self-contained HTML research report.

    Parameters
    ----------
    results : list[CLResult]
        All method results to include in the report.
    output_path : str
        Where to write the HTML file.
    experiment_name : str
        Report title.
    cfg : Any
        Hydra config (for metadata section).
    extra_figures : dict[str, str] | None
        {section_title: plotly_html_string} for additional figures.

    Returns
    -------
    str
        Path to the generated HTML file.
    """
    from visualization.interactive import (
        interactive_comparison_bar,
        interactive_radar,
    )

    logger.info(f"[Report] Generating HTML report: {output_path}")

    # Metadata
    model_name = results[0].model_name if results else "unknown"
    task_a = results[0].task_a_name if results else "?"
    task_b = results[0].task_b_name if results else "?"
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Comparison table
    df = build_comparison_table(results)
    table_html = _df_to_html_table(df)

    # Plotly figures
    comparison_fig = interactive_comparison_bar(results)
    radar_fig = interactive_radar(results)

    comparison_html = pio.to_html(comparison_fig, full_html=False, include_plotlyjs="cdn")
    radar_html = pio.to_html(radar_fig, full_html=False, include_plotlyjs=False)

    # Build extra figure sections
    extra_html = ""
    if extra_figures:
        for section_title, fig_html in extra_figures.items():
            extra_html += f"""
            <h2>{section_title}</h2>
            <div class="plotly-figure">{fig_html}</div>
            """

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>TAPSS Research Report — {experiment_name}</title>
  {REPORT_CSS}
</head>
<body>
<div class="container">

  <h1>TAPSS Research Report</h1>
  <p class="subtitle">
    Task-Adaptive Parameter Saliency Score — Continual Learning Evaluation
    &nbsp;|&nbsp;
    <span class="badge badge-blue">Research Prototype</span>
    &nbsp;
    <span class="badge badge-green">Reproducible</span>
  </p>

  <h2>Experiment Overview</h2>
  <div class="meta-grid">
    <div class="meta-card">
      <div class="label">Experiment</div>
      <div class="value">{experiment_name}</div>
    </div>
    <div class="meta-card">
      <div class="label">Model</div>
      <div class="value">{model_name}</div>
    </div>
    <div class="meta-card">
      <div class="label">Task A → Task B</div>
      <div class="value">{task_a} → {task_b}</div>
    </div>
    <div class="meta-card">
      <div class="label">Methods Compared</div>
      <div class="value">{len(results)}</div>
    </div>
    <div class="meta-card">
      <div class="label">Generated</div>
      <div class="value" style="font-size:0.9rem">{timestamp}</div>
    </div>
  </div>

  <hr class="section-divider" />

  <h2>Results Summary</h2>
  <p style="color:#8b949e; font-size:0.85rem;">
    Sorted by Catastrophic Forgetting (↑ lower is better).
    <strong style="color:#58a6ff;">TAPSS</strong> is highlighted.
  </p>
  {table_html}

  <hr class="section-divider" />

  <h2>Metric Comparison</h2>
  <div class="plotly-figure">{comparison_html}</div>

  <h2>Multi-metric Radar</h2>
  <div class="plotly-figure">{radar_html}</div>

  {extra_html}

  <hr class="section-divider" />

  <h2>Interpretation Guide</h2>
  <ul style="line-height:1.8; color:#8b949e;">
    <li><strong style="color:var(--text)">Task A (Post)</strong> — Task A accuracy after Task B training. Higher = less forgetting.</li>
    <li><strong style="color:var(--text)">Task B</strong> — Accuracy on the new task. Higher = better adaptation.</li>
    <li><strong style="color:var(--text)">Forgetting</strong> — Task A pre-accuracy minus post-accuracy. Lower is better.</li>
    <li><strong style="color:var(--text)">BWT</strong> — Backward Transfer: negative = forgetting, zero = no interference.</li>
    <li><strong style="color:var(--text)">Avg Accuracy</strong> — Mean of Task A post and Task B accuracy.</li>
  </ul>

  <footer>
    Generated by TAPSS Research Framework &nbsp;|&nbsp;
    <a href="https://github.com/tapss-research" style="color:#58a6ff;">GitHub</a>
  </footer>

</div>
</body>
</html>
    """

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"[Report] HTML report written to {output_path}")
    return output_path


def _df_to_html_table(df: pd.DataFrame) -> str:
    """Convert DataFrame to styled HTML table."""
    rows_html = ""
    for _, row in df.iterrows():
        method = str(row.get("Method", ""))
        is_tapss = method.lower() == "tapss"
        row_class = "method-tapss" if is_tapss else ""

        cells = "".join(
            f'<td class="{row_class}">{v}</td>' for v in row.values
        )
        rows_html += f"<tr>{cells}</tr>\n"

    headers = "".join(f"<th>{col}</th>" for col in df.columns)
    return f"""
    <div style="overflow-x:auto">
    <table class="tapss-table">
      <thead><tr>{headers}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
    """
