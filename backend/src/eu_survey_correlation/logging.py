"""Rich-based logging for the pipeline scripts.

Usage:
    from eu_survey_correlation.logging import console, log, print_match, print_summary_table

    log.info("Loaded 341 matches")
    log.success("Model saved")
    log.warning("No candidates found")

    print_match(42, question, vote_summary, score=0.72, days=120, decision="ACCEPT")
"""

from __future__ import annotations

import logging
from typing import Any

from rich.columns import Columns
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# ── Logger with Rich handler (no timestamp — saves space) ─────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=True,
            show_time=False,
        )
    ],
)
log = logging.getLogger("eu_survey")


# ── Match display ─────────────────────────────────────────────────────
def print_match(
    idx: int | str,
    question: str,
    vote_summary: str,
    *,
    score: float | None = None,
    days: float | None = None,
    overlap: float | None = None,
    probability: float | None = None,
    decision: str | None = None,
    threshold: float | None = None,
) -> None:
    """Print a side-by-side match comparison with panels."""
    # Header
    parts = [f"[bold blue]Match #{idx}[/]"]
    if score is not None:
        parts.append(f"sim={score:.3f}")
    if probability is not None:
        parts.append(f"P={probability:.3f}")
    console.rule(" ".join(parts))

    # Side-by-side panels (adapt width to terminal)
    width = min((console.width - 4) // 2, 60)
    console.print(
        Columns(
            [
                Panel(question, title="[cyan]Question[/]", width=width, padding=(0, 1)),
                Panel(
                    vote_summary,
                    title="[green]Vote Summary[/]",
                    width=width,
                    padding=(0, 1),
                ),
            ]
        )
    )

    # Footer metrics
    metrics: list[str] = []
    if score is not None:
        metrics.append(f"score={score:.3f}")
    if days is not None:
        metrics.append(f"days={int(days)}")
    if overlap is not None:
        metrics.append(f"overlap={overlap:.2f}")
    if probability is not None:
        metrics.append(f"P(accept)={probability:.3f}")

    if decision:
        color = (
            "green"
            if decision.upper() == "ACCEPT"
            else "red" if decision.upper() == "REFUSE" else "yellow"
        )
        metrics.append(f"[bold {color}]{decision.upper()}[/]")
    elif threshold is not None and probability is not None:
        if probability >= threshold:
            metrics.append("[bold green]ACCEPT[/]")
        else:
            metrics.append("[bold red]REFUSE[/]")

    if metrics:
        console.print(f"[dim]{'  '.join(metrics)}[/]")
    console.print()


# ── Candidate table ───────────────────────────────────────────────────
def print_candidates_table(
    candidates: list[dict],
    threshold: float = 0.5,
    title: str = "Active Learning Candidates",
    max_rows: int = 2,
) -> None:
    """Print a table of candidate pairs ranked by uncertainty."""
    table = Table(title=title, show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("P(acc)", justify="right", style="bold", width=7)
    table.add_column("Question", max_width=45, overflow="ellipsis")
    table.add_column("Vote", max_width=45, overflow="ellipsis")
    table.add_column("Decision", justify="center", width=10)

    for i, c in enumerate(candidates[:max_rows], 1):
        p = c.get("predicted_probability", 0)
        q = str(c.get("question_clean", ""))
        v = str(c.get("vote_summary_clean", c.get("summary_clean", "")))
        unc = abs(p - threshold)

        if unc < 0.05:
            decision = "[yellow]UNCERTAIN[/]"
        elif p >= threshold:
            decision = "[green]ACCEPT[/]"
        else:
            decision = "[red]REFUSE[/]"

        table.add_row(str(i), f"{p:.3f}", q, v, decision)

    console.print(table)


# ── Summary / section helpers ─────────────────────────────────────────
def print_section(title: str) -> None:
    """Print a section separator."""
    console.rule(f"[bold]{title}[/]")


def print_kv(label: str, value: Any, style: str = "") -> None:
    """Print a key-value pair."""
    if style:
        console.print(f"  [dim]{label}:[/] [{style}]{value}[/]")
    else:
        console.print(f"  [dim]{label}:[/] {value}")


def print_metrics(
    metrics: dict[str, dict[str, float]], title: str = "Cross-Validation Results"
) -> None:
    """Print CV metrics as a Rich table."""
    table = Table(title=title)
    table.add_column("Metric", style="bold")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right", style="dim")

    for name in ["f1", "precision", "recall", "pr_auc"]:
        if name in metrics:
            m = metrics[name]
            table.add_row(name.upper(), f"{m['mean']:.3f}", f"±{m['std']:.3f}")

    console.print(table)
