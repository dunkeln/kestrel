from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from training.sft.records import GenerationStats


logger = logging.getLogger(__name__)


def generation_progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]sft[/bold]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )


def render_summary(console: Console, stats: GenerationStats, gold_path: Path) -> None:
    table = Table(title="sft generation summary")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("total", str(stats.total))
    table.add_row("written", str(stats.written))
    table.add_row("skipped", str(stats.skipped))
    table.add_row("deduped", str(stats.deduped))
    table.add_row("agreement", f"{stats.agreement_rate:.1%}")
    table.add_row("contrastive", str(stats.contrastive))
    table.add_row("records", str(gold_path))
    console.print(table)


def log_event(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}, sort_keys=True))


def stats_fields(stats: GenerationStats) -> dict[str, Any]:
    return {
        "total": stats.total,
        "written": stats.written,
        "skipped": stats.skipped,
        "deduped": stats.deduped,
        "agreement_rate": stats.agreement_rate,
        "contrastive": stats.contrastive,
    }


__all__ = ["generation_progress", "log_event", "render_summary", "stats_fields"]
