from collections import Counter
from typing import Any

from rich.console import Console
from rich.table import Table


def render_plotqa_qa_report(
    *,
    console: Console,
    records: list[dict[str, Any]],
    title: str,
) -> None:
    if not records:
        return

    total = len(records)
    score_rows = [record.get("metadata", {}).get("score", {}) for record in records]
    answer_modes = Counter(row.get("plotqa_qa_answer_mode", "unknown") for row in score_rows)
    numeric_rows = [
        row for row in score_rows if "plotqa_qa_numeric_prediction_parse_ok" in row
    ]

    table = Table(title=f"PlotQA QA Report {title}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("total", str(total))
    table.add_row("exact match", _percent(sum(bool(row.get("plotqa_qa_exact_match")) for row in score_rows) / total))
    table.add_row("numeric rows", str(len(numeric_rows)))
    if numeric_rows:
        table.add_row(
            "numeric prediction parse",
            _percent(
                sum(bool(row.get("plotqa_qa_numeric_prediction_parse_ok")) for row in numeric_rows)
                / len(numeric_rows)
            ),
        )
        mean_absolute_error = _mean_present(numeric_rows, "plotqa_qa_absolute_error")
        table.add_row(
            "mean absolute error",
            f"{mean_absolute_error:.4f}" if mean_absolute_error is not None else "N/A",
        )
    for mode, count in answer_modes.most_common():
        table.add_row(mode, f"{count} ({_percent(count / total)})")

    console.print(table)


def _mean_present(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), int | float)]
    return sum(values) / len(values) if values else None


def _percent(value: float) -> str:
    return f"{value:.1%}"
