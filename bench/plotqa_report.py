import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.table import Table


@dataclass(frozen=True)
class ClassifiedPlotQARecord:
    score: float
    bucket: str
    metadata: dict[str, Any]
    prediction: str


def classify_plotqa_record(
    *,
    score: float,
    metadata: dict[str, Any],
    prediction: str,
) -> ClassifiedPlotQARecord:
    match (
        metadata.get("plotqa_gold_parse_ok", False),
        bool(prediction.strip()),
        _has_placeholder(prediction),
        _has_structural_tags(prediction),
        metadata.get("plotqa_prediction_parse_ok", False),
        metadata.get("plotqa_exact_match", False),
        score,
    ):
        case (False, _, _, _, _, _, _):
            bucket = "gold_parse_error"
        case (_, False, _, _, _, _, _):
            bucket = "empty_output"
        case (_, _, True, _, _, _, _):
            bucket = "placeholder_output"
        case (_, _, _, False, _, _, _):
            bucket = "no_structural_tags"
        case (_, _, _, _, False, _, _):
            bucket = "unparsed_schema_variant"
        case (_, _, _, _, _, True, _):
            bucket = "exact_match"
        case (_, _, _, _, _, _, 1.0):
            bucket = "semantic_match"
        case (_, _, _, _, _, _, score_value) if score_value > 0.0:
            bucket = "partial_semantic_match"
        case _:
            bucket = "parsed_but_no_semantic_match"

    return ClassifiedPlotQARecord(
        score=score,
        bucket=bucket,
        metadata=metadata,
        prediction=prediction,
    )


def render_plotqa_classification_report(
    *,
    console: Console,
    records: list[ClassifiedPlotQARecord],
    title: str,
) -> None:
    if not records:
        return

    report_console = console if console.size.width >= 220 else Console(width=220)
    total = len(records)
    buckets = Counter(record.bucket for record in records)

    section_tables = [
        _metric_table(
            "Parse",
            [
                ("total", str(total)),
                ("gold parse", _percent(_rate(records, "plotqa_gold_parse_ok"))),
                ("prediction parse", _percent(_rate(records, "plotqa_prediction_parse_ok"))),
                ("exact match", _percent(_rate(records, "plotqa_exact_match"))),
            ],
        ),
        _metric_table(
            "Quality",
            [
                ("semantic match", _percent(sum(record.score == 1.0 for record in records) / total)),
                ("semantic mean", f"{_mean_score(records):.4f}"),
                ("point precision", _percent(_mean_metadata(records, "plotqa_point_precision"))),
                ("point recall", _percent(_mean_metadata(records, "plotqa_point_recall"))),
                ("point f1", _percent(_mean_metadata(records, "plotqa_point_f1"))),
            ],
        ),
        _metric_table(
            "Structure",
            [
                ("series precision", _percent(_mean_metadata(records, "plotqa_series_precision"))),
                ("series recall", _percent(_mean_metadata(records, "plotqa_series_recall"))),
                ("series f1", _percent(_mean_metadata(records, "plotqa_series_f1"))),
                ("x precision", _percent(_mean_metadata(records, "plotqa_x_label_precision"))),
                ("x recall", _percent(_mean_metadata(records, "plotqa_x_label_recall"))),
                ("x f1", _percent(_mean_metadata(records, "plotqa_x_label_f1"))),
                ("count match", _percent(_mean_metadata(records, "plotqa_count_match"))),
                ("gold facts", f"{_mean_metadata(records, 'plotqa_gold_facts'):.2f}"),
                ("pred facts", f"{_mean_metadata(records, 'plotqa_predicted_facts'):.2f}"),
            ],
        ),
        _metric_table(
            "Failures",
            [
                ("over extract", _percent(_over_extraction_rate(records))),
                ("under extract", _percent(_under_extraction_rate(records))),
                ("empty output", _percent(buckets["empty_output"] / total)),
                ("placeholder", _percent(buckets["placeholder_output"] / total)),
                ("unparsed schema", _percent(buckets["unparsed_schema_variant"] / total)),
            ],
        ),
        _bucket_table(buckets, total),
    ]
    grid = Table.grid(expand=True)
    for _ in section_tables:
        grid.add_column()
    grid.add_row(*section_tables)

    report_console.rule(f"[bold]PlotQA Classification Report[/bold] {title}")
    report_console.print(grid)


def _bucket_table(buckets: Counter[str], total: int) -> Table:
    table = Table(title="Failure Buckets")
    table.add_column("bucket")
    table.add_column("count", justify="right")
    table.add_column("rate", justify="right")
    for bucket, count in buckets.most_common():
        table.add_row(bucket, str(count), _percent(count / total))
    return table


def _metric_table(title: str, rows: list[tuple[str, str]]) -> Table:
    table = Table(title=title)
    table.add_column("metric")
    table.add_column("value", justify="right")
    for metric, value in rows:
        table.add_row(metric, value)
    return table


def _rate(records: list[ClassifiedPlotQARecord], metadata_key: str) -> float:
    return sum(bool(record.metadata.get(metadata_key, False)) for record in records) / len(records)


def _mean_score(records: list[ClassifiedPlotQARecord]) -> float:
    return sum(record.score for record in records) / len(records)


def _mean_metadata(records: list[ClassifiedPlotQARecord], metadata_key: str) -> float:
    return sum(_numeric(record.metadata.get(metadata_key, 0.0)) for record in records) / len(records)


def _over_extraction_rate(records: list[ClassifiedPlotQARecord]) -> float:
    return sum(
        _numeric(record.metadata.get("plotqa_predicted_facts", 0))
        > _numeric(record.metadata.get("plotqa_gold_facts", 0))
        for record in records
    ) / len(records)


def _under_extraction_rate(records: list[ClassifiedPlotQARecord]) -> float:
    return sum(
        _numeric(record.metadata.get("plotqa_predicted_facts", 0))
        < _numeric(record.metadata.get("plotqa_gold_facts", 0))
        for record in records
    ) / len(records)


def _numeric(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _has_placeholder(value: str) -> bool:
    text = str(value).lower()
    return bool(re.search(r"\b(?:x|y|x1|x2|y1|y2|series name)\b", text))


def _has_structural_tags(value: str) -> bool:
    text = str(value).lower()
    return any(tag in text for tag in ("<s_y", "<s_x", "<s_name", "<series"))


def _percent(value: float) -> str:
    return f"{value:.1%}"
