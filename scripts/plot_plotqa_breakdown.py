import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.scorers import score_prediction


@dataclass(frozen=True)
class PlotQABreakdown:
    label: str
    total: int
    prediction_parse: float
    model_quality: float
    component_f1: float
    fact_coverage: float
    value_f1: float
    series_f1: float
    x_label_f1: float


@click.command()
@click.argument(
    "jsonl_paths",
    nargs=-1,
    required=True,
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
)
@click.option(
    "--label",
    "labels",
    multiple=True,
    help="Display label for each JSONL path. May be passed once per input file.",
)
@click.option(
    "--title",
    default="PlotQA-Structure Extraction Breakdown",
    show_default=True,
)
@click.option(
    "--subtitle",
    default="Grouped diagnostics from canonical PlotQA-structure fact scoring",
    show_default=True,
)
@click.option(
    "--output-dir",
    default=Path("assets"),
    show_default=True,
    type=click.Path(path_type=Path, file_okay=False),
)
@click.option(
    "--output-name",
    default="plotqa_structure_breakdown",
    show_default=True,
)
def main(
    jsonl_paths: tuple[Path, ...],
    labels: tuple[str, ...],
    title: str,
    subtitle: str,
    output_dir: Path,
    output_name: str,
) -> None:
    """Plot PlotQA-structure component averages from bench JSONL outputs."""
    if labels and len(labels) != len(jsonl_paths):
        raise click.ClickException("--label must be passed once per JSONL path.")

    display_labels = labels or tuple(path.stem for path in jsonl_paths)
    rows = [
        _read_breakdown(path, label=label)
        for path, label in zip(jsonl_paths, display_labels, strict=True)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_name}.png"
    svg_path = output_dir / f"{output_name}.svg"
    _plot_breakdown(rows, title=title, subtitle=subtitle, output_paths=[png_path, svg_path])

    click.echo(f"Wrote {png_path}")
    click.echo(f"Wrote {svg_path}")


def _read_breakdown(path: Path, *, label: str) -> PlotQABreakdown:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise click.ClickException(f"No bench records found in {path}.")

    datasets = {record["dataset"] for record in records}
    if datasets != {"plotqa_structure"}:
        raise click.ClickException(
            f"Expected only PlotQA-structure records in {path}; found {sorted(datasets)}."
        )

    total = len(records)
    score_metadata = [_score_metadata(record) for record in records]
    return PlotQABreakdown(
        label=label,
        total=total,
        prediction_parse=_average_bool(score_metadata, "plotqa_prediction_parse_ok"),
        model_quality=_average_metadata(score_metadata, "plotqa_model_quality_score"),
        component_f1=_average_metadata(score_metadata, "plotqa_component_f1"),
        fact_coverage=_average_metadata(score_metadata, "plotqa_fact_coverage"),
        value_f1=_average_metadata(score_metadata, "plotqa_value_f1"),
        series_f1=_average_metadata(score_metadata, "plotqa_series_f1"),
        x_label_f1=_average_metadata(score_metadata, "plotqa_x_label_f1"),
    )


def _score_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return score_prediction(
        answer_type="structure",
        gold=record["gold"],
        prediction=record["prediction"],
        dataset="plotqa_structure",
        task_type="structure_extraction",
    ).metadata


def _average_bool(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row.get(key, False)) for row in rows) / len(rows)


def _average_metadata(rows: list[dict[str, Any]], key: str) -> float:
    return sum(_numeric(row.get(key, 0.0)) for row in rows) / len(rows)


def _numeric(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _plot_breakdown(
    rows: list[PlotQABreakdown],
    *,
    title: str,
    subtitle: str,
    output_paths: list[Path],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    metrics = [
        ("Prediction parse", [row.prediction_parse for row in rows], "#2563EB"),
        ("Model quality", [row.model_quality for row in rows], "#7C3AED"),
        ("Component F1", [row.component_f1 for row in rows], "#9333EA"),
        ("Fact coverage", [row.fact_coverage for row in rows], "#F97316"),
        ("Value F1", [row.value_f1 for row in rows], "#14B8A6"),
        ("Series F1", [row.series_f1 for row in rows], "#0F766E"),
        ("X label F1", [row.x_label_f1 for row in rows], "#475569"),
    ]

    figure_height = max(5.6, 0.78 * len(metrics) + 2.2)
    figure_width = max(8.8, 1.5 * len(rows) + 7)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    group_centers = list(range(len(metrics)))
    bar_height = min(0.18, 0.65 / max(len(rows), 1))
    offsets = [
        (index - (len(rows) - 1) / 2) * (bar_height + 0.04)
        for index in range(len(rows))
    ]

    for run_index, row in enumerate(rows):
        values = [metric_values[run_index] for _, metric_values, _ in metrics]
        colors = [color for _, _, color in metrics]
        y_positions = [center + offsets[run_index] for center in group_centers]
        bars = ax.barh(
            y_positions,
            values,
            height=bar_height,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            label=f"{row.label} ({row.total})",
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                min(value + 0.018, 0.985),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1%}",
                va="center",
                ha="left",
                fontsize=8.5,
                color="#0F172A",
                fontweight="semibold",
            )

    ax.set_yticks(group_centers, [name for name, _, _ in metrics], fontsize=10.5, color="#0F172A")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.tick_params(axis="x", colors="#64748B", labelsize=9)
    ax.grid(axis="x", color="#CBD5E1", linewidth=0.9, alpha=0.45)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    if len(rows) > 1:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.28),
            ncols=min(len(rows), 3),
            frameon=False,
            fontsize=9,
        )

    fig.text(0.08, 0.965, title, ha="left", va="top", fontsize=17, fontweight="bold", color="#0F172A")
    fig.text(0.08, 0.91, subtitle, ha="left", va="top", fontsize=10, color="#64748B")
    fig.text(
        0.08,
        0.03,
        "Source: Kestrel bench JSONL records",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#94A3B8",
    )
    bottom = 0.24 if len(rows) > 1 else 0.15
    fig.subplots_adjust(left=0.23, right=0.94, top=0.78, bottom=bottom)

    for path in output_paths:
        fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
