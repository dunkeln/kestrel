import json
from dataclasses import dataclass
from pathlib import Path

import click


@dataclass(frozen=True)
class BenchAccuracy:
    dataset: str
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


@click.command()
@click.argument(
    "jsonl_paths",
    nargs=-1,
    required=True,
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
)
@click.option(
    "--title",
    default="Qwen-VL Wide Bench Accuracy",
    show_default=True,
)
@click.option(
    "--subtitle",
    default="Typed exact-match accuracy over normalized test samples",
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
    default="wide_bench_accuracy",
    show_default=True,
)
def main(
    jsonl_paths: tuple[Path, ...],
    title: str,
    subtitle: str,
    output_dir: Path,
    output_name: str,
) -> None:
    """Plot benchmark accuracy bars from bench JSONL outputs."""
    rows = [_read_accuracy(path) for path in jsonl_paths]
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{output_name}.png"
    svg_path = output_dir / f"{output_name}.svg"
    _plot_accuracy(rows, title=title, subtitle=subtitle, output_paths=[png_path, svg_path])

    click.echo(f"Wrote {png_path}")
    click.echo(f"Wrote {svg_path}")


def _read_accuracy(path: Path) -> BenchAccuracy:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise click.ClickException(f"No bench records found in {path}.")

    datasets = {record["dataset"] for record in records}
    if len(datasets) != 1:
        raise click.ClickException(f"Expected one dataset per JSONL file: {path}.")

    return BenchAccuracy(
        dataset=_display_dataset(datasets.pop()),
        correct=sum(bool(record["correct"]) for record in records),
        total=len(records),
    )


def _plot_accuracy(
    rows: list[BenchAccuracy],
    *,
    title: str,
    subtitle: str,
    output_paths: list[Path],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    rows = sorted(rows, key=lambda row: row.dataset)
    labels = [row.dataset for row in rows]
    accuracies = [row.accuracy for row in rows]
    displayed_accuracies = [max(accuracy, 0.004) for accuracy in accuracies]

    figure_width = max(8.5, 1.35 * len(rows) + 3)
    fig, ax = plt.subplots(figsize=(figure_width, 5.2), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(
        labels,
        displayed_accuracies,
        color="#2563EB",
        width=0.62,
        edgecolor="#1E40AF",
        linewidth=0.8,
    )

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.9, alpha=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelrotation=0, labelsize=10, colors="#334155")
    ax.tick_params(axis="y", labelsize=9, colors="#64748B")
    ax.set_ylabel("Accuracy", color="#334155", fontsize=10, labelpad=10)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")

    for bar, row in zip(bars, rows, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(row.accuracy + 0.025, 0.97),
            f"{row.accuracy:.1%}\n{row.correct}/{row.total}",
            ha="center",
            va="bottom",
            color="#0F172A",
            fontsize=9,
            fontweight="semibold",
        )

    fig.text(0.08, 0.965, title, ha="left", va="top", fontsize=17, fontweight="bold", color="#0F172A")
    fig.text(0.08, 0.915, subtitle, ha="left", va="top", fontsize=10, color="#64748B")
    fig.text(
        0.08,
        0.03,
        "Source: Kestrel bench JSONL records",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#94A3B8",
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.16)

    for path in output_paths:
        fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _display_dataset(dataset: str) -> str:
    names = {
        "chartqa": "ChartQA",
        "figureqa": "FigureQA",
        "chartbench": "ChartBench",
        "mmc_benchmark": "MMC-Benchmark",
        "plotqa_qa": "PlotQA QA",
        "plotqa_structure": "PlotQA Structure",
    }
    return names.get(dataset, dataset)


if __name__ == "__main__":
    main()
