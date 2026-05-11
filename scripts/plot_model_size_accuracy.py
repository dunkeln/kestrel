import json
from dataclasses import dataclass
from pathlib import Path

import click


@dataclass(frozen=True)
class AccuracyPoint:
    model_label: str
    dataset: str
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


@click.command()
@click.argument("labeled_jsonl_paths", nargs=-1, required=True)
@click.option(
    "--title",
    default="Qwen-VL Base Model Size Accuracy",
    show_default=True,
)
@click.option(
    "--subtitle",
    default="Typed exact-match accuracy over 256 normalized test samples per dataset",
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
    default="runpod_base_model_size_wide_bench_accuracy",
    show_default=True,
)
def main(
    labeled_jsonl_paths: tuple[str, ...],
    title: str,
    subtitle: str,
    output_dir: Path,
    output_name: str,
) -> None:
    """Plot grouped benchmark accuracy bars from label:path JSONL inputs."""
    rows = [_read_labeled_accuracy(value) for value in labeled_jsonl_paths]
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{output_name}.png"
    svg_path = output_dir / f"{output_name}.svg"
    _plot_accuracy(rows, title=title, subtitle=subtitle, output_paths=[png_path, svg_path])

    click.echo(f"Wrote {png_path}")
    click.echo(f"Wrote {svg_path}")


def _read_labeled_accuracy(value: str) -> AccuracyPoint:
    label, path = _split_labeled_path(value)
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise click.ClickException(f"No bench records found in {path}.")

    datasets = {record["dataset"] for record in records}
    if len(datasets) != 1:
        raise click.ClickException(f"Expected one dataset per JSONL file: {path}.")

    return AccuracyPoint(
        model_label=label,
        dataset=_display_dataset(datasets.pop()),
        correct=sum(bool(record["correct"]) for record in records),
        total=len(records),
    )


def _split_labeled_path(value: str) -> tuple[str, Path]:
    if ":" not in value:
        raise click.ClickException(
            "Expected each input as label:path, for example tiny:artifacts/evals/bench/base_chartqa_256.jsonl."
        )
    label, path_text = value.split(":", 1)
    label = label.strip()
    path = Path(path_text)
    if not label:
        raise click.ClickException(f"Missing label in input: {value}")
    if not path.exists():
        raise click.ClickException(f"JSONL path does not exist: {path}")
    return label, path


def _plot_accuracy(
    rows: list[AccuracyPoint],
    *,
    title: str,
    subtitle: str,
    output_paths: list[Path],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    dataset_order = ["ChartQA", "FigureQA", "ChartBench", "MMC-Benchmark"]
    model_order = _model_order(rows)
    rows_by_key = {(row.dataset, row.model_label): row for row in rows}

    figure_width = max(9.8, 1.8 * len(dataset_order) + 3.8)
    fig, ax = plt.subplots(figsize=(figure_width, 5.6), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = ["#2563EB", "#14B8A6", "#F97316", "#7C3AED", "#475569"]
    group_width = 0.74
    bar_width = group_width / max(len(model_order), 1)
    x_centers = list(range(len(dataset_order)))

    for model_index, model_label in enumerate(model_order):
        offset = (model_index - (len(model_order) - 1) / 2) * bar_width
        x_values = [center + offset for center in x_centers]
        points = [rows_by_key.get((dataset, model_label)) for dataset in dataset_order]
        accuracies = [point.accuracy if point is not None else 0.0 for point in points]
        displayed = [max(value, 0.004) for value in accuracies]
        bars = ax.bar(
            x_values,
            displayed,
            width=bar_width * 0.88,
            color=colors[model_index % len(colors)],
            edgecolor="white",
            linewidth=0.8,
            label=model_label,
        )
        for bar, accuracy, point in zip(bars, accuracies, points, strict=True):
            if point is None:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(accuracy + 0.025, 0.97),
                f"{accuracy:.0%}",
                ha="center",
                va="bottom",
                color="#0F172A",
                fontsize=8.5,
                fontweight="semibold",
            )

    ax.set_xticks(x_centers, dataset_order)
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

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=len(model_order), frameon=False, fontsize=9.5)

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
    fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.22)

    for path in output_paths:
        fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _model_order(rows: list[AccuracyPoint]) -> list[str]:
    preferred = ["2B", "3B", "7B", "tiny", "small", "aight"]
    labels = {row.model_label for row in rows}
    ordered = [label for label in preferred if label in labels]
    ordered.extend(sorted(labels - set(ordered)))
    return ordered


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
