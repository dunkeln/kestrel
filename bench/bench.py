import json
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Protocol

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from inference.contracts import InferenceRequest, InferenceResult
from inference.backends.pytorch_qwenvl import QwenVlBackend
from training.datasets.contracts import AnswerType, EvalSample
from training.datasets.images import (
    MissingSampleImageError,
    resolve_sample_image as resolve_dataset_sample_image,
)
from training.datasets.loaders import DatasetLoader, system_prompt


class InferenceCompatibleModel(Protocol):
    def predict(self, request: InferenceRequest) -> InferenceResult:
        ...


class BenchReporter(Protocol):
    def start(self) -> None:
        ...

    def advance(self, record: "BenchRecord") -> None:
        ...

    def stop(self) -> None:
        ...


@dataclass(frozen=True)
class Score:
    correct: bool
    expected: str
    actual: str


@dataclass(frozen=True)
class BenchRecord:
    sample_id: str
    dataset: str
    answer_type: AnswerType
    gold: str
    prediction: str
    correct: bool
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchSummary:
    total: int
    correct: int
    accuracy: float
    average_latency_ms: float | None = None


def run_bench(
    model: InferenceCompatibleModel,
    *,
    dataset: str,
    split: str = "test",
    samples: int | None,
    batch_size: int,
    output_path: Path | None = None,
    reporter: BenchReporter | None = None,
) -> BenchSummary:
    loader = DatasetLoader(dataset, split=split, streaming=True)
    sample_stream = loader.batch(offset=0, limit=None)
    if samples is not None:
        sample_stream = islice(sample_stream, samples)
    return run_samples(
        model,
        sample_stream,
        batch_size=batch_size,
        output_path=output_path,
        reporter=reporter,
    )


def run_samples(
    model: InferenceCompatibleModel,
    samples: Iterable[EvalSample],
    *,
    batch_size: int,
    output_path: Path | None = None,
    reporter: BenchReporter | None = None,
) -> BenchSummary:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    records: list[BenchRecord] = []

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("")

    if reporter is not None:
        reporter.start()

    try:
        for sample_batch in _chunks(samples, batch_size):
            for sample in sample_batch:
                request = request_from_sample(sample)
                result = model.predict(request)
                score = score_prediction(
                    answer_type=sample.answer_type,
                    gold=sample.answer,
                    prediction=result.prediction,
                )
                record = BenchRecord(
                    sample_id=sample.id,
                    dataset=sample.dataset,
                    answer_type=sample.answer_type,
                    gold=sample.answer,
                    prediction=result.prediction,
                    correct=score.correct,
                    latency_ms=_latency_ms(result.metadata),
                    metadata={
                        "chart_type": sample.chart_type,
                        "task_type": sample.task_type,
                        "result": result.metadata,
                    },
                )
                records.append(record)
                if output_path is not None:
                    _append_jsonl(output_path, record)
                if reporter is not None:
                    reporter.advance(record)
    finally:
        if reporter is not None:
            reporter.stop()

    return summarize(records)


def request_from_sample(sample: EvalSample) -> InferenceRequest:
    return InferenceRequest(
        sample_id=sample.id,
        image=resolve_sample_image(sample),
        system_prompt=system_prompt(sample),
        prompt=sample.question,
        metadata={
            "dataset": sample.dataset,
            "answer_type": sample.answer_type,
            "chart_type": sample.chart_type,
            "task_type": sample.task_type,
        },
    )


def resolve_sample_image(
    sample: EvalSample,
    dataset_root: Path = Path("artifacts/datasets"),
) -> Any:
    return resolve_dataset_sample_image(sample, dataset_root=dataset_root)


def score_prediction(
    *,
    answer_type: AnswerType,
    gold: str,
    prediction: str,
) -> Score:
    expected = _normalize_by_type(answer_type, gold)
    actual = _normalize_by_type(answer_type, prediction)
    return Score(correct=expected == actual, expected=expected, actual=actual)


def summarize(records: list[BenchRecord]) -> BenchSummary:
    total = len(records)
    correct = sum(record.correct for record in records)
    latencies = [
        record.latency_ms for record in records if record.latency_ms is not None
    ]
    return BenchSummary(
        total=total,
        correct=correct,
        accuracy=correct / total if total else 0.0,
        average_latency_ms=sum(latencies) / len(latencies) if latencies else None,
    )


def _normalize_by_type(answer_type: AnswerType, value: str) -> str:
    if answer_type == "numeric":
        return _normalize_numeric(value)
    if answer_type == "yes_no":
        return _normalize_yes_no(value)
    return _normalize_text(value)


def _normalize_numeric(value: str) -> str:
    text = str(value).strip().replace(",", "").rstrip("%")
    try:
        return str(float(text))
    except ValueError:
        return _normalize_text(value)


def _normalize_yes_no(value: str) -> str:
    text = _normalize_text(value)
    if text == "true":
        return "yes"
    if text == "false":
        return "no"
    return text


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().strip(" \t\r\n.!?").split())


def _latency_ms(metadata: dict[str, Any]) -> float | None:
    value = metadata.get("latency_ms")
    if isinstance(value, int | float):
        return float(value)
    return None


def _append_jsonl(path: Path, record: BenchRecord) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def _chunks(samples: Iterable[EvalSample], batch_size: int):
    batch: list[EvalSample] = []
    for sample in samples:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class RichBenchReporter:
    def __init__(
        self,
        *,
        console: Console,
        dataset: str,
        split: str,
        sample_limit: int | None,
        batch_size: int,
    ):
        self.console = console
        self.dataset = dataset
        self.split = split
        self.sample_limit = sample_limit
        self.batch_size = batch_size
        self.correct = 0
        self.total = 0
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]bench[/bold]"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        self.task_id = None

    def start(self) -> None:
        sample_text = "all samples" if self.sample_limit is None else str(self.sample_limit)
        self.console.print(
            f"[bold]dataset[/bold]={self.dataset} [bold]split[/bold]={self.split} "
            f"[bold]samples[/bold]={sample_text} [bold]batch_size[/bold]={self.batch_size}"
        )
        total = self.sample_limit if self.sample_limit is not None else None
        self.progress.start()
        self.task_id = self.progress.add_task("bench", total=total)

    def advance(self, record: BenchRecord) -> None:
        self.total += 1
        self.correct += int(record.correct)
        accuracy = self.correct / self.total if self.total else 0.0
        assert self.task_id is not None
        self.progress.update(
            self.task_id,
            advance=1,
            description=f"accuracy {accuracy:.2%}",
        )

    def stop(self) -> None:
        self.progress.stop()


def render_summary(
    *,
    console: Console,
    summary: BenchSummary,
    output_path: Path | None,
) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("total", str(summary.total))
    table.add_row("correct", str(summary.correct))
    table.add_row("accuracy", f"{summary.accuracy:.2%}")
    if summary.average_latency_ms is not None:
        table.add_row("avg latency", f"{summary.average_latency_ms:.1f} ms")
    if output_path is not None:
        table.add_row("records", str(output_path))
    console.print(Panel(table, title="bench summary", expand=False))


@click.command()
@click.option("--dataset", default="chartqa", show_default=True)
@click.option("--split", default="test", show_default=True)
@click.option("--batch-size", default=32, show_default=True, type=int)
@click.option("--samples", type=int)
@click.option("--all-samples", is_flag=True)
@click.option(
    "--config",
    default=Path("config.toml"),
    show_default=True,
    type=click.Path(path_type=Path, dir_okay=False),
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path, dir_okay=False),
)
def main(
    dataset: str,
    split: str,
    batch_size: int,
    samples: int | None,
    all_samples: bool,
    config: Path,
    output_path: Path | None,
) -> None:
    """Run a small Kestrel inference bench."""
    sample_limit = _resolve_sample_limit(samples=samples, all_samples=all_samples)
    console = Console()
    reporter = RichBenchReporter(
        console=console,
        dataset=dataset,
        split=split,
        sample_limit=sample_limit,
        batch_size=batch_size,
    )
    model = QwenVlBackend.from_config(config)
    summary = run_bench(
        model,
        dataset=dataset,
        split=split,
        samples=sample_limit,
        batch_size=batch_size,
        output_path=output_path,
        reporter=reporter,
    )
    render_summary(console=console, summary=summary, output_path=output_path)


def _resolve_sample_limit(samples: int | None, all_samples: bool) -> int | None:
    if samples is None and not all_samples:
        raise click.UsageError("Pass exactly one of --samples or --all-samples.")
    if samples is not None and all_samples:
        raise click.UsageError("Pass exactly one of --samples or --all-samples.")
    if samples is not None and samples <= 0:
        raise click.UsageError("--samples must be greater than 0.")
    return samples


if __name__ == "__main__":
    main()
