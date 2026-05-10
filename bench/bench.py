import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

import click

from inference.contracts import InferenceRequest, InferenceResult
from inference.local_mps.qwenvl import build_backend
from training.datasets.contracts import AnswerType, EvalSample
from training.datasets.loaders import DatasetLoader, system_prompt


class InferenceCompatibleModel(Protocol):
    def predict(self, request: InferenceRequest) -> InferenceResult:
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
    samples: int,
    batch_size: int,
    output_path: Path | None = None,
) -> BenchSummary:
    loader = DatasetLoader(dataset, split=split, streaming=True)
    return run_samples(
        model,
        loader.batch(offset=0, limit=samples),
        batch_size=batch_size,
        output_path=output_path,
    )


def run_samples(
    model: InferenceCompatibleModel,
    samples: Iterable[EvalSample],
    *,
    batch_size: int,
    output_path: Path | None = None,
) -> BenchSummary:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    records: list[BenchRecord] = []

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("")

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

    return summarize(records)


def request_from_sample(sample: EvalSample) -> InferenceRequest:
    return InferenceRequest(
        sample_id=sample.id,
        image=sample.image,
        system_prompt=system_prompt(sample),
        prompt=sample.question,
        metadata={
            "dataset": sample.dataset,
            "answer_type": sample.answer_type,
            "chart_type": sample.chart_type,
            "task_type": sample.task_type,
        },
    )


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
    return " ".join(str(value).strip().lower().split())


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


@click.command()
@click.option("--dataset", default="chartqa", show_default=True)
@click.option("--split", default="test", show_default=True)
@click.option("--batch-size", default=32, show_default=True, type=int)
@click.option("--samples", required=True, type=int)
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
    samples: int,
    config: Path,
    output_path: Path | None,
) -> None:
    """Run a small Kestrel inference bench."""
    model = build_backend(config)
    summary = run_bench(
        model,
        dataset=dataset,
        split=split,
        samples=samples,
        batch_size=batch_size,
        output_path=output_path,
    )
    click.echo(json.dumps(asdict(summary), sort_keys=True))


if __name__ == "__main__":
    main()
