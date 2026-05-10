import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from training.datasets.contracts import AnswerType, EvalSample, SupervisionType

DEFAULT_EVAL_ROOT = Path("artifacts/evals")


@dataclass(frozen=True)
class FailureRecord:
    sample_id: str
    dataset: str
    question: str
    gold: str
    prediction: str
    answer_type: AnswerType
    supervision: SupervisionType
    image_ref: str
    chart_type: str | None = None
    task_type: str | None = None
    error_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FailureSliceLoader:
    def __init__(self, failure_dir: Path, eval_root: Path = DEFAULT_EVAL_ROOT):
        self.failure_dir = failure_dir
        self.eval_root = eval_root

    def records(self):
        with (self.failure_dir / "failures.jsonl").open() as handle:
            for line in handle:
                yield FailureRecord(**json.loads(line))

    def samples(self):
        for record in self.records():
            yield EvalSample(
                id=record.sample_id,
                dataset=record.dataset,
                image=self._load_image(record.image_ref),
                question=record.question,
                answer=record.gold,
                answer_type=record.answer_type,
                supervision=record.supervision,
                chart_type=record.chart_type,
                task_type=record.task_type,
                tag=record.metadata.get("tag", "synthetic"),
                metadata={
                    **record.metadata,
                    "prediction": record.prediction,
                    "error_type": record.error_type,
                    "image_ref": record.image_ref,
                },
            )

    def _load_image(self, image_ref: str):
        image_path = self.eval_root / image_ref
        if image_path.exists():
            return Image.open(image_path)
        return image_ref


def failure_from_sample(
    sample: EvalSample,
    prediction: str,
    error_type: str | None = None,
    eval_root: Path = DEFAULT_EVAL_ROOT,
) -> FailureRecord:
    image_ref = _ensure_image_ref(sample, eval_root)
    return FailureRecord(
        sample_id=sample.id,
        dataset=sample.dataset,
        question=sample.question,
        gold=sample.answer,
        prediction=prediction,
        answer_type=sample.answer_type,
        supervision=sample.supervision,
        image_ref=image_ref,
        chart_type=sample.chart_type,
        task_type=sample.task_type,
        error_type=error_type,
        metadata={**sample.metadata, "tag": sample.tag},
    )


def egress_failure_batch(
    name: str,
    failures: list[FailureRecord],
    eval_root: Path = DEFAULT_EVAL_ROOT,
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    batch_name = f"{name}_{timestamp}"
    failure_dir = eval_root / "failures" / batch_name
    failure_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "schema_version": 1,
        "name": name,
        "batch_name": batch_name,
        "created_at": timestamp,
        "count": len(failures),
    }
    _write_json(failure_dir / "manifest.json", manifest)
    _write_jsonl(failure_dir / "failures.jsonl", failures)
    return failure_dir


def _ensure_image_ref(sample: EvalSample, eval_root: Path) -> str:
    if isinstance(sample.image, Image.Image):
        image_ref = f"images/{sample.dataset}/{_safe_name(sample.id)}.png"
        image_path = eval_root / image_ref
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if not image_path.exists():
            sample.image.save(image_path)
        return image_ref

    return (
        sample.metadata.get("image_ref")
        or sample.metadata.get("image_archive")
        or str(sample.image)
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, records: list[FailureRecord]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def _safe_name(value: str) -> str:
    import re

    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return name.strip("._-") or "batch"
