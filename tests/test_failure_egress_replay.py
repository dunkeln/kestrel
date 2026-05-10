import random

from PIL import Image

from training.datasets.contracts import EvalSample
from training.evals.failures import (
    FailureSliceLoader,
    egress_failure_batch,
    failure_from_sample,
)


def test_data_batch_failure_egress_and_replay(tmp_path):
    eval_root = tmp_path / "artifacts" / "evals"
    batch = [
        EvalSample(
            id=f"chartqa:{index}",
            dataset="chartqa",
            image=Image.new("RGB", (8, 8), "white"),
            question=f"What is value {index}?",
            answer=str(index),
            answer_type="numeric",
            supervision="benchmark",
            task_type="value_extraction",
            metadata={"source_type": "synthetic"},
        )
        for index in range(8)
    ]

    rng = random.Random(7)
    failure_samples = rng.sample(batch, 3)
    failures = [
        failure_from_sample(
            sample,
            prediction=f"bad-{sample.answer}",
            error_type="wrong_numeric",
            eval_root=eval_root,
        )
        for sample in failure_samples
    ]

    failure_dir = egress_failure_batch(
        "unit_failure_slice",
        failures,
        eval_root=eval_root,
    )
    replayed = list(FailureSliceLoader(failure_dir, eval_root=eval_root).samples())

    assert failure_dir.name.startswith("unit_failure_slice_")
    assert (failure_dir / "manifest.json").exists()
    assert (failure_dir / "failures.jsonl").exists()
    assert len(replayed) == len(failures)

    for replayed_sample, failure in zip(replayed, failures, strict=True):
        assert isinstance(replayed_sample, EvalSample)
        assert isinstance(replayed_sample.image, Image.Image)
        assert replayed_sample.id == failure.sample_id
        assert replayed_sample.dataset == failure.dataset
        assert replayed_sample.question == failure.question
        assert replayed_sample.answer == failure.gold
        assert replayed_sample.answer_type == failure.answer_type
        assert replayed_sample.supervision == failure.supervision
        assert replayed_sample.task_type == failure.task_type
        assert replayed_sample.metadata["prediction"] == failure.prediction
        assert replayed_sample.metadata["error_type"] == failure.error_type
        assert replayed_sample.metadata["image_ref"].startswith("images/chartqa/")
