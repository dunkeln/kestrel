import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from training.datasets.contracts import EvalSample
from training.distillation.contrastive import build_contrastive_record, should_emit_contrastive
from training.distillation.sample_stream import StreamConfig, iter_sample_batches


def test_stream_batches_honor_samples_batch_size_and_shuffle_seed():
    config = StreamConfig(
        dataset="chartqa",
        split="train",
        samples=5,
        batch_size=2,
        shuffle_seed=7,
        shuffle_buffer_size=3,
    )

    batches_a = list(iter_sample_batches(config, loader_factory=_FakeLoader))
    batches_b = list(iter_sample_batches(config, loader_factory=_FakeLoader))
    ids_a = [[sample.id for sample in batch] for batch in batches_a]
    ids_b = [[sample.id for sample in batch] for batch in batches_b]

    assert [len(batch) for batch in batches_a] == [2, 2, 1]
    assert sum(len(batch) for batch in batches_a) == 5
    assert ids_a == ids_b
    assert ids_a != [["sample-0", "sample-1"], ["sample-2", "sample-3"], ["sample-4"]]


def test_stream_all_samples_mode_has_no_limit():
    config = StreamConfig(
        dataset="chartqa",
        split="train",
        all_samples=True,
        batch_size=4,
        shuffle_seed=None,
    )

    batches = list(iter_sample_batches(config, loader_factory=_FakeLoader))

    assert [len(batch) for batch in batches] == [4, 4, 2]


def test_stream_rejects_conflicting_size_modes():
    config = StreamConfig(
        dataset="chartqa",
        split="train",
        samples=1,
        all_samples=True,
        batch_size=1,
    )

    try:
        list(iter_sample_batches(config, loader_factory=_FakeLoader))
    except ValueError as exc:
        assert "cannot set both samples and all_samples" in str(exc)
    else:
        raise AssertionError("expected conflicting stream size modes to fail")


def test_contrastive_record_is_deterministic_and_tagged():
    record = {
        "image": "images/a.png",
        "input": "Is A higher than B?",
        "output": "<perceive>x</perceive><extract>y</extract><answer>yes</answer>",
        "metadata": {
            "dataset": "figureqa",
            "sample_id": "sample-1",
            "answer": "yes",
            "answer_type": "yes_no",
            "example_type": "gold",
        },
    }

    assert should_emit_contrastive(record, rate=1.0, seed=13)
    assert not should_emit_contrastive(record, rate=math.nan, seed=13)
    first = build_contrastive_record(record, seed=13)
    second = build_contrastive_record(record, seed=13)

    assert first == second
    assert first is not None
    assert first["output"].endswith("<answer>no</answer>")
    assert first["metadata"]["example_type"] == "contrastive"
    assert first["metadata"]["gold_answer"] == "yes"
    assert first["metadata"]["perturbed_answer"] == "no"
    assert first["metadata"]["perturbation_type"] == "yes_no_flip"


class _FakeLoader:
    def __init__(self, *_args, **_kwargs):
        pass

    def samples(self):
        yield from [_sample(f"sample-{index}", "yes") for index in range(10)]


def _sample(sample_id: str, answer: str) -> EvalSample:
    return EvalSample(
        id=sample_id,
        dataset="chartqa",
        image=Image.new("RGB", (8, 8), color="white"),
        question="Is A higher than B?",
        answer=answer,
        answer_type="yes_no",
        supervision="benchmark",
        task_type="yes_no",
        tag="real",
    )
