import json
import sys
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from click.testing import CliRunner
from PIL import Image
from rich.console import Console

from training.datasets.contracts import EvalSample
from training.distillation.generate import main
from training.distillation.pipeline import SampleResult, run_generation


def test_cli_requires_exactly_one_size_option():
    runner = CliRunner()

    missing = runner.invoke(main, ["--dataset", "chartqa"])
    conflicting = runner.invoke(
        main,
        ["--dataset", "chartqa", "--samples", "1", "--all"],
    )

    assert missing.exit_code == 2
    assert "Pass exactly one of --samples or --all" in missing.output
    assert conflicting.exit_code == 2
    assert "Pass exactly one of --samples or --all" in conflicting.output


def test_generation_writes_gold_contrastive_skipped_and_manifest(tmp_path):
    samples = [
        _sample("sample-0", "yes"),
        _sample("sample-1", "no"),
        _sample("sample-2", "no"),
    ]

    async def fake_process_batch(batch, *, config):
        return [
            _fake_result(_pipeline_sample(sample), skip=index == 1)
            for index, sample in enumerate(batch)
        ]

    stats = _run(
        run_generation(
            dataset="chartqa",
            split="train",
            samples=3,
            all_samples=False,
            batch_size=2,
            shuffle_seed=0,
            shuffle_buffer_size=1,
            contrastive_rate=1.0,
            output_root=tmp_path,
            console=Console(file=StringIO(), force_terminal=False),
            process_batch_fn=fake_process_batch,
            loader_factory=lambda *_, **__: _StaticLoader(samples),
        )
    )

    gold = _jsonl(tmp_path / "chartqa_train.jsonl")
    contrastive = _jsonl(tmp_path / "chartqa_contrastive_train.jsonl")
    skipped = _jsonl(tmp_path / "chartqa_skipped_train.jsonl")
    manifest = json.loads((tmp_path / "chartqa_train_manifest.json").read_text())

    assert stats.written == 2
    assert stats.skipped == 1
    assert stats.contrastive == 2
    assert len(gold) == 2
    assert len(contrastive) == 2
    assert len(skipped) == 1
    assert gold[0]["record_key"]
    assert gold[0]["record_key"] == gold[0]["metadata"]["record_key"]
    assert contrastive[0]["record_key"] in {record["record_key"] for record in gold}
    assert gold[0]["metadata"]["example_type"] == "gold"
    assert gold[0]["image"].startswith("images/")
    assert contrastive[0]["metadata"]["example_type"] == "contrastive"
    assert skipped[0]["skip_reason"] == "fake_skip"
    assert manifest["stats"]["written"] == 2
    assert manifest["stats"]["contrastive"] == 2


def test_generation_resume_backfills_and_skips_existing_record(tmp_path):
    sample = _sample("sample-0", "yes")
    legacy = {
        "image": "images/legacy.png",
        "input": sample.question,
        "output": "<answer>yes</answer>",
        "metadata": {
            "dataset": sample.dataset,
            "split": "train",
            "sample_id": sample.id,
            "answer": sample.answer,
            "answer_type": sample.answer_type,
            "task_type": sample.task_type,
            "image_sha256": _image_sha(sample),
        },
    }
    (tmp_path / "chartqa_train.jsonl").write_text(json.dumps(legacy) + "\n")

    async def fail_if_called(batch, *, config):
        raise AssertionError("provider should not be called for indexed samples")

    stats = _run(
        run_generation(
            dataset="chartqa",
            split="train",
            samples=1,
            all_samples=False,
            batch_size=1,
            shuffle_seed=0,
            shuffle_buffer_size=1,
            contrastive_rate=0.0,
            output_root=tmp_path,
            console=Console(file=StringIO(), force_terminal=False),
            process_batch_fn=fail_if_called,
            loader_factory=lambda *_, **__: _StaticLoader([sample]),
            resume=True,
        )
    )

    rows = _jsonl(tmp_path / "chartqa_train.jsonl")
    assert stats.deduped == 1
    assert rows[0]["record_key"]
    assert rows[0]["metadata"]["record_key"] == rows[0]["record_key"]


def test_generation_writes_manifest_on_failure(tmp_path):
    async def failing_process_batch(batch, *, config):
        raise RuntimeError("provider failed")

    try:
        _run(
            run_generation(
                dataset="chartqa",
                split="train",
                samples=1,
                all_samples=False,
                batch_size=1,
                shuffle_seed=0,
                shuffle_buffer_size=1,
                contrastive_rate=0.0,
                output_root=tmp_path,
                console=Console(file=StringIO(), force_terminal=False),
                process_batch_fn=failing_process_batch,
                loader_factory=lambda *_, **__: _StaticLoader([_sample("sample-0", "yes")]),
            )
        )
    except RuntimeError as exc:
        assert "provider failed" in str(exc)
    else:
        raise AssertionError("expected generation failure")

    manifest = json.loads((tmp_path / "chartqa_train_manifest.json").read_text())
    assert manifest["stats"]["total"] == 1
    assert manifest["stats"]["written"] == 0


class _StaticLoader:
    def __init__(self, samples):
        self._samples = samples

    def samples(self):
        yield from self._samples


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


def _fake_result(sample: dict, *, skip: bool) -> SampleResult:
    if skip:
        return SampleResult(
            sample=sample,
            sft_record=None,
            agreed_output=None,
            teacher_outputs={},
            teacher_errors={},
            skip_reason="fake_skip",
        )
    output = (
        "<perceive>bar chart</perceive>"
        "<extract>A is higher than B</extract>"
        f"<answer>{sample['answer']}</answer>"
    )
    return SampleResult(
        sample=sample,
        sft_record={
            "image": sample["imgname"],
            "input": sample["question"],
            "output": output,
            "metadata": {"reasoning_schema": "stage1"},
        },
        agreed_output=sample["answer"],
        teacher_outputs={"claude": output, "openai": output},
        teacher_errors={"claude": None, "openai": None},
    )


def _pipeline_sample(sample):
    if isinstance(sample, dict):
        return sample
    return {
        "imgname": sample.image,
        "question": sample.question,
        "answer": sample.answer,
    }


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def _image_sha(sample: EvalSample) -> str:
    import hashlib
    from io import BytesIO

    buffer = BytesIO()
    sample.image.save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()
