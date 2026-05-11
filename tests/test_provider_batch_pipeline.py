import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from training.sft.provider_batch_contracts import ProviderBatchResult
from training.sft.provider_batch_pipeline import process_batch_provider_batched
from training.sft.reasoning_pipeline import PipelineConfig


def test_provider_batch_pipeline_uses_batched_teacher_and_synthesis_calls(tmp_path):
    image_path = tmp_path / "chart.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    calls = []

    async def anthropic_batch(requests, *, poll_interval_seconds, label):
        calls.append(("anthropic", label, len(requests)))
        return {
            request.custom_id: ProviderBatchResult(
                request.custom_id,
                _response_for(label, request.prompt),
                None,
            )
            for request in requests
        }

    async def openai_batch(requests, *, poll_interval_seconds, label):
        calls.append(("openai", label, len(requests)))
        return {
            request.custom_id: ProviderBatchResult(
                request.custom_id,
                _teacher_response(request.prompt),
                None,
            )
            for request in requests
        }

    results = _run(
        process_batch_provider_batched(
            [
                {
                    "imgname": str(image_path),
                    "question": "Is A higher than B?",
                    "answer": "yes",
                    "answer_type": "yes_no",
                    "task_type": "yes_no",
                    "metadata": {},
                }
            ],
            config=PipelineConfig(),
            poll_interval_seconds=1,
            anthropic_batch_fn=anthropic_batch,
            openai_batch_fn=openai_batch,
        )
    )

    assert len(results) == 1
    assert results[0].written
    assert results[0].metadata["provider_batch"] is True
    assert results[0].agreed_output == "yes"
    assert calls == [
        ("anthropic", "teacher_claude", 1),
        ("openai", "teacher_openai", 1),
        ("anthropic", "synthesis_attempt_1", 1),
    ]


def test_provider_batch_pipeline_adjudicates_plotqa_numeric_disagreement(tmp_path):
    image_path = tmp_path / "chart.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    calls = []

    async def anthropic_batch(requests, *, poll_interval_seconds, label):
        calls.append(("anthropic", label, len(requests)))
        return {
            request.custom_id: ProviderBatchResult(
                request.custom_id,
                _plotqa_response(label, request.prompt, provider="claude"),
                None,
            )
            for request in requests
        }

    async def openai_batch(requests, *, poll_interval_seconds, label):
        calls.append(("openai", label, len(requests)))
        return {
            request.custom_id: ProviderBatchResult(
                request.custom_id,
                _plotqa_response(label, request.prompt, provider="openai"),
                None,
            )
            for request in requests
        }

    results = _run(
        process_batch_provider_batched(
            [
                {
                    "imgname": str(image_path),
                    "question": "What is the difference between A and B?",
                    "answer": "7.3",
                    "answer_type": "numeric",
                    "task_type": "arithmetic",
                    "dataset": "plotqa_qa",
                    "metadata": {"plotqa_answer_mode": "computed_oov"},
                }
            ],
            config=PipelineConfig(),
            poll_interval_seconds=1,
            anthropic_batch_fn=anthropic_batch,
            openai_batch_fn=openai_batch,
        )
    )

    assert len(results) == 1
    assert results[0].written
    assert results[0].agreed_output == "7.3"
    assert results[0].metadata["plotqa_numeric_adjudicated"] is True
    assert results[0].metadata["adjudicator_model"] == "claude-haiku-4-5"
    assert "<compute>" in results[0].sft_record["output"]
    assert calls == [
        ("anthropic", "teacher_claude", 1),
        ("openai", "teacher_openai", 1),
        ("anthropic", "plotqa_numeric_adjudication_attempt_1", 1),
    ]


def _response_for(label: str, prompt: str) -> str:
    if label.startswith("synthesis"):
        answer = _answer(prompt)
        return (
            "<perceive>bar chart</perceive>"
            "<extract>A is visibly higher than B</extract>"
            f"<answer>{answer}</answer>"
        )
    return _teacher_response(prompt)


def _teacher_response(prompt: str) -> str:
    answer = _answer(prompt)
    return (
        "<perceive>bar chart</perceive>"
        "<extract>A is visibly higher than B</extract>"
        f"<answer>{answer}</answer>"
    )


def _plotqa_response(label: str, prompt: str, *, provider: str) -> str:
    if label.startswith("plotqa_numeric_adjudication"):
        return (
            "<perceive>line chart</perceive>"
            "<extract>A is 10.0 and B is 2.7</extract>"
            "<compute>10.0 - 2.7 = 7.3</compute>"
            "<answer>7.3</answer>"
        )
    answer = "7.30" if provider == "claude" else "7.3"
    return (
        "<perceive>line chart</perceive>"
        "<extract>A is 10.0 and B is 2.7</extract>"
        "<compute>10.0 - 2.7 = 7.3</compute>"
        f"<answer>{answer}</answer>"
    )


def _answer(prompt: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>|Known correct answer: (.+)", prompt)
    if not match:
        return "yes"
    return (match.group(1) or match.group(2)).strip()


def _run(coro):
    import asyncio

    return asyncio.run(coro)
