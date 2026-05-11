import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.datasets.contracts import EvalSample
from training.sft.reasoning_pipeline import PipelineConfig
from training.sft.reasoning_synthesis import build_synthesis_prompt, lint_synthesis
from training.sft.teacher_ensemble import (
    CONSTITUTIONAL_CONSTRAINT,
    GROK_MODEL,
    OPENAI_MODEL,
    STAGE1_REASONING_PROMPT,
    build_stage_prompt,
    contest,
    dissenting_providers,
    retry_prompt,
    validate_reasoning,
    validation_error,
)


def test_stage1_teacher_prompt_uses_constitutional_template():
    sample = {
        "imgname": "chart.png",
        "question": "What is the value for A?",
        "answer": "42",
    }

    assert build_stage_prompt(sample, stage=1) == STAGE1_REASONING_PROMPT.format(
        constitutional_constraint=CONSTITUTIONAL_CONSTRAINT,
        question="What is the value for A?",
        answer="42",
    )


def test_teacher_models_use_current_lightweight_vlm_choices():
    assert OPENAI_MODEL == "gpt-4o-mini"
    assert GROK_MODEL == "grok-4.3"


def test_eval_sample_prompt_reuses_dataset_task_guidance_without_replacing_schema():
    sample = EvalSample(
        id="chartqa:1",
        dataset="chartqa",
        image="chart.png",
        question="What is the value for A?",
        answer="42",
        answer_type="numeric",
        supervision="benchmark",
        task_type="value_extraction",
    )

    prompt = build_stage_prompt(sample, stage=1)

    assert "<perceive>" in prompt
    assert "<extract>" in prompt
    assert "<answer>42</answer>" in prompt
    assert "Dataset task guidance:" in prompt
    assert "Read the chart value requested. Return only the value." in prompt
    assert "the XML structure above is still required" in prompt


def test_validation_and_contest_use_required_stage1_tags():
    yes_a = "<perceive>a</perceive><extract>b</extract><answer>yes</answer>"
    yes_b = "<perceive>a</perceive><extract>b</extract><answer> yes </answer>"
    no = "<perceive>a</perceive><extract>b</extract><answer>no</answer>"

    assert validate_reasoning(yes_a, stage=1)
    assert validation_error("<perceive>a</perceive>", stage=1) == (
        "missing or empty XML tags: extract, answer"
    )
    assert contest({"claude": yes_a, "openai": yes_b, "grok": no}, stage=1) == "yes"
    assert dissenting_providers(
        {"claude": yes_a, "openai": yes_b, "grok": no},
        agreed_output="yes",
        stage=1,
    ) == ["grok"]


def test_retry_prompt_carries_error_and_previous_response():
    repaired = retry_prompt(
        "base prompt",
        error="missing answer tag",
        previous_response="<perceive>x</perceive>",
        agreed_output="42",
    )

    assert "Error: missing answer tag" in repaired
    assert "Required agreed output: 42" in repaired
    assert "Previous response:" in repaired
    assert "<perceive>x</perceive>" in repaired


def test_synthesis_prompt_uses_agreed_answer_and_colleagues():
    prompt = build_synthesis_prompt(
        claude_reasoning="claude trace",
        gpt_reasoning="gpt trace",
        grok_reasoning="grok trace",
        agreed_output="42",
        stage=1,
    )

    assert "Colleague GPT" in prompt
    assert "gpt trace" in prompt
    assert "grok trace" in prompt
    assert "<answer>42</answer>" in prompt


def test_synthesis_lint_flags_unnecessary_extra_facts():
    text = (
        "<perceive>Area chart with yearly x-axis.</perceive>"
        "<extract>In 2013 the value is 0.5, in 2014 it is 0.6, and in 2016 it is 0.67.</extract>"
        "<answer>true</answer>"
    )

    errors = lint_synthesis(
        text,
        stage=1,
        sample={
            "question": "Was there a significant increase in 2014?",
            "answer": "true",
        },
    )

    assert errors
    assert "unnecessary numeric/date evidence" in errors[0]


def test_pipeline_config_controls_batching_and_retries():
    config = PipelineConfig(
        input_batch_size=2,
        max_concurrent_samples=1,
        max_api_attempts=3,
        max_teacher_repair_attempts=2,
        max_synthesis_validation_attempts=4,
    )

    assert config.input_batch_size == 2
    assert config.max_teacher_repair_attempts == 2
    assert config.max_synthesis_validation_attempts == 4
