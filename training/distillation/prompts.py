from __future__ import annotations

from training.datasets.contracts import EvalSample
from training.datasets.loaders import system_prompt
from training.distillation.validation import stage1_requires_compute


ROLE_ASSIGNMENTS = {
    "claude": "constitutional teacher — structure, consistency, conservatism",
    "openai": "numerical analyst — precise value extraction, axis reading",
}

CONSTITUTIONAL_CONSTRAINT = """
STRICT RULES:
- Only reference what is physically visible in the chart
- Never use external knowledge about the data topic
- If a value is ambiguous, state the uncertainty explicitly
- Complete every XML tag — partial responses are invalid
- Never reverse-engineer reasoning from the answer
"""

STAGE1_REASONING_PROMPT = """You are analyzing a chart to answer a question.

{constitutional_constraint}

Question: {question}
Known correct answer: {answer}

Generate grounded reasoning following this exact structure:

<perceive>Chart type, axis labels, units, legend items visible in the image</perceive>
<extract>Task-relevant chart structure: only the visible labels, series, x/category values, y/numeric values, or visual relation needed to answer</extract>
<answer>{answer}</answer>

Complete all tags. Be concise but specific. Do not serialize the full chart unless the question explicitly asks for structure."""

STAGE1_COMPUTE_REASONING_PROMPT = """You are analyzing a chart to answer a question.

{constitutional_constraint}

Question: {question}
Known correct answer: {answer}

Generate grounded reasoning following this exact structure:

<perceive>Chart type, axis labels, units, legend items visible in the image</perceive>
<extract>Task-relevant chart structure: only the visible labels, series, x/category values, and y/numeric values needed for the calculation</extract>
<compute>Arithmetic or comparison performed only from the extracted chart values</compute>
<answer>{answer}</answer>

Complete all tags. Be concise but specific. Do not serialize the full chart unless the question explicitly asks for structure."""

STAGE2_REASONING_PROMPT = """You are a chart faithfulness judge.

{constitutional_constraint}

Claim: {claim}
Known correct verdict: {verdict}

Evaluate the claim using this exact structure:

<perceive>Chart type, axis labels, units, legend items visible in the image</perceive>
<extract>Task-relevant chart structure for the claim: only visible values, labels, trends, or visual relations needed to judge it</extract>
<compare>How extracted values support or contradict each component of the claim</compare>
<verdict>{verdict}</verdict>
<confidence>0.0-1.0</confidence>
<rubric>
  numerical: PASS/FAIL
  trend: PASS/FAIL
  label: PASS/FAIL
  scope: PASS/FAIL
</rubric>

Complete all tags. Never skip a claim component in compare."""

PLOTQA_NUMERIC_ADJUDICATION_PROMPT = """You are the lead adjudicator for a PlotQA numeric chart question.

Two teachers produced valid chart reasoning traces, but their final numeric answers did not match.

Claude Haiku:
{claude_reasoning}

GPT:
{gpt_reasoning}

Known correct answer: {gold_answer}

Question: {question}

Resolve the disagreement conservatively using only values visible in the chart and facts present in at least one teacher trace.

Constitutional rules you must follow:
{constitutional_constraint}

Output exactly:
<perceive>...</perceive>
<extract>task-relevant chart structure only</extract>
<compute>arithmetic or comparison performed only from extracted values</compute>
<answer>{gold_answer}</answer>"""

PRIVILEGED_SYNTHESIS_PROMPT = """You are the lead synthesizer in a panel of two expert chart analysts.

Your two colleagues have analyzed this chart:

Colleague Claude Haiku (strong at constitutional structure and conservative faithfulness):
{claude_reasoning}

Colleague GPT (strong at precise numerical extraction):
{gpt_reasoning}

Your role as lead synthesizer:
1. Extract the most conservative structural reading from Claude Haiku's analysis
2. Extract the most precise numerical readings from GPT's analysis
3. Synthesize both into a single constitutionally grounded trace
4. Resolve any conflicts conservatively — if colleagues disagree on a value, state the range or flag uncertainty
5. Never add anything not present in at least one colleague's reasoning

Constitutional rules you must follow:
{constitutional_constraint}

Output exactly:
<perceive>...</perceive>
<extract>task-relevant chart structure only</extract>
<compare>...</compare>
<verdict>{agreed_verdict}</verdict>
<confidence>...</confidence>
<rubric>
  numerical: PASS/FAIL
  trend: PASS/FAIL
  label: PASS/FAIL
  scope: PASS/FAIL
</rubric>"""

STAGE1_SYNTHESIS_PROMPT = """You are the lead synthesizer in a panel of two expert chart analysts.

Your two colleagues have analyzed this chart:

Colleague Claude Haiku (strong at constitutional structure and conservative faithfulness):
{claude_reasoning}

Colleague GPT (strong at precise numerical extraction):
{gpt_reasoning}

Your role as lead synthesizer:
1. Extract the most conservative perceptual description from Claude Haiku's analysis
2. Extract the most precise numerical readings from GPT's analysis
3. Synthesize both into a single constitutionally grounded trace
4. Resolve any conflicts conservatively
5. Never add anything not present in at least one colleague's reasoning

Constitutional rules you must follow:
{constitutional_constraint}

Output exactly:
<perceive>...</perceive>
<extract>task-relevant chart structure only</extract>
<answer>{agreed_answer}</answer>"""

STAGE1_COMPUTE_SYNTHESIS_PROMPT = """You are the lead synthesizer in a panel of two expert chart analysts.

Your two colleagues have analyzed this chart:

Colleague Claude Haiku (strong at constitutional structure and conservative faithfulness):
{claude_reasoning}

Colleague GPT (strong at precise numerical extraction):
{gpt_reasoning}

Your role as lead synthesizer:
1. Extract the most conservative perceptual description from Claude Haiku's analysis
2. Extract the most precise numerical readings from GPT's analysis
3. Synthesize both into a single constitutionally grounded trace
4. Resolve any conflicts conservatively
5. Never add anything not present in at least one colleague's reasoning

Constitutional rules you must follow:
{constitutional_constraint}

Output exactly:
<perceive>...</perceive>
<extract>task-relevant chart structure only</extract>
<compute>arithmetic or comparison performed only from extracted values</compute>
<answer>{agreed_answer}</answer>"""


def build_stage_prompt(sample: dict | EvalSample, *, stage: int) -> str:
    task_prompt = _task_prompt(sample)
    if stage == 1:
        template = (
            STAGE1_COMPUTE_REASONING_PROMPT
            if stage1_requires_compute(sample)
            else STAGE1_REASONING_PROMPT
        )
        prompt = template.format(
            constitutional_constraint=CONSTITUTIONAL_CONSTRAINT,
            question=_sample_value(sample, "question"),
            answer=_sample_value(sample, "answer"),
        )
        return _with_task_prompt(prompt, task_prompt)

    prompt = STAGE2_REASONING_PROMPT.format(
        constitutional_constraint=CONSTITUTIONAL_CONSTRAINT,
        claim=_sample_value(sample, "claim"),
        verdict=_sample_value(sample, "verdict"),
    )
    return _with_task_prompt(prompt, task_prompt)


def build_synthesis_prompt(
    *,
    claude_reasoning: str,
    gpt_reasoning: str,
    agreed_output: str,
    stage: int,
    requires_compute: bool = False,
) -> str:
    if stage == 1:
        template = (
            STAGE1_COMPUTE_SYNTHESIS_PROMPT
            if requires_compute
            else STAGE1_SYNTHESIS_PROMPT
        )
        return template.format(
            claude_reasoning=claude_reasoning,
            gpt_reasoning=gpt_reasoning,
            constitutional_constraint=CONSTITUTIONAL_CONSTRAINT,
            agreed_answer=agreed_output,
        )
    return PRIVILEGED_SYNTHESIS_PROMPT.format(
        claude_reasoning=claude_reasoning,
        gpt_reasoning=gpt_reasoning,
        constitutional_constraint=CONSTITUTIONAL_CONSTRAINT,
        agreed_verdict=agreed_output,
    )


def _sample_value(sample: dict | EvalSample, key: str) -> str:
    if isinstance(sample, EvalSample):
        if key == "question":
            return sample.question
        if key == "answer":
            return sample.answer
        raise KeyError(f"EvalSample does not provide `{key}` for stage 2.")
    return str(sample[key])


def _task_prompt(sample: dict | EvalSample) -> str | None:
    if isinstance(sample, EvalSample):
        return system_prompt(sample)
    value = sample.get("task_prompt") or sample.get("system_prompt")
    if value is None:
        return None
    return str(value)


def _with_task_prompt(prompt: str, task_prompt: str | None) -> str:
    if not task_prompt:
        return prompt
    return "\n".join(
        [
            prompt,
            "",
            "Dataset task guidance:",
            task_prompt,
            "Use this guidance only for answer focus; the XML structure above is still required.",
        ]
    )
