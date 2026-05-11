from __future__ import annotations

import logging
from dataclasses import dataclass
import re

from training.sft.teacher_ensemble import (
    CONSTITUTIONAL_CONSTRAINT,
    get_claude_reasoning,
    retry_prompt,
    validate_reasoning,
    validation_error,
)


logger = logging.getLogger(__name__)

PRIVILEGED_SYNTHESIS_PROMPT = """You are the lead synthesizer in a panel of three expert chart analysts.

Your two colleagues have analyzed this chart:

Colleague GPT (strong at precise numerical extraction):
{gpt_reasoning}

Colleague Grok (strong at trend and directional reasoning):
{grok_reasoning}

Your role as lead synthesizer:
1. Extract the most precise numerical readings from GPT's analysis
2. Extract the clearest trend reasoning from Grok's analysis
3. Synthesize both into a single constitutionally grounded trace
4. Resolve any conflicts conservatively — if colleagues disagree on a value, state the range or flag uncertainty
5. Never add anything not present in at least one colleague's reasoning

Constitutional rules you must follow:
{constitutional_constraint}

Output exactly:
<perceive>...</perceive>
<extract>...</extract>
<compare>...</compare>
<verdict>{agreed_verdict}</verdict>
<confidence>...</confidence>
<rubric>
  numerical: PASS/FAIL
  trend: PASS/FAIL
  label: PASS/FAIL
  scope: PASS/FAIL
</rubric>"""

STAGE1_SYNTHESIS_PROMPT = """You are the lead synthesizer in a panel of three expert chart analysts.

Your two colleagues have analyzed this chart:

Colleague GPT (strong at precise numerical extraction):
{gpt_reasoning}

Colleague Grok (strong at trend and directional reasoning):
{grok_reasoning}

Your role as lead synthesizer:
1. Extract the most precise numerical readings from GPT's analysis
2. Extract the clearest perceptual description from Grok's analysis
3. Synthesize both into a single constitutionally grounded trace
4. Resolve any conflicts conservatively
5. Never add anything not present in at least one colleague's reasoning

Constitutional rules you must follow:
{constitutional_constraint}

Output exactly:
<perceive>...</perceive>
<extract>...</extract>
<answer>{agreed_answer}</answer>"""


@dataclass(frozen=True)
class SynthesisResult:
    text: str | None
    error: str | None
    attempts: int
    prompt: str
    lint_errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.text is not None and self.error is None


async def synthesize(
    image_b64: str,
    claude_reasoning: str,
    gpt_reasoning: str,
    grok_reasoning: str,
    agreed_output: str,
    stage: int = 1
) -> str | None:
    """Build the stage prompt, call Claude, and return valid synthesized reasoning."""
    result = await synthesize_with_retries(
        image_b64=image_b64,
        claude_reasoning=claude_reasoning,
        gpt_reasoning=gpt_reasoning,
        grok_reasoning=grok_reasoning,
        agreed_output=agreed_output,
        stage=stage,
    )
    return result.text


async def synthesize_with_retries(
    *,
    image_b64: str,
    claude_reasoning: str,
    gpt_reasoning: str,
    grok_reasoning: str,
    agreed_output: str,
    stage: int,
    max_validation_attempts: int = 3,
    sample: dict | None = None,
) -> SynthesisResult:
    base_prompt = build_synthesis_prompt(
        claude_reasoning=claude_reasoning,
        gpt_reasoning=gpt_reasoning,
        grok_reasoning=grok_reasoning,
        agreed_output=agreed_output,
        stage=stage,
    )
    prompt = base_prompt
    last_error = "synthesis did not run"
    last_lint_errors: tuple[str, ...] = ()
    for attempt in range(1, max_validation_attempts + 1):
        text = await get_claude_reasoning(image_b64, prompt)
        structure_error = validation_error(text, stage=stage)
        last_lint_errors = lint_synthesis(text, stage=stage, sample=sample)
        last_error = structure_error or _lint_error_message(last_lint_errors) or ""
        if text is not None and not structure_error and not last_lint_errors:
            return SynthesisResult(
                text=text,
                error=None,
                attempts=attempt,
                prompt=prompt,
                lint_errors=(),
            )

        logger.info("synthesis validation failed on attempt %s: %s", attempt, last_error)
        prompt = retry_prompt(
            base_prompt,
            error=last_error,
            previous_response=text,
            agreed_output=agreed_output,
        )

    return SynthesisResult(
        text=None,
        error=last_error or "invalid synthesis response",
        attempts=max_validation_attempts,
        prompt=prompt,
        lint_errors=last_lint_errors,
    )


def build_synthesis_prompt(
    *,
    claude_reasoning: str,
    gpt_reasoning: str,
    grok_reasoning: str,
    agreed_output: str,
    stage: int,
) -> str:
    _ = claude_reasoning
    if stage == 1:
        return STAGE1_SYNTHESIS_PROMPT.format(
            gpt_reasoning=gpt_reasoning,
            grok_reasoning=grok_reasoning,
            constitutional_constraint=CONSTITUTIONAL_CONSTRAINT,
            agreed_answer=agreed_output,
        )
    return PRIVILEGED_SYNTHESIS_PROMPT.format(
        gpt_reasoning=gpt_reasoning,
        grok_reasoning=grok_reasoning,
        constitutional_constraint=CONSTITUTIONAL_CONSTRAINT,
        agreed_verdict=agreed_output,
    )


def lint_synthesis(
    text: str | None,
    *,
    stage: int,
    sample: dict | None = None,
) -> tuple[str, ...]:
    if text is None:
        return ()
    errors: list[str] = []
    extract = _tag_value(text, "extract")
    if extract is None:
        return ()

    if stage == 1 and sample is not None:
        question = str(sample.get("question", ""))
        answer = str(sample.get("answer", ""))
        extra_numbers = _extra_numbers(extract, allowed_text=f"{question} {answer}")
        if len(extra_numbers) > _allowed_extra_number_count(question):
            errors.append(
                "extract includes unnecessary numeric/date evidence: "
                + ", ".join(extra_numbers)
                + ". Keep only facts needed to answer the question."
            )
        if _simple_question(question) and _word_count(extract) > 42:
            errors.append(
                "extract is too broad for a simple question. Keep only the minimal evidence needed."
            )

    return tuple(errors)


def _lint_error_message(errors: tuple[str, ...]) -> str | None:
    if not errors:
        return None
    return " ".join(errors)


def _tag_value(text: str, tag: str) -> str | None:
    match = re.search(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def _extra_numbers(text: str, *, allowed_text: str) -> list[str]:
    allowed = set(_numbers(allowed_text))
    return [number for number in _numbers(text) if number not in allowed]


def _numbers(text: str) -> list[str]:
    return re.findall(r"\b\d+(?:\.\d+)?\b", text)


def _allowed_extra_number_count(question: str) -> int:
    lowered = question.lower()
    if any(term in lowered for term in ("how many", "count", "number of")):
        return 1
    if any(term in lowered for term in ("increase", "decrease", "change", "difference", "higher", "lower", "minimum", "maximum")):
        return 2
    return 1


def _simple_question(question: str) -> bool:
    lowered = question.lower()
    return lowered.startswith(("is ", "are ", "was ", "were ", "does ", "do ", "how many"))


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


__all__ = [
    "PRIVILEGED_SYNTHESIS_PROMPT",
    "STAGE1_SYNTHESIS_PROMPT",
    "SynthesisResult",
    "build_synthesis_prompt",
    "lint_synthesis",
    "synthesize",
    "synthesize_with_retries",
]
