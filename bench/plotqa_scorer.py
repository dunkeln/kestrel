import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlotQAScore:
    score: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PlotQASeries:
    name: str
    x: tuple[str, ...]
    y: tuple[float, ...]


@dataclass(frozen=True)
class PlotQAFact:
    series: str
    x: str
    y: float


@dataclass(frozen=True)
class PrecisionRecallF1:
    precision: float
    recall: float
    f1: float


def score_plotqa_structure(gold: str, prediction: str) -> PlotQAScore:
    gold_series = parse_plotqa_structure(gold)
    predicted_series = parse_plotqa_structure(prediction)
    gold_facts = _facts(gold_series)
    predicted_facts = _facts(predicted_series)
    raw_exact_match = _normalize_text(gold) == _normalize_text(prediction)

    if not gold_facts or not predicted_facts:
        return PlotQAScore(
            score=1.0 if raw_exact_match else 0.0,
            metadata={
                "plotqa_gold_parse_ok": bool(gold_facts),
                "plotqa_prediction_parse_ok": bool(predicted_facts),
                "plotqa_series_precision": 0.0,
                "plotqa_series_recall": 0.0,
                "plotqa_series_f1": 0.0,
                "plotqa_x_precision": 0.0,
                "plotqa_x_recall": 0.0,
                "plotqa_x_f1": 0.0,
                "plotqa_x_label_precision": 0.0,
                "plotqa_x_label_recall": 0.0,
                "plotqa_x_label_f1": 0.0,
                "plotqa_point_precision": 0.0,
                "plotqa_point_recall": 0.0,
                "plotqa_point_f1": 1.0 if raw_exact_match else 0.0,
                "plotqa_count_match": 0.0,
                "plotqa_gold_series": len(gold_series),
                "plotqa_predicted_series": len(predicted_series),
                "plotqa_gold_facts": len(gold_facts),
                "plotqa_predicted_facts": len(predicted_facts),
                "plotqa_parse_ok": bool(gold_facts and predicted_facts),
                "plotqa_exact_match": raw_exact_match,
            },
        )

    series_metrics = _set_prf(
        {_normalize_text(series.name) for series in gold_series},
        {_normalize_text(series.name) for series in predicted_series},
    )
    x_metrics = _set_prf(
        {_normalize_text(x) for series in gold_series for x in series.x},
        {_normalize_text(x) for series in predicted_series for x in series.x},
    )
    point_metrics = _fact_prf(gold_facts, predicted_facts)
    count_score = 1.0 if len(gold_facts) == len(predicted_facts) else 0.0
    return PlotQAScore(
        score=point_metrics.f1,
        metadata={
            "plotqa_gold_parse_ok": True,
            "plotqa_prediction_parse_ok": True,
            "plotqa_series_precision": series_metrics.precision,
            "plotqa_series_recall": series_metrics.recall,
            "plotqa_series_f1": series_metrics.f1,
            "plotqa_x_precision": x_metrics.precision,
            "plotqa_x_recall": x_metrics.recall,
            "plotqa_x_f1": x_metrics.f1,
            "plotqa_x_label_precision": x_metrics.precision,
            "plotqa_x_label_recall": x_metrics.recall,
            "plotqa_x_label_f1": x_metrics.f1,
            "plotqa_point_precision": point_metrics.precision,
            "plotqa_point_recall": point_metrics.recall,
            "plotqa_point_f1": point_metrics.f1,
            "plotqa_count_match": count_score,
            "plotqa_gold_series": len(gold_series),
            "plotqa_predicted_series": len(predicted_series),
            "plotqa_gold_facts": len(gold_facts),
            "plotqa_predicted_facts": len(predicted_facts),
            "plotqa_parse_ok": True,
            "plotqa_exact_match": raw_exact_match,
        },
    )


def parse_plotqa_structure(value: str) -> list[PlotQASeries]:
    value_without_bboxes = _remove_bbox_blocks(value)
    series = _parse_plotqa_tags(value_without_bboxes)
    if series:
        return series

    return _parse_series_tags(value_without_bboxes)


def _parse_plotqa_tags(value: str) -> list[PlotQASeries]:
    names = [_strip_tag_text(text) for text in _tag_values(value, "s_name")]
    x_groups = [_split_values(text) for text in _tag_values(value, "s_x")]
    y_groups = [_split_values(text) for text in _tag_values(value, "s_y")]
    if not y_groups:
        return []

    series: list[PlotQASeries] = []
    for index, raw_y_values in enumerate(y_groups):
        raw_x_values = _select_group(x_groups, index)
        x_values, y_values = _normalize_axis_values(raw_x_values, raw_y_values)
        if not x_values or not y_values:
            continue
        name = names[index] if index < len(names) and names[index] else f"series_{index}"
        length = min(len(x_values), len(y_values))
        series.append(
            PlotQASeries(
                name=name,
                x=tuple(x_values[:length]),
                y=tuple(y_values[:length]),
            )
        )
    return series


def _parse_series_tags(value: str) -> list[PlotQASeries]:
    parsed: list[PlotQASeries] = []
    for attributes, body in _series_blocks(value):
        attr_values = _parse_attributes(attributes)
        name = (
            attr_values.get("name")
            or attr_values.get("label")
            or _first_tag_value(body, "name")
            or _first_tag_value(body, "s_name")
            or ""
        )
        raw_x_values = _values_from_variant(attr_values.get("x") or _first_tag_value(body, "x") or _first_tag_value(body, "s_x"))
        raw_y_values = _values_from_variant(attr_values.get("y") or _first_tag_value(body, "y") or _first_tag_value(body, "s_y"))
        x_values, y_values = _normalize_axis_values(raw_x_values, raw_y_values)
        if not name or not x_values or not y_values:
            continue

        length = min(len(x_values), len(y_values))
        parsed.append(
            PlotQASeries(
                name=_strip_tag_text(name),
                x=tuple(x_values[:length]),
                y=tuple(y_values[:length]),
            )
        )
    return parsed


def _tag_values(value: str, tag: str) -> list[str]:
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.IGNORECASE | re.DOTALL)
    return pattern.findall(str(value))


def _first_tag_value(value: str, tag: str) -> str | None:
    values = _tag_values(value, tag)
    if not values:
        return None
    return values[0]


def _series_blocks(value: str) -> list[tuple[str, str]]:
    text = str(value)
    blocks = re.findall(r"<series\b([^>]*)>(.*?)</series>", text, flags=re.IGNORECASE | re.DOTALL)
    self_closing = [
        (attributes, "")
        for attributes in re.findall(r"<series\b([^>]*)/>", text, flags=re.IGNORECASE | re.DOTALL)
    ]
    return blocks + self_closing


def _parse_attributes(value: str) -> dict[str, str]:
    return {
        key.lower(): raw_value
        for key, raw_value in re.findall(r"""([\w:-]+)\s*=\s*["']([^"']*)["']""", value)
    }


def _remove_bbox_blocks(value: str) -> str:
    return re.sub(r"<s_bboxes>.*?</s_bboxes>", "", str(value), flags=re.IGNORECASE | re.DOTALL)


def _split_values(value: str) -> list[str]:
    return [
        _strip_tag_text(item)
        for item in re.split(r"<sep\s*/>", value, flags=re.IGNORECASE)
        if _strip_tag_text(item)
    ]


def _values_from_variant(value: str | None) -> list[str]:
    if value is None:
        return []
    text = _strip_tag_text(value)
    if not text:
        return []
    if re.search(r"<sep\s*/>", text, flags=re.IGNORECASE):
        return _split_values(text)

    delimiter_pattern = r"[,;\n]+"
    if re.search(delimiter_pattern, text):
        return [
            _strip_tag_text(item)
            for item in re.split(delimiter_pattern, text)
            if _strip_tag_text(item)
        ]

    whitespace_values = [_strip_tag_text(item) for item in text.split() if _strip_tag_text(item)]
    if len(whitespace_values) > 1 and _parse_all_float_values(whitespace_values) is not None:
        return whitespace_values

    return [text]


def _normalize_axis_values(
    raw_x_values: list[str],
    raw_y_values: list[str],
) -> tuple[list[str], list[float]]:
    if _contains_placeholder(raw_x_values) or _contains_placeholder(raw_y_values):
        return [], []

    y_numbers = _parse_all_float_values(raw_y_values)
    if y_numbers is not None:
        return raw_x_values, y_numbers

    x_numbers = _parse_all_float_values(raw_x_values)
    if x_numbers is not None:
        return raw_y_values, x_numbers

    return [], []


def _contains_placeholder(values: list[str]) -> bool:
    placeholders = {"x", "y", "x1", "x2", "y1", "y2", "series name"}
    return any(_normalize_text(value) in placeholders for value in values)


def _parse_all_float_values(values: list[str]) -> list[float] | None:
    if not values:
        return None
    parsed: list[float] = []
    for item in values:
        try:
            parsed.append(float(item.replace(",", "").rstrip("%")))
        except ValueError:
            return None
    return parsed


def _strip_tag_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def _select_group(groups: list[list[str]], index: int) -> list[str]:
    if not groups:
        return []
    if len(groups) == 1:
        return groups[0]
    if index < len(groups):
        return groups[index]
    return groups[-1]


def _fact_prf(
    gold_facts: list[PlotQAFact],
    predicted_facts: list[PlotQAFact],
    tolerance: float = 1e-3,
) -> PrecisionRecallF1:
    if not gold_facts and not predicted_facts:
        return PrecisionRecallF1(precision=1.0, recall=1.0, f1=1.0)
    if not gold_facts or not predicted_facts:
        return PrecisionRecallF1(precision=0.0, recall=0.0, f1=0.0)

    unmatched = list(predicted_facts)
    matches = 0
    for gold in gold_facts:
        match_index = next(
            (
                index
                for index, predicted in enumerate(unmatched)
                if _fact_matches(gold, predicted, tolerance)
            ),
            None,
        )
        if match_index is not None:
            matches += 1
            unmatched.pop(match_index)

    precision = matches / len(predicted_facts)
    recall = matches / len(gold_facts)
    return PrecisionRecallF1(
        precision=precision,
        recall=recall,
        f1=_harmonic_mean(precision, recall),
    )


def _facts(series: list[PlotQASeries]) -> list[PlotQAFact]:
    return [
        PlotQAFact(
            series=_normalize_text(item.name),
            x=_normalize_text(x),
            y=y,
        )
        for item in series
        for x, y in zip(item.x, item.y, strict=False)
    ]


def _fact_matches(
    gold: PlotQAFact,
    predicted: PlotQAFact,
    tolerance: float,
) -> bool:
    return (
        gold.series == predicted.series
        and gold.x == predicted.x
        and abs(gold.y - predicted.y) <= tolerance
    )


def _set_prf(gold: set[str], predicted: set[str]) -> PrecisionRecallF1:
    if not gold and not predicted:
        return PrecisionRecallF1(precision=1.0, recall=1.0, f1=1.0)
    if not gold or not predicted:
        return PrecisionRecallF1(precision=0.0, recall=0.0, f1=0.0)
    matches = len(gold & predicted)
    precision = matches / len(predicted)
    recall = matches / len(gold)
    return PrecisionRecallF1(
        precision=precision,
        recall=recall,
        f1=_harmonic_mean(precision, recall),
    )


def _harmonic_mean(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().strip(" \t\r\n.!?").split())
