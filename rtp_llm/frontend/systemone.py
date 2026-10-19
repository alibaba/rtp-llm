"""Pure SystemOne protocol conversion for a single deployed Kev model.

Rendering and answer semantics follow jaredpalmer/kev's api.py at revision
1f04cfa28e6a708485b28a9585010b9e436b776e. This adapter has no model dependencies.
"""

from __future__ import annotations

import math
from typing import Any


def _check_json(value: Any, path: str, ancestors: set[int]) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return
    if not isinstance(value, (dict, list)):
        raise ValueError(f"{path} must contain only JSON values")
    identity = id(value)
    if identity in ancestors:
        raise ValueError(f"{path} must not contain a circular reference")
    ancestors.add(identity)
    try:
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"{path} must have string object keys")
                _check_json(item, f"{path}.{key}", ancestors)
        else:
            for index, item in enumerate(value):
                _check_json(item, f"{path}[{index}]", ancestors)
    finally:
        ancestors.remove(identity)


def _render(value: Any, indent: int = 0) -> str:
    if value is None:
        return ""
    if not isinstance(value, (dict, list)):
        return str(value)
    padding = "  " * indent
    if isinstance(value, list):
        return "\n".join(
            f"{padding}- {_render(item, indent + 1).lstrip()}" for item in value
        )
    lines = []
    for key, item in value.items():
        if isinstance(item, (dict, list)):
            lines.append(f"{padding}{key}:\n{_render(item, indent + 1)}")
        else:
            lines.append(f"{padding}{key}: {_render(item)}")
    return "\n".join(lines)


def _option_text(name: str, description: Any) -> str:
    if description is None or description == "":
        return name
    return f"{name}: {_render(description)}"


def _question(qid: str, question: Any, max_options: int) -> tuple[dict, dict]:
    if not isinstance(question, dict):
        raise ValueError(f"questions[{qid!r}] must be an object")
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "noul":
        if criteria is not None and not isinstance(criteria, dict):
            raise ValueError(f"questions[{qid!r}].criteria must be an object or null")
        criteria = criteria or {}
        keys = ["false", "true"]
        options = [
            _option_text("no", criteria.get("false")),
            _option_text("yes", criteria.get("true")),
        ]
    elif kind == "choice":
        if not isinstance(criteria, dict):
            raise ValueError(f"questions[{qid!r}].criteria must be an object")
        keys = list(criteria)
        options = [_option_text(key, value) for key, value in criteria.items()]
    elif kind == "score":
        if not isinstance(criteria, list):
            raise ValueError(f"questions[{qid!r}].criteria must be an array")
        keys = [str(index) for index in range(len(criteria))]
        options = [_render(value) for value in criteria]
    else:
        raise ValueError(f"questions[{qid!r}].type must be choice, score, or noul")
    if not 1 <= len(options) <= max_options:
        raise ValueError(f"questions[{qid!r}] must have 1..{max_options} options")
    if any(not option for option in options):
        raise ValueError(f"questions[{qid!r}] options must render as nonempty strings")
    metadata = {"id": qid, "type": kind, "keys": keys}
    if kind == "score":
        metadata["legend"] = dict(zip(keys, options))
    record = {"instr": _render(question.get("instructions")), "options": options}
    return record, metadata


def parse_request(
    request: dict, *, max_options: int = 32
) -> tuple[str, dict, list[dict]]:
    """Validate JSON input and return model name, codec record, and answer metadata.

    Question and criterion insertion order is preserved. The model string is
    returned verbatim; routing to another deployed model is not performed here.
    Invalid requests raise ValueError before any inference work is enqueued.
    """
    if type(max_options) is not int or max_options < 1:
        raise ValueError("max_options must be a positive integer")
    if not isinstance(request, dict):
        raise ValueError("SystemOne request must be an object")
    _check_json(request, "request", set())
    if "state" not in request:
        raise ValueError("state is required")
    model = request.get("model", "kev-latest")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty string")
    questions = request.get("questions")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
        raise ValueError("questions must be an object with 1..64 entries")
    records, metadata = [], []
    for qid, question in questions.items():
        record, item_metadata = _question(qid, question, max_options)
        records.append(record)
        metadata.append(item_metadata)
    return model, {"state": _render(request["state"]), "questions": records}, metadata


def _confidence(probabilities: list[float], kind: str) -> float:
    count = len(probabilities)
    if count == 1:
        return 1.0
    total = sum(probabilities)
    normalized = (
        [value / total for value in probabilities] if total else [1 / count] * count
    )
    if kind == "choice":
        uniform = 1 / count
        return (max(normalized) - uniform) / (1 - uniform)
    mode = max(range(count), key=normalized.__getitem__)
    spread = sum(abs(level - (count - 1) / 2) for level in range(count)) / count
    deviation = sum(
        probability * abs(level - mode) for level, probability in enumerate(normalized)
    )
    return max(0.0, 1.0 - deviation / spread)


def _round(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("SystemOne answer must contain only finite numbers")
    return round(float(value), 4)


def _validate_probabilities(values: Any, option_count: int) -> list[float]:
    if not isinstance(values, list) or len(values) != option_count:
        raise ValueError("probability row must match its question's option count")
    probabilities = []
    for value in values:
        if type(value) not in (int, float):
            raise ValueError("probabilities must be finite nonnegative numbers")
        try:
            value = float(value)
        except OverflowError as error:
            raise ValueError(
                "probabilities must be finite nonnegative numbers"
            ) from error
        if not math.isfinite(value) or value < 0:
            raise ValueError("probabilities must be finite nonnegative numbers")
        probabilities.append(value)
    if not math.isfinite(sum(probabilities)):
        raise ValueError("probability total must be finite")
    return probabilities


def to_answers(probs: list[list[float]], meta: list[dict]) -> dict:
    """Convert one unpadded probability row per question to SystemOne answers."""
    if (
        not isinstance(probs, list)
        or not isinstance(meta, list)
        or len(probs) != len(meta)
    ):
        raise ValueError("probability rows must match question metadata")
    answers = {}
    for values, item in zip(probs, meta):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("question metadata must contain a string id")
        qid, kind, keys = item["id"], item.get("type"), item.get("keys")
        if (
            kind not in ("noul", "choice", "score")
            or not isinstance(keys, list)
            or not keys
            or any(not isinstance(key, str) for key in keys)
            or len(set(keys)) != len(keys)
            or qid in answers
        ):
            raise ValueError("invalid or duplicate question metadata")
        probabilities = _validate_probabilities(values, len(keys))
        if kind == "noul":
            if keys != ["false", "true"]:
                raise ValueError("noul metadata must use false/true option order")
            answers[qid] = {"type": kind, "noul": _round(probabilities[1])}
            continue
        distribution = {key: _round(value) for key, value in zip(keys, probabilities)}
        confidence = _round(_confidence(probabilities, kind))
        if kind == "choice":
            winner = max(range(len(keys)), key=probabilities.__getitem__)
            answers[qid] = {
                "type": kind,
                "choice": keys[winner],
                "confidence": confidence,
                "probabilities": distribution,
            }
        else:
            if keys != [str(index) for index in range(len(keys))] or not isinstance(
                item.get("legend"), dict
            ):
                raise ValueError("score metadata requires indexed levels and a legend")
            answers[qid] = {
                "type": kind,
                "score": _round(
                    sum(i * value for i, value in enumerate(probabilities))
                ),
                "legend": item["legend"],
                "probabilities": distribution,
                "confidence": confidence,
            }
    return answers
