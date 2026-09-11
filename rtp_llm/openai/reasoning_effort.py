"""Model-specific reasoning effort validation without JSON type coercion."""

from typing import Any

V41_EFFORT_BUDGETS = {
    "minimal": 50,
    "low": 50,
    "medium": 75,
    "high": 75,
    "xhigh": 75,
    "max": 100,
}


def normalize_v41_reasoning_effort(value: Any) -> int:
    # The request converter handles `none` as a thinking switch. Explicitly
    # enabled thinking with `none` uses the recipe's default effort.
    if value is None or value == "none":
        return 75
    if type(value) is int and 1 <= value <= 100:
        return value
    if type(value) is str and value in V41_EFFORT_BUDGETS:
        return V41_EFFORT_BUDGETS[value]
    raise ValueError(
        "reasoning_effort must be an integer in 1..100 or "
        "none/minimal/low/medium/high/xhigh/max"
    )


def validate_reasoning_effort_for_model(value: Any, model_type: str) -> None:
    if model_type == "deepseek_v41":
        normalize_v41_reasoning_effort(value)
    elif value is not None and type(value) is not str:
        raise ValueError("reasoning_effort must be a string for this model")
