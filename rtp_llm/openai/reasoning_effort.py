"""Model-specific reasoning effort validation without JSON type coercion."""

from typing import Any

V41_EFFORT_BUDGETS = {"low": 25, "high": 50, "xhigh": 75, "max": 100}


def normalize_v41_reasoning_effort(value: Any) -> int:
    if value is None:
        return 50
    if type(value) is int and 1 <= value <= 100:
        return value
    if type(value) is str and value in V41_EFFORT_BUDGETS:
        return V41_EFFORT_BUDGETS[value]
    raise ValueError(
        "reasoning_effort must be an integer in 1..100 or low/high/xhigh/max"
    )


def validate_reasoning_effort_for_model(value: Any, model_type: str) -> None:
    if model_type == "deepseek_v41":
        normalize_v41_reasoning_effort(value)
    elif value is not None and type(value) is not str:
        raise ValueError("reasoning_effort must be a string for this model")
