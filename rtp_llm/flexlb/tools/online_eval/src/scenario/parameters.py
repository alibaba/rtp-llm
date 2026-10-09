"""Strict mapping validation shared by action parameter schemas."""

import copy


def validate_fields(params, plan, fields, required=()):
    if (
        not isinstance(params, dict)
        or set(params) - set(fields)
        or set(required) - set(params)
    ):
        raise ValueError(f"{plan.path}: invalid action parameters")
    return copy.deepcopy(params)
