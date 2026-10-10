"""Action adapter for the shared strict field-set contract."""

import copy
from input_contract import mapping_fields


def validate_fields(params, plan, fields, required=()):
    try:
        return copy.deepcopy(mapping_fields(params, fields, plan.path, required=required))
    except ValueError as exc:
        raise ValueError(f"{plan.path}: invalid action parameters: {exc}") from exc
