"""Action adapter for the shared strict field-set contract."""

import copy
from input_contract import mapping_fields


def validate_fields(params, plan, fields, required=()):
    return copy.deepcopy(mapping_fields(params, fields, plan.path, required=required))


def reference_parameters(params, plan, *, resources):
    value = validate_fields(params, plan, resources, resources)
    for name, kind in resources.items():
        plan.reference(value[name], kind)
    return value
