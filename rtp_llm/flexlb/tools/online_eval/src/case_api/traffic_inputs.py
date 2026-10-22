"""Compose the complete shared Java flow contract into authored case inputs."""

from adapters.java_flow_spec import JavaFlowSpec
from case_api.input_contract import section_fields
from case_api.numeric_parameters import SOURCE_NUMBERS, JAVA_FLOW_NUMBERS


def java_flow_inputs(*, max_requests=False, numbers=None):
    definitions = JavaFlowSpec.input_fields(max_requests=max_requests)
    rules = {**SOURCE_NUMBERS, **JAVA_FLOW_NUMBERS, **(numbers or {}),
             **{'traffic.' + name: rule for name, rule in definitions.items() if rule is not None}}
    return section_fields('traffic', definitions.keys() | {'kind'}, rules)
