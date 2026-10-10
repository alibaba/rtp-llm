"""Program-owned numeric types; YAML can only narrow their legal ranges."""

from input_contract import NumberRule
from traffic.contracts import PRIORITY, JAVA_LENGTH


NONNEGATIVE = NumberRule(integer=False, minimum=0)
COUNT = NumberRule(integer=True, minimum=0)
POSITIVE_COUNT = NumberRule(integer=True, minimum=1)
FRACTION = NumberRule(integer=False, minimum=0, maximum=1)
OFFSET = NumberRule(integer=False)


def number_fields(rule, *paths):
    return dict.fromkeys(paths, rule)


SOURCE_NUMBERS = {
    **number_fields(COUNT, 'traffic.source.parameters.count',
                    'traffic.source.parameters.output_tokens'),
    'traffic.source.parameters.priority': PRIORITY,
}
OUTPUT_DISTRIBUTION_NUMBERS = number_fields(COUNT,
    'traffic.source.parameters.output_distribution.mean_tokens',
    'traffic.source.parameters.output_distribution.seed')
JAVA_FLOW_NUMBERS = {
    **number_fields(NONNEGATIVE, 'traffic.client.playback.qps', 'traffic.poll_s'),
    'traffic.client.playback.max_laps': COUNT,
}
CAPTURE_NUMBERS = number_fields(POSITIVE_COUNT,
    'observation.capture.max_samples', 'observation.capture.max_bytes')

# Shared paths have one meaning across programs. A binding cannot silently change
# a shared field's type/range; scenario-specific budgets belong in YAML narrowing.
COMMON_NUMBERS = {
    **SOURCE_NUMBERS, **OUTPUT_DISTRIBUTION_NUMBERS, **JAVA_FLOW_NUMBERS,
    **CAPTURE_NUMBERS, 'procedure.setup_timeout_s': NONNEGATIVE,
    'observation.sample_s': NONNEGATIVE, 'observation.max_gap_s': NONNEGATIVE,
    'traffic.input_len': JAVA_LENGTH, 'traffic.output_len': JAVA_LENGTH,
    'traffic.count': POSITIVE_COUNT, 'checks.offered_load.expected': FRACTION,
}


def parameter_rules(defaults):
    from scenario.loader import ScenarioError

    if not isinstance(defaults, dict):
        raise ScenarioError("numeric parameter contract must be a mapping")
    for path, rule in defaults.items():
        if (type(path) is not str or not path or any(not part for part in path.split('.'))
                or not isinstance(rule, NumberRule)):
            raise ScenarioError("invalid program numeric parameter contract")
        if path in COMMON_NUMBERS and rule != COMMON_NUMBERS[path]:
            raise ScenarioError("conflicting shared numeric field: " + path)
    return dict(defaults)


def narrow_parameters(rules, refinements):
    from scenario.loader import ScenarioError

    if not isinstance(refinements, dict):
        raise ScenarioError("parameter_schema must be a mapping")
    unknown = set(refinements) - set(rules)
    if unknown:
        raise ScenarioError("parameter_schema: unknown configuration fields " + str(sorted(unknown)))
    try:
        return {path: rule.narrow(refinements[path], path) if path in refinements else rule
                for path, rule in rules.items()}
    except ValueError as exc:
        raise ScenarioError(str(exc)) from exc
