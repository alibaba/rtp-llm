"""Stage state and variant dimensions stay isolated across compiled plans."""

import copy
from unittest.mock import patch

import pytest

from scenario.program_compiler import configure_program
from scenario.contracts import StageHandler
from scenario.errors import ScenarioError
from scenario.document_io import load_document
from scenario.stage_compiler import stages


@pytest.mark.parametrize('suffix, message', [
    ([dict(id='again', action='setup')], 'setup must occur exactly once'),
    ([dict(id='close', action='teardown'), dict(id='again', action='request')], 'no stage may follow teardown'),
])
def test_setup_and_teardown_order_is_strict(suffix, message):
    with pytest.raises(ScenarioError, match=message):
        stages([dict(id='setup', action='setup'), *suffix], 'plan.stages', 5, {})


def test_compilation_has_fresh_outputs_and_lifecycle_state_after_failure():
    setup = dict(id='setup', action='setup')
    request = dict(id='send', action='request')
    wait = dict(id='wait', action='wait', params={'requests': {'$ref': 'stages.send.output.requests'}})
    first = stages([setup, request, wait], 'plan.stages', 5, {})
    with pytest.raises(ScenarioError, match='unknown or forward reference'):
        stages([setup, wait], 'plan.stages', 5, {})
    assert stages([setup, request, wait], 'plan.stages', 5, {}) == first
    assert 'params' not in request


def test_handler_replacement_environment_reaches_only_later_stages():
    seen = []
    replacement = {'nested': {'workers': [2, 3]}}
    original = {'nested': {'workers': [1]}}

    def validate(params, plan):
        seen.append(copy.deepcopy(plan.environment))
        plan.environment['nested']['workers'].append(99)
        return params

    replace = StageHandler('replace', validate, lambda *_: None, {},
                           next_environment=lambda _: replacement)
    observe = StageHandler('observe', validate, lambda *_: None, {})
    sequence = [dict(id='setup', action='setup'), dict(id='replace', action='replace'),
                dict(id='observe', action='observe')]
    for _ in range(2):
        stages(sequence, 'plan.stages', 5, {'replace': replace, 'observe': observe}, env=original)
    assert seen == [original, replacement, original, replacement]
    assert original == {'nested': {'workers': [1]}}
    assert replacement == {'nested': {'workers': [2, 3]}}


@pytest.mark.parametrize('variants', [None, [], {}, 'unknown'])
def test_malformed_variants_are_configuration_errors(variants):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    config = load_document(root / 'config/scenarios/request_completion.yaml')
    config['variants'] = variants
    with pytest.raises(ScenarioError, match='expected a nonempty list'):
        configure_program(config, 'case.yaml')


def test_variant_parameter_guard_runs_before_the_variant_program():
    from pathlib import Path
    from cases.request_completion import program
    from dataclasses import replace
    from case_api import registry
    from case_registry_fixtures import snapshot
    from unittest.mock import Mock
    root = Path(__file__).resolve().parents[1]
    config = load_document(root / 'config/scenarios/request_completion.yaml')
    config['variants'] = [{'id': 'invalid', 'parameters': {'traffic': {'count': 0}}}]
    build = Mock(wraps=program.default)
    original = registry.registry().cases['request_completion']
    changed = replace(original, definition=replace(original.definition, builders={'default': build}))
    with patch.object(registry, '_snapshot', snapshot(changed, include_existing=True)):
        with pytest.raises(ScenarioError, match='traffic.count'):
            configure_program(config, 'case.yaml')
    assert build.call_count == 1
