from dataclasses import dataclass
from typing import Annotated
import pytest
from mock_framework.config import ConfigError, Registry, Step
from mock_framework.config.model import Reference
from mock_framework.config.registry import case_interface, Bounds
from mock_framework.config.environment import expand_config


def test_compiles_without_java_catalog_and_read_can_have_no_assertion(case_data, compile_data):
    case = compile_data(case_data)
    assert case.environment['prefill']['config'] == {'a_b_c': 128, 'cache': {'sync': {'delay_ms': 10}}}
    steps = case.phases[0].steps
    assert all(isinstance(s, Step) for s in steps)
    assert steps[1].expect is None
    assert steps[1].args['targets'] == Reference('leader', ())
    assert steps[3].expect.comparisons['gte'] == Reference('tps', ())
    assert str(case.view).endswith('shared/views/default/dashboard.json')


@pytest.mark.parametrize('change', [
    lambda d: d.update(schema_version=3),
    lambda d: d['environment']['prefill']['deployment'].update(replicas=True),
    lambda d: d['phases'][0].update(parallel=True),
    lambda d: d['phases'][0]['steps'][0].update(run='background'),
    lambda d: d['phases'][0]['steps'][0].update(interface='client/traffic/run'),
    lambda d: d['phases'][0]['steps'][0].update(within='-1s'),
    lambda d: d['phases'][0]['steps'][1]['args'].update(targets={'ref': 'leader.nodes'}),
    lambda d: d['phases'][0]['steps'][1]['args'].update(targets={'ref': 'steps.tps'}),
    lambda d: d['phases'][0]['steps'][1]['args'].update(targets=[]),
    lambda d: d['phases'][0]['steps'][1]['args'].update(unknown=1),
    lambda d: d['phases'][0]['steps'][2]['args'].update(range=['0s', '0s']),
    lambda d: d['phases'][0]['steps'][2]['args'].update(range='all'),
    lambda d: d['phases'][0]['steps'][2].update(expect={'count': 1}),
    lambda d: d['phases'][0]['steps'][2].update(expect={'gte': 2, 'lte': 1}),
    lambda d: d['phases'][0]['steps'][3].update(expect={'gte': {'ref': 'steps.leader'}}),
    lambda d: d['phases'][0]['steps'][3].update(expect={'gte': {'ref': 'steps.tps.value'}}),
])
def test_rejects_invalid_case_before_execution(case_data, compile_data, change):
    case_data['environment']['prefill']['deployment'] = {}
    change(case_data)
    with pytest.raises(ConfigError):
        compile_data(case_data)


def test_old_call_check_wrappers_are_rejected(case_data, compile_data):
    case_data['phases'][0]['steps'][0] = {'check': case_data['phases'][0]['steps'][0]}
    with pytest.raises(ConfigError):
        compile_data(case_data)


def test_global_reference_is_typed(case_data, compile_data):
    case_data['phases'][0]['steps'][0]['args'] = {'targets': {'ref': 'globals.masters'}}
    assert compile_data(case_data).phases[0].steps[0].args['targets'] == Reference('globals', ('masters',))


@dataclass(frozen=True)
class ExtensionArgs:
    threshold: Annotated[int, Bounds(1, 8)]


@case_interface('extension/queue', effect='read')
async def queue(ctx, args: ExtensionArgs) -> int:
    return args.threshold


def test_function_registration_defines_contract_without_compiler_changes(case_data, compile_data):
    registry = Registry((queue,))
    case_data['phases'][0]['steps'] = [{'id': 'extension', 'describe': 'Python extension',
        'interface': 'extension/queue', 'args': {'threshold': 4}, 'expect': {'lte': 8}}]
    assert compile_data(case_data, interfaces=registry).phases[0].steps[0].args == {'threshold': 4}
    for bad in ('4', True, 0, 9):
        case_data['phases'][0]['steps'][0]['args']['threshold'] = bad
        with pytest.raises(ConfigError):
            compile_data(case_data, interfaces=registry)
    with pytest.raises(ConfigError, match='duplicate'):
        registry.register(queue)


@pytest.mark.parametrize('raw', [
    {'cache': 1, 'cache.sync.enabled': True}, {'cache.sync.enabled': True, 'cache': 1},
    {'cache': {'sync': {}}, 'cache.sync.enabled': True}, {'cache..enabled': True},
])
def test_config_collisions(raw):
    with pytest.raises(ConfigError):
        expand_config(raw)


def test_native_payload_and_duplicate_keys(case_data, compile_data):
    assert expand_config({'raw_json': {'original.key': 'untouched'}}) == {'raw_json': {'original.key': 'untouched'}}
    case_data['environment']['master']['config']['FLEXLB_CONFIG'] = '{"scheduler":{},"scheduler":{}}'
    with pytest.raises(ConfigError, match='Duplicate native JSON'):
        compile_data(case_data)


def test_nullable_reference_preserves_both_possible_types():
    from mock_framework.config.compiler import compatible
    from mock_framework.config.registry import schema
    assert compatible(schema(list[str] | None), schema(list[str] | None))
    assert not compatible(schema(list[str] | None), schema(list[str]))


def test_typed_records_inside_result_collections_remain_referenceable_values():
    import asyncio
    @case_interface('extension/records', effect='read')
    async def records(ctx, args: ExtensionArgs) -> dict[str, list[ExtensionArgs]]:
        return {'records': [args]}
    registry = Registry((records,))
    assert asyncio.run(registry.get('extension/records').invoke(None, {'threshold': 4})) == {'records': [{'threshold': 4}]}


@pytest.mark.parametrize('wait', ['0s', '20ms', '1.5s'])
def test_wait_after_is_a_step_duration(case_data, compile_data, wait):
    assert compile_data(case_data).phases[0].steps[0].wait_after == '0s'
    case_data['phases'][0]['steps'][0]['wait_after'] = wait
    assert compile_data(case_data).phases[0].steps[0].wait_after == wait


@pytest.mark.parametrize('wait', ['-1s', '1', 1, True, None, {'ref': 'steps.leader'}])
def test_invalid_wait_after_rejected_before_execution(case_data, compile_data, wait):
    case_data['phases'][0]['steps'][0]['wait_after'] = wait
    with pytest.raises(ConfigError):
        compile_data(case_data)


def test_wait_interface_is_removed(case_data, compile_data):
    assert 'runtime/wait' not in Registry.standard().interfaces
    case_data['phases'][0]['steps'][0] = {'id': 'wait', 'describe': 'old wait',
        'interface': 'runtime/wait', 'args': {'duration': '1s'}}
    with pytest.raises(ConfigError, match='Unknown or unimplemented'):
        compile_data(case_data)
