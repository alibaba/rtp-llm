from copy import deepcopy

import pytest

from mock_framework.config import ConfigError, CallStep, CheckStep, Registry
from mock_framework.config.model import Reference
from mock_framework.config.environment import expand_config
from conftest import ROOT


def test_compiles_multi_jvm_config_references_and_unresolved_ranges(case_data, compile_data):
    case = compile_data(case_data)
    assert case.environment['client']['deployment']['replicas'] == 4
    assert case.environment['client']['config']['driver']['channels'] == 8
    assert case.environment['prefill']['config'] == {'a_b_c': 128, 'cache': {'sync': {'delay_ms': 10}}}
    assert case.environment['master']['config']['FLEXLB_CONFIG'].startswith('{"schemaVersion"')
    steps = case.phases[0].steps
    assert isinstance(steps[0], CallStep) and steps[0].run == 'background'
    assert isinstance(steps[3], CheckStep)
    assert steps[3].expect.comparisons['eq'] == Reference('removal', ('remaining',))
    assert steps[4].args == {'stat': 'max', 'range': 'all'}
    assert str(case.view).endswith('shared/views/default/dashboard.json')
    assert str(steps[0].args['input']['trace']).endswith('shared/datasets/example.pb.zst')
    assert steps[5].args == {'promql': 'max_over_time(example_metric[1m])'}


@pytest.mark.parametrize('change, message', [
    (lambda d: d.update(schema_version=3), 'unknown'),
    (lambda d: d['environment']['client']['deployment'].update(replicas=True), 'replicas'),
    (lambda d: d['environment']['prefill'].update(engine='vllm'), 'Only RTP'),
    (lambda d: d['phases'][0].update(parallel=True), 'unknown'),
    (lambda d: d['phases'][0]['steps'][0]['call']['args']['playback']['arrival'].update(qps='100'), 'number'),
    (lambda d: d['phases'][0]['steps'][0]['call']['args']['playback']['arrival'].update(speed=2), 'forbids speed'),
    (lambda d: d['phases'][0]['steps'][0]['call']['args']['input'].update(parameters={}), 'unknown'),
    (lambda d: d['phases'][0]['steps'][4]['check']['args'].update(stat='median'), 'expected one of'),
    (lambda d: d['phases'][0]['steps'][4]['check']['args'].pop('stat'), 'missing'),
    (lambda d: d['phases'][0]['steps'][4]['check'].update(interface='client/requests/ttft_max'), 'Unknown interface'),
    (lambda d: d['phases'][0]['steps'][4]['check']['args'].update(range='lastest'), 'unsupported range'),
    (lambda d: d['phases'][0]['steps'][4]['check']['args'].update(range='interval'), 'unsupported range'),
    (lambda d: d['phases'][0]['steps'][4]['check']['args'].update(range=['0s', '0s']), 'start < end'),
    (lambda d: d['phases'][0]['steps'][5]['check']['args'].update(range=['-1m', '0s']), 'unknown'),
    (lambda d: d['phases'][0]['steps'][2]['call'].update(run='background'), 'run mode'),
    (lambda d: d['phases'][0]['steps'][2]['call']['args'].update(count=0), 'minimum'),
    (lambda d: d['phases'][0]['steps'][2]['call']['args'].update(count=2**31), 'overflow'),
    (lambda d: d['phases'][0]['steps'][3]['check']['expect'].update(eq={'ref':'traffic.started_at'}), 'type mismatch'),
    (lambda d: d['phases'][0]['steps'][3]['check']['expect'].update(eq={'ref':'removal.missing'}), 'Unknown result'),
    (lambda d: d['phases'][0]['steps'][3]['check']['expect'].update(eq={'ref':'drain.drained_at'}), 'earlier step'),
    (lambda d: d['phases'][0]['steps'][4]['check'].update(expect={'count':1}), 'collection'),
    (lambda d: d['phases'][0]['steps'][4]['check']['expect'].update(gte='3s'), 'lower bound'),
    (lambda d: d['phases'][0]['steps'][0]['call'].update(within='-1s'), 'out of range'),
])
def test_rejects_invalid_case_at_compile_time(case_data, compile_data, change, message):
    change(case_data)
    with pytest.raises(ConfigError, match=message):
        compile_data(case_data)


def test_rejects_second_traffic_even_after_stop_and_drain(case_data, compile_data):
    second = deepcopy(case_data['phases'][0]['steps'][0])
    second['call']['id'] = 'second_traffic'
    case_data['phases'][0]['steps'].append(second)
    with pytest.raises(ConfigError, match='Only one'):
        compile_data(case_data)


def test_all_requires_start_but_latest_and_interval_do_not(case_data, compile_data):
    case_data['phases'][0]['steps'] = [case_data['phases'][0]['steps'][4]]
    with pytest.raises(ConfigError, match='preceding traffic'):
        compile_data(case_data)
    for value in ('latest', ['-120s', '-60s']):
        case_data['phases'][0]['steps'][0]['check']['args']['range'] = value
        assert compile_data(case_data).phases[0].steps[0].args['range'] == value


@pytest.mark.parametrize('raw', [
    {'cache': 1, 'cache.sync.enabled': True},
    {'cache.sync.enabled': True, 'cache': 1},
    {'cache': {'sync': {}}, 'cache.sync.enabled': True},
    {'cache..enabled': True},
])
def test_rejects_config_path_collisions(raw):
    with pytest.raises(ConfigError):
        expand_config(raw)


def test_native_nested_payload_is_not_expanded():
    native = {'raw_json': {'original.key': 'untouched'}, 'cache.sync.enabled': True}
    result = expand_config(native)
    assert result == {'raw_json': {'original.key': 'untouched'}, 'cache': {'sync': {'enabled': True}}}
    assert native['cache.sync.enabled'] is True


def test_rejects_duplicate_keys_inside_master_json(case_data, compile_data):
    case_data['environment']['master']['config']['FLEXLB_CONFIG'] = '{"scheduler":{},"scheduler":{}}'
    with pytest.raises(ConfigError, match='Duplicate native JSON'):
        compile_data(case_data)


def test_java_extension_compiles_without_python_interface_changes(case_data, compile_data):
    registry = Registry.load(ROOT / 'java/interfaces/target/test-extension-catalog.json')
    case_data['phases'][0]['steps'] = [{'check': {
        'id':'extension', 'describe':'new Java method', 'interface':'extension/queue',
        'args':{'threshold':4}, 'expect':{'lte':10},
    }}]
    step = compile_data(case_data, catalog=registry).phases[0].steps[0]
    assert step.interface == 'extension/queue' and step.args == {'threshold':4}
    case_data['phases'][0]['steps'][0]['check']['args']['threshold'] = '4'
    with pytest.raises(ConfigError, match='integer'):
        compile_data(case_data, catalog=registry)


@pytest.mark.parametrize('output', [
    0, 128,
    {'distribution':'uniform', 'min':64, 'max':256, 'seed':42},
    {'distribution':'normal', 'mean':128, 'stddev':32, 'min':1, 'max':512, 'seed':42},
    {'distribution':'lognormal', 'mu':4.5, 'sigma':0.8, 'min':1, 'max':2048, 'seed':42},
    {'distribution':'choice', 'values':[64,128,512], 'seed':42},
    {'distribution':'choice', 'values':[64,128,512], 'weights':[0,2,1], 'seed':42},
])
def test_output_distributions_compile_from_java_schema(case_data, compile_data, output):
    prepare = case_data['phases'][0]['steps'][0]['call']['args']['input']['prepare']
    prepare['output_tokens'] = output
    result = compile_data(case_data).phases[0].steps[0].args['input']['prepare']
    assert result['output_tokens'] == output
    assert result['priority'] == 1


@pytest.mark.parametrize('output', [
    -1, 1.5, True, '128', 2**31,
    {'distribution':'uniform', 'min':256, 'max':64, 'seed':42},
    {'distribution':'uniform', 'min':1, 'max':64},
    {'distribution':'uniform', 'min':1.5, 'max':64, 'seed':42},
    {'distribution':'uniform', 'min':1, 'max':64, 'seed':2**63},
    {'distribution':'normal', 'mean':128, 'stddev':-1, 'min':1, 'max':512, 'seed':42},
    {'distribution':'normal', 'mean':float('nan'), 'stddev':1, 'min':1, 'max':512, 'seed':42},
    {'distribution':'normal', 'mean':128, 'stddev':1, 'seed':42},
    {'distribution':'normal', 'mean':128, 'stddev':1, 'min':1, 'max':512, 'seed':42, 'sigma':1},
    {'distribution':'lognormal', 'mu':float('inf'), 'sigma':1, 'min':1, 'max':512, 'seed':42},
    {'distribution':'lognormal', 'mu':4.5, 'sigma':-1, 'min':1, 'max':512, 'seed':42},
    {'distribution':'choice', 'values':[], 'seed':42},
    {'distribution':'choice', 'values':[64,-1], 'seed':42},
    {'distribution':'choice', 'values':[64,128], 'weights':[1], 'seed':42},
    {'distribution':'choice', 'values':[64,128], 'weights':[0,0], 'seed':42},
    {'distribution':'choice', 'values':[64,128], 'weights':[-1,2], 'seed':42},
    {'distribution':'choice', 'values':[64,128], 'weights':[float('inf'),2], 'seed':42},
    {'distribution':'unknown', 'seed':42},
])
def test_invalid_output_distributions_fail_before_execution(case_data, compile_data, output):
    case_data['phases'][0]['steps'][0]['call']['args']['input']['prepare']['output_tokens'] = output
    with pytest.raises(ConfigError, match='no matching parameter variant'):
        compile_data(case_data)
