import asyncio
import json
import os
from pathlib import Path

import pytest
import yaml

from mock_framework.config import Registry, compile_case, load_yaml
from mock_framework.execution import ExecutionError, run_case
from mock_framework.execution.launch import local_launch
from mock_framework.execution.master import master_port
from mock_framework.input import read_inputs

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'cases/master_ha_failover/case.yaml'


def test_ha_case_keeps_one_playback_and_full_rolling_order_without_progress_or_sample_floor():
    raw = load_yaml(SOURCE)
    case = compile_case(SOURCE, Registry.standard())
    steps = [step for phase in case.phases for step in phase.steps]
    assert [step.id for step in steps if step.interface in ('master/stop', 'master/restart')] == [
        'kill_a', 'restore_a', 'kill_b', 'restore_b']
    assert sum(step.interface == 'client/start' for step in steps) == 1
    assert sum(step.interface == 'client/stop' for step in steps) == 1
    assert not any(step.interface in ('client/state/get', 'master/idle') for step in steps)
    rates = [step for step in steps if step.interface == 'client/success_rate']
    assert [step.id for step in rates] == ['baseline', 'b_success', 'a_success', 'both_success']
    assert all(step.args == {'range': ['-20s', '0s']} and step.expect.comparisons == {'gte': .95} for step in rates)
    assert 'min_samples' not in SOURCE.read_text()
    assert case.environment['client']['deployment']['replicas'] == 2
    import hashlib
    manifest = json.loads((SOURCE.parent / 'deepseek.manifest.json').read_text())
    assert manifest['requests'] == 844864 and manifest['source_span_ms'] == 959995
    with (SOURCE.parent / 'deepseek.binpb.zst').open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == manifest['sha256']
    source = ROOT.parent / 'tools/online_eval/data/performance'
    preset = json.loads((source / 'deepseek_v4_flash_l20c.json').read_text())
    calibration = json.loads((source / 'dsv4_l20_mock_calibration.json').read_text())
    p, d = case.environment['prefill'], case.environment['decode']
    assert (p['deployment']['replicas'], d['deployment']['replicas']) == (48, 192)
    assert (p['config']['seq_size_per_block'], d['config']['seq_size_per_block']) == (512, 128)
    assert (p['config']['test_block_num'], d['config']['test_block_num']) == (21554, 221485)
    assert p['config']['compute']['expression'] == calibration['prefill_expression']
    assert p['config']['compute']['noise']['scale'] == preset['prefill']['scale']
    assert d['config']['decode']['tokens_per_step'] == preset['decode']['tokens_per_step']
    assert p['config']['enable_memory_cache'] is True
    assert 'generation' not in p['config'] and 'generation' not in d['config']
    assert (p['config']['cp_size'], p['config']['cp_enabled'], p['config']['tp_size'], p['config']['dp_size']) == (4, True, 4, 1)
    assert (d['config']['cp_size'], d['config']['cp_enabled'], d['config']['tp_size'], d['config']['dp_size']) == (1, False, 1, 1)
    assert p['config']['max_generate_batch_size'] == 64 and d['config']['max_generate_batch_size'] == 128
    assert p['config']['max_batch_tokens_without_cache'] == 50000
    assert p['config']['max_inited_kv_cache_streams'] == 128 and p['config']['memory_cache_blocks'] == 52295
    assert 'load_cache_timeout_ms' not in p['config'] and d['config']['load_cache_timeout_ms'] == 5000
    for config in (p['config'], d['config']):
        assert not set(config) & {'max_context_batch_size', 'cp_force_single_prefill', 'decode_retry_times',
                                  'decode_retry_timeout_ms', 'decode_retry_interval_ms', 'block_tree_transfer_worker_count',
                                  'memory_cache_max_descriptors_per_transfer_batch', 'prefill_prepare_resource_pool_size',
                                  'prefill_stop_stream_wait_timeout_ms'}
    assert not set(d['config']) & {'memory_eviction_policy', 'memory_cache_sync_timeout_ms',
                                  'block_tree_memory_evict_low_watermark_ratio', 'block_tree_memory_evict_high_watermark_ratio',
                                  'prefill_retry_times', 'prefill_max_wait_timeout_ms', 'max_inited_kv_cache_streams'}
    traffic, = [step for step in steps if step.interface == 'client/start']
    assert traffic.args['replay_speed'] == 5 and 'qps' not in traffic.args
    assert traffic.args['concurrency'] == 4096
    from contextlib import closing
    from itertools import islice
    with closing(read_inputs(SOURCE.parent / 'deepseek.binpb.zst')) as inputs:
        sample = list(islice(inputs, 8))
    assert sample[0].arrival_offset_ms == 0
    assert all(row.HasField('arrival_offset_ms') for row in sample)
    assert any(len(row.token_ids) != 3 for row in sample)
    for phase in raw['phases']:
        for step in phase['steps']:
            if step['id'] in ('kill_a', 'kill_b', 'restore_a', 'restore_b'):
                assert step['wait_after'] == '30s'


def records(output):
    return [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]


@pytest.mark.skipif(not os.environ.get('JAVA_HOME') or not os.environ.get('PROMETHEUS_BIN'),
                    reason='Java artifacts, JAVA_HOME and PROMETHEUS_BIN required')
def test_native_master_restores_address_and_refreshes_only_its_current_discovery(tmp_path):
    from test_master_lifecycle import step, assert_reaped
    document = load_yaml(ROOT / 'cases/basic/case.yaml')
    document['environment'].pop('client')
    document['phases'] = [dict(id='lifecycle', describe='Native kill, restoration, and discovery refresh', timeout='120s', steps=[
        step('a', 'master/leadership/get', expect={'count': 1}),
        step('killed', 'master/stop', args={'targets': {'ref': 'steps.a'}, 'mode': 'kill'}),
        step('restored', 'master/restart', args={'targets': {'ref': 'steps.a'}}, within='60s', wait_after='2s'),
        step('leader', 'master/leadership/get', expect={'eq': {'ref': 'steps.a'}}),
        step('added', 'engine/scale_out', args={'role': 'prefill', 'count': 1}, expect={'count': 1})])]
    source = tmp_path / 'case.yaml'; source.write_text(yaml.safe_dump(document))
    registry = Registry.standard(); case = compile_case(source, registry)
    launch = local_launch(case, Path(os.environ['JAVA_HOME']) / 'bin/java', os.environ['PROMETHEUS_BIN'], framework=ROOT)
    output = tmp_path / 'run'
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result['status'] == 'PASS', result
    values = {item['step']: item['result'] for item in records(output) if item['event'] == 'step_finished'}
    assert values['a'] == values['restored']
    first = output / 'components/master-0'
    new = first / 'restart-1'
    original = json.loads((first / 'endpoints.json').read_text())
    restored = json.loads((new / 'endpoints.json').read_text())
    assert original['rpc'] == restored['rpc'] and original['http'] == restored['http']
    assert len(json.loads((first / 'discovery.json').read_text())['p-domain']) == 1
    assert len(json.loads((new / 'discovery.json').read_text())['p-domain']) == 2
    process = [item for item in records(output) if item['event'] == 'process_started' and item['node'].startswith('master-')]
    assert len(process) == 2 and process[0]['pid'] != process[1]['pid']
    assert_reaped(output)


@pytest.mark.skipif(not os.environ.get('JAVA_HOME') or not os.environ.get('PROMETHEUS_BIN'),
                    reason='Java artifacts, JAVA_HOME and PROMETHEUS_BIN required')
def test_shipped_ha_case_runs_with_independent_native_masters(tmp_path):
    try:
        master_port(2)
    except ExecutionError as error:
        if 'enable loopback aliases' in str(error):
            pytest.skip(str(error))
        raise
    registry = Registry.standard(); case = compile_case(SOURCE, registry)
    launch = local_launch(case, Path(os.environ['JAVA_HOME']) / 'bin/java', os.environ['PROMETHEUS_BIN'], framework=ROOT)
    output = tmp_path / 'run'
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result['status'] == 'PASS', result
    gates = [json.loads(line) for line in (output / 'gates.jsonl').read_text().splitlines()]
    assert all(gate['status'] == 'PASS' for gate in gates)
    assert {item['step'] for item in gates if item['interface'] == 'client/success_rate'} == {
        'baseline', 'b_success', 'a_success', 'both_success'}
    log = records(output)
    starts = [item for item in log if item['event'] == 'process_started' and item['node'].startswith('master-')]
    assert len(starts) == 4 and len({item['pid'] for item in starts}) == 4
    assert len([item for item in log if item['event'] == 'process_stopped' and item['code'] == -9]) == 2
    assert json.loads((output / 'topology.json').read_text()) == []
    from test_master_lifecycle import assert_reaped
    assert_reaped(output)


@pytest.mark.skipif(not os.environ.get('JAVA_HOME') or not os.environ.get('PROMETHEUS_BIN'),
                    reason='Java artifacts, JAVA_HOME and PROMETHEUS_BIN required')
def test_success_rate_through_native_clients_and_prometheus(tmp_path):
    document = load_yaml(ROOT / 'cases/basic/case.yaml')
    document['environment']['client']['deployment'] = {'replicas': 2}
    for phase in document['phases']:
        for step in phase['steps']:
            if step['interface'] == 'client/start':
                step['args']['requests_file'] = str(ROOT / 'cases/basic/requests.binpb.zst')
            if step['interface'] == 'client/qps':
                step.update(interface='client/success_rate', args={'range': ['-20s', '0s']}, expect={'eq': 1.0})
    source = tmp_path / 'case.yaml'; source.write_text(yaml.safe_dump(document))
    registry = Registry.standard(); case = compile_case(source, registry)
    launch = local_launch(case, Path(os.environ['JAVA_HOME']) / 'bin/java', os.environ['PROMETHEUS_BIN'], framework=ROOT)
    output = tmp_path / 'run'
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result['status'] == 'PASS', result
    gates = [json.loads(line) for line in (output / 'gates.jsonl').read_text().splitlines()]
    rate, = [gate for gate in gates if gate['interface'] == 'client/success_rate']
    assert rate['actual'] == 1.0
    from test_master_lifecycle import assert_reaped
    assert_reaped(output)
