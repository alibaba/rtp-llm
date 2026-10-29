import asyncio
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import time
import urllib.parse
import urllib.request

import pytest
import yaml

from mock_framework.config import Registry, compile_case
from mock_framework.execution import prepare_launch, run_case

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not os.environ.get('PROMETHEUS_BIN'), reason='Set PROMETHEUS_BIN for real Prometheus integration')
@pytest.mark.parametrize('mode', ['pass', 'gate_fail', 'cancel', 'collector_crash'])
def test_real_prometheus_ingestion_semantic_queries_and_recorded_gates(tmp_path, mode):
    binary = Path(os.environ['PROMETHEUS_BIN']).resolve()
    assert binary.is_file()
    url = None

    def query(expression):
        with urllib.request.urlopen(url + '/api/v1/query?' + urllib.parse.urlencode({'query': expression}), timeout=2) as response:
            return json.load(response)['data']['result']

    def wait_until(condition, seconds=10):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                if condition():
                    return
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(.05)
        raise AssertionError('Prometheus did not become ready or ingest the expected samples')

    def gate(id, interface, expect, **args):
        return {'id': id, 'describe': id, 'interface': interface, 'args': args, 'expect': expect}

    data = {'id': 'prometheus_integration', 'describe': '真实节点、采集和查询门禁集成测试',
            'environment': {'prefill': {'engine': 'rtp', 'deployment': {'replicas': 2}, 'config': {
                'seq_size_per_block': 2, 'max_seq_len': 64, 'reuse_cache': True,
                'test_block_num': 17, 'reserve_block_ratio': 0, 'compute.forward.expression': '2 * totalComputeTokens',
            }}},
            'phases': [{'id': 'observe', 'describe': '检查已采集的实际事件', 'timeout': '20s', 'steps': [
                {'id': 'targets', 'describe': '记录原始PromQL响应', 'interface': 'prometheus/query',
                 'args': {'promql': 'sum(up{job="mock"})'}, 'wait_after': '8s'},
                gate('hit_rate', 'engine/cache/hit_rate', {'gte': .75, 'lte': .77}, range=['-60s', '0s']),
                gate('compute_tps', 'engine/prefill/tps', {'gte': 1}, range=['-60s', '0s']),
                gate('with_cache_tps', 'engine/prefill/tps_with_cache', {'gte': {'ref': 'steps.compute_tps'}}, range=['-60s', '0s']),
            ]}]}
    if mode == 'gate_fail':
        data['phases'][0]['steps'][1]['expect'] = {'eq': 0}
    path = tmp_path / 'case.yaml'
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    registry = Registry.standard()
    case = compile_case(path, registry)
    java = Path(os.environ.get('JAVA_HOME', '/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home')) / 'bin/java'
    jar = ROOT / 'java/engine/rtp/target/mock-rtp-1.0-SNAPSHOT-all.jar'
    classpath = (ROOT / 'java/engine/rtp/target/rtp-classpath.txt').read_text().strip()
    launch = prepare_launch({'components': {'prefill': {'command': [str(java), '-jar', str(jar)]}},
                             'prometheus': {'binary': str(binary), 'scrape_interval': '100ms'}}, case, tmp_path)
    output = tmp_path / 'result'

    async def run():
        nonlocal url
        task = asyncio.create_task(run_case(case, registry, launch, output))
        try:
            for _ in range(2000):
                if all((output / f'components/prefill-{n}/ready.json').is_file() for n in range(2)):
                    break
                if task.done():
                    raise AssertionError(await task)
                await asyncio.sleep(.01)
            else:
                raise AssertionError('Nodes did not start')
            endpoints = [json.loads((output / f'components/prefill-{n}/endpoints.json').read_text()) for n in range(2)]
            for _ in range(1000):
                metadata = json.loads((output / 'prometheus/collection.json').read_text())
                if 'baseline_at' in metadata:
                    break
                if task.done():
                    raise AssertionError(await task)
                await asyncio.sleep(.02)
            else:
                raise AssertionError('Runner did not obtain a baseline')
            url = metadata['url']
            if mode in ('cancel', 'collector_crash'):
                if mode == 'cancel':
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                else:
                    events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
                    pid = next(event['pid'] for event in events if event['event'] == 'process_started' and event['node'] == 'prometheus')
                    os.kill(pid, 9)
                    assert (await task)['status'] == 'ERROR'
                return
            for n, (length, count) in enumerate(((5, 2), (9, 10))):
                completed = await asyncio.to_thread(subprocess.run,
                    [str(java), '-cp', classpath, 'org.flexlb.mock.engine.rtp.MetricsProbe', endpoints[n]['rpc'],
                     str(length), str(count), str(tmp_path / f'probe-{n}.json')], capture_output=True, text=True, timeout=10)
                assert completed.returncode == 0, completed.stderr
            await asyncio.to_thread(wait_until, lambda: (samples := query('sum(mock_engine_forward_tokens_total{job="mock",kind="output"})'))
                                    and float(samples[0]['value'][1]) == 12)
            result = await task
            assert result['status'] == ('FAIL' if mode == 'gate_fail' else 'PASS'), result
            assert result['cleanup_errors'] == []
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
    gates = [json.loads(line) for line in (output / 'gates.jsonl').read_text().splitlines()]
    if mode == 'pass':
        assert len(gates) == 3 and all(gate['status'] == 'PASS' for gate in gates)
        events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
        raw = next(e['result'] for e in events if e.get('step') == 'targets' and e['event'] == 'step_finished')
        assert raw['samples'][0]['value'] == 2
    elif mode == 'gate_fail':
        assert len(gates) == 1 and gates[0]['status'] == 'FAIL'
    else:
        assert gates == []
    metadata = json.loads((output / 'prometheus/collection.json').read_text())
    assert metadata['complete'] is (mode != 'collector_crash')
    assert list((output / 'prometheus/data/wal').glob('*'))
    if mode != 'collector_crash':
        assert set(metadata['final_scrapes']) == {'prefill-0', 'prefill-1'}
    tool = binary.with_name('promtool')
    if mode == 'pass' and tool.is_file():
        dumped = subprocess.run([str(tool), 'tsdb', 'dump',
            '--match={__name__="mock_engine_forward_tokens_total",kind="output"}', str(output / 'prometheus/data')],
            capture_output=True, text=True, timeout=15)
        assert dumped.returncode == 0, dumped.stderr
        (output / 'prometheus/offline-samples.txt').write_text(dumped.stdout)
        for node, expected in [('prefill-0', 2), ('prefill-1', 10)]:
            # Prometheus can append a stale marker after the owned node stops.
            address = json.loads((output / f'components/{node}/endpoints.json').read_text())['rpc']
            samples = [float(line.rsplit(' ', 2)[1]) for line in dumped.stdout.splitlines() if f'instance="{address}"' in line]
            samples = [value for value in samples if math.isfinite(value)]
            assert samples and samples[-1] == expected
    assert all('query_time' in gate for gate in gates)
    events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
    for event in events:
        if event['event'] == 'process_started':
            with pytest.raises(ProcessLookupError):
                os.kill(event['pid'], 0)
