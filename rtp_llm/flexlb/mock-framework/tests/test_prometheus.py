import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import socket
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
def test_real_prometheus_ingestion_semantic_queries_and_recorded_gates(tmp_path):
    binary = Path(os.environ['PROMETHEUS_BIN']).resolve()
    assert binary.is_file()
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    targets = tmp_path / 'targets.json'
    targets.write_text('[]')
    config = tmp_path / 'prometheus.yaml'
    config.write_text(yaml.safe_dump({
        'global': {'scrape_interval': '100ms', 'scrape_timeout': '100ms'},
        'scrape_configs': [{'job_name': 'stage8', 'file_sd_configs': [
            {'files': [str(targets)], 'refresh_interval': '100ms'}]}],
    }))

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
        return {'check': {'id': id, 'describe': id, 'interface': interface, 'args': args, 'expect': expect}}

    data = {'id': 'prometheus_integration', 'describe': '真实节点、采集和查询门禁集成测试',
            'environment': {'prefill': {'engine': 'rtp', 'deployment': {'replicas': 2}, 'config': {
                'seq_size_per_block': 2, 'max_seq_len': 64, 'reuse_cache': True,
                'capacity.device_blocks': 16, 'capacity.reserve_blocks': 0,
                'prometheus.url': url, 'prometheus.job': 'stage8',
            }}, 'compute': {'prefill': {'expression': '2 * totalComputeTokens'}}},
            'phases': [{'id': 'observe', 'describe': '检查已采集的实际事件', 'timeout': '20s', 'steps': [
                {'call': {'id': 'collect', 'describe': '为测试原生客户端保留采集时间',
                          'interface': 'runtime/wait', 'args': {'duration': '8s'}}},
                gate('targets', 'prometheus/query', {'eq': 2}, promql='sum(up{job="stage8"})'),
                gate('hit_rate', 'engine/cache/hit_rate', {'gte': .75, 'lte': .77}, range=['-60s', '0s']),
                gate('p0_hit', 'engine/cache/hit_rate', {'gte': .39, 'lte': .41}, range=['-60s', '0s'],
                     engines={'instances': [{'id': 'prefill-0'}]}),
                gate('p1_hit', 'engine/cache/hit_rate', {'gte': .79, 'lte': .81}, range=['-60s', '0s'],
                     engines={'instances': [{'id': 'prefill-1'}]}),
                gate('compute_tps', 'engine/prefill/tps', {'gte': 1}, range=['-60s', '0s']),
                gate('with_cache_tps', 'engine/prefill/tps_with_cache', {'gte': {'ref': 'compute_tps.value'}}, range=['-60s', '0s']),
            ]}]}
    path = tmp_path / 'case.yaml'
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    registry = Registry.load(ROOT / 'build/interfaces.json')
    case = compile_case(path, registry)
    java = Path(os.environ.get('JAVA_HOME', '/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home')) / 'bin/java'
    jar = ROOT / 'java/engine/rtp/target/mock-rtp-1.0-SNAPSHOT-all.jar'
    classpath = (ROOT / 'java/engine/rtp/target/rtp-classpath.txt').read_text().strip()
    launch = prepare_launch({'components': {'prefill': {'command': [str(java), '-jar', str(jar)]}},
                             'bindings': {'engine/': 'prefill-0', 'prometheus/': 'prefill-0'}}, case, tmp_path)
    output = tmp_path / 'result'

    async def run():
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
            temporary = targets.with_suffix('.tmp')
            temporary.write_text(json.dumps([{'targets': [item['metrics'].removeprefix('http://').removesuffix('/metrics') for item in endpoints]}]))
            temporary.replace(targets)
            await asyncio.to_thread(wait_until, lambda: len(query('mock_engine_requests_started_total{job="stage8"}')) == 2)
            await asyncio.sleep(.5)  # Baseline scrapes before requests, so counter increases are observable.
            for n, (length, count) in enumerate(((5, 2), (9, 10))):
                completed = await asyncio.to_thread(subprocess.run,
                    [str(java), '-cp', classpath, 'org.flexlb.mock.engine.rtp.MetricsProbe', endpoints[n]['rpc'],
                     str(length), str(count), str(tmp_path / f'probe-{n}.json')], capture_output=True, text=True, timeout=10)
                assert completed.returncode == 0, completed.stderr
            await asyncio.to_thread(wait_until, lambda: (samples := query('sum(mock_engine_forward_tokens_total{job="stage8",kind="output"})'))
                                    and float(samples[0]['value'][1]) == 12)
            result = await task
            assert result['status'] == 'PASS', result
            assert result['cleanup_errors'] == []
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    with (tmp_path / 'prometheus.log').open('wb') as log:
        process = subprocess.Popen([str(binary), '--config.file=' + str(config), '--web.listen-address=127.0.0.1:' + str(port),
                                    '--storage.tsdb.path=' + str(tmp_path / 'prometheus-data'), '--storage.tsdb.retention.time=1h'],
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_until(lambda: urllib.request.urlopen(url + '/-/ready', timeout=1).close() is None)
            asyncio.run(run())
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    gates = [json.loads(line) for line in (output / 'gates.jsonl').read_text().splitlines()]
    assert len(gates) == 6 and all(gate['status'] == 'PASS' for gate in gates)
    assert datetime.fromisoformat(gates[0]['actual']['query_time'].replace('Z', '+00:00')) == datetime.fromisoformat(gates[0]['query_time'])
    assert all('observation' in gate for gate in gates)
    events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
    for event in events:
        if event['event'] == 'process_started':
            with pytest.raises(ProcessLookupError):
                os.kill(event['pid'], 0)
