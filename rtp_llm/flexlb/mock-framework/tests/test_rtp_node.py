import asyncio
import json
import os
from pathlib import Path
import subprocess
import urllib.request

from mock_framework.config import Registry, compile_case
from mock_framework.execution import prepare_launch, run_case

ROOT = Path(__file__).resolve().parents[1]


def test_runner_starts_isolated_packaged_rtp_nodes_and_native_client_reads_them(tmp_path):
    import yaml
    java = Path(os.environ.get('JAVA_HOME', '/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home')) / 'bin/java'
    jar = ROOT / 'java/engine/rtp/target/mock-rtp-1.0-SNAPSHOT-all.jar'
    classpath = (ROOT / 'java/engine/rtp/target/rtp-classpath.txt').read_text().strip()
    environment = {role: {'engine': 'rtp', 'config': {'SEQ_SIZE_PER_BLOCK': '16', 'MAX_SEQ_LEN': '4096',
                    'capacity.device_blocks': 8, 'capacity.reserve_blocks': 0}} for role in ('prefill', 'decode')}
    environment['compute'] = {role: {'expression': '1 + batchSize'} for role in ('prefill', 'decode')}
    data = {'id': 'native_node_probe', 'describe': '仅验证节点启动及原生状态协议', 'environment': environment,
            'phases': [{'id': 'observe', 'describe': '测试客户端在此期间读取原生状态', 'timeout': '10s',
                        'steps': [{'call': {'id': 'wait', 'describe': '保留节点供测试客户端读取', 'interface': 'runtime/wait',
                                            'args': {'duration': '5s'}}}]}]}
    path = tmp_path / 'case.yaml'; path.write_text(yaml.safe_dump(data, allow_unicode=True))
    registry = Registry.load(ROOT / 'build/interfaces.json')
    case = compile_case(path, registry)
    launch = prepare_launch({'components': {role: {'command': [str(java), '-jar', str(jar)], 'startup_timeout': '10s'}
                                            for role in ('prefill', 'decode')}, 'bindings': {}}, case, tmp_path)
    output = tmp_path / 'result'

    async def run():
        task = asyncio.create_task(run_case(case, registry, launch, output))
        try:
            for _ in range(2000):
                if all((output / f'components/{role}-0/ready.json').exists() for role in ('prefill', 'decode')):
                    break
                if task.done():
                    raise AssertionError(await task)
                await asyncio.sleep(.01)
            else:
                raise AssertionError('RTP nodes did not become ready')
            for role in ('prefill', 'decode'):
                directory = output / f'components/{role}-0'
                endpoints = json.loads((directory / 'endpoints.json').read_text())
                with urllib.request.urlopen(endpoints['control'] + '/health', timeout=2) as response:
                    assert json.load(response)['interfaces'] == []
                probe_result = tmp_path / f'{role}.json'
                command = [str(java), '-cp', classpath, 'org.flexlb.mock.engine.rtp.NativeProbe', endpoints['rpc'], str(probe_result)]
                if role == 'decode':
                    prefill = json.loads((output / 'components/prefill-0/endpoints.json').read_text())
                    command.append(prefill['rpc'])
                completed = await asyncio.to_thread(subprocess.run, command,
                    capture_output=True, text=True, timeout=10)
                assert completed.returncode == 0, completed.stderr
                actual = json.loads(probe_result.read_text())
                assert actual['health'] == 'OK' and actual['alive']
                assert actual['role'] == f'RoleType.{role.upper()}'
                assert actual['total_tokens'] == actual['available_tokens'] == 128
                assert actual['block_size'] == 16 and actual['cache_keys'] == 0
                if role == 'prefill':
                    assert actual['local_output_tokens'] == 1
                else:
                    assert actual['transfer_loaded'] is True
                    assert actual['prefill_available_while_pinned'] == 112
                    assert actual['prefill_available_after_load'] == 128
                    assert actual['decode_available_after_load_cancel'] == 128
                    assert actual['decode_reserved_tokens'] == 16
                    assert actual['decode_available_after_cancel'] == 128
                    assert endpoints['remote_generation']['stages'] == ['ALLOCATE', 'LOAD']
                    assert endpoints['remote_generation']['load'] is True
                with urllib.request.urlopen(endpoints['metrics'], timeout=2) as response:
                    assert response.status == 200
                    assert response.headers.get_content_type() == 'text/plain'
                    metrics = response.read().decode('utf-8')
                (directory / 'metrics.prom').write_text(metrics)
                samples = [line for line in metrics.splitlines() if line.startswith('mock_engine_forward_seconds_count{')]
                assert len(samples) == 1
                assert float(samples[0].rsplit(' ', 1)[1]) == (1 if role == 'prefill' else 0)
                assert f'role="{role}"' in samples[0]
                effective = json.loads((directory / 'effective-config.json').read_text())
                assert effective['compute'][role]['noise']['scale'] == 1
            assert (await task)['status'] == 'PASS'
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())
    events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
    started = [event for event in events if event['event'] == 'process_started']
    assert len(started) == 2
    for event in started:
        try:
            os.kill(event['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError(f'Leaked RTP node {event["pid"]}')
