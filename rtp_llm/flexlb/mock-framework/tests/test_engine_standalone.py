"""Single-engine service startup owns no Case files and exits with its worker."""
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]


def ports():
    while True:
        with socket.socket() as http, socket.socket() as rpc:
            http.bind(('127.0.0.1', 0))
            port = http.getsockname()[1]
            if port > 65534:
                continue
            try:
                rpc.bind(('127.0.0.1', port + 1))
            except OSError:
                continue
            return port


def wait_healthy(child, port, log):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        assert child.poll() is None, log.read_text()
        try:
            with urlopen(f'http://127.0.0.1:{port}/health', timeout=.5) as response:
                assert json.load(response)['status'] == 'ok'
                return
        except (URLError, TimeoutError):
            time.sleep(.02)
    raise AssertionError(log.read_text())


def command():
    return [str(Path(os.environ['JAVA_HOME']) / 'bin/java'), '-XX:ActiveProcessorCount=2',
            '-jar', str(ROOT / 'java/engine/target/mock-engine-service-1.0-SNAPSHOT-all.jar')]


def environment(role, port):
    env = {key: value for key, value in os.environ.items() if not key.startswith('MOCK_')}
    env.update(ROLE_TYPE=role.upper(), START_PORT=str(port), MOCK_COMPUTE_FORWARD_EXPRESSION='1',
               MOCK_BIND_HOST='0.0.0.0', MOCK_ADVERTISE_HOST='127.0.0.1')
    if role == 'vit':
        env.update(PATCH_SIZE='14', DOWNSAMPLE_RATIO='3', MIN_PIXELS='0', MAX_IMAGE_TOKENS='1024', HIDDEN_SIZE='4')
        env['MOCK_VIT_SKIP_DOWNLOAD'] = 'true'
    else:
        env.update(SEQ_SIZE_PER_BLOCK='16', MAX_SEQ_LEN='4096', TEST_BLOCK_NUM='9', RESERVE_BLOCK_RATIO='0')
    return env


@pytest.mark.parametrize('source', ['file', 'env'])
@pytest.mark.parametrize('role', ['prefill', 'decode', 'vit'])
def test_service_restarts_without_case_artifacts(tmp_path, role, source):
    port = ports()
    env = environment(role, port)
    if source == 'file':
        config = {'mock.compute.forward.expression': '1'}
        if role == 'vit':
            config.update({'patch_size': 14, 'downsample_ratio': 3, 'min_pixels': 0, 'max_image_tokens': 1024, 'hidden_size': 4})
            config['mock.vit.skip.download'] = True
        component = {'config': config}
        if role != 'vit':
            component['engine'] = 'rtp'
            config.update(seq_size_per_block=16, max_seq_len=4096, test_block_num=9, reserve_block_ratio=0)
        path = tmp_path / 'config.json'
        path.write_text(json.dumps(dict(id=role + '-0', component_role=role, component=component)))
        env['MOCK_CONFIG_FILE'] = str(path)
        env['MOCK_COMPUTE_FORWARD_EXPRESSION'] = 'invalid expression: file is the only source'
    # Old experiment files must neither prevent service startup nor be overwritten.
    sentinel = tmp_path / 'effective-config.json'
    sentinel.write_text('local experiment evidence')
    for attempt in range(2):
        log = tmp_path / f'process-{attempt}.log'
        with log.open('w') as stream:
            child = subprocess.Popen(command(), env=env, cwd=tmp_path, stdout=stream, stderr=stream)
            try:
                wait_healthy(child, port, log)
                started = next(json.loads(line) for line in log.read_text().splitlines() if line.startswith('{'))
                assert started['event'] == 'engine_started'
                assert started['endpoints']['role'] == role
                assert started['endpoints']['rpc'] == f'127.0.0.1:{port + 1}'
                assert started['effective_config']['mock.compute.forward.expression'] == '1'
                with urlopen(f'http://127.0.0.1:{port}/metrics', timeout=2) as response:
                    assert response.status == 200
            finally:
                child.terminate()
                try:
                    child.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
                    raise
            assert child.returncode in (0, 143)
        assert sentinel.read_text() == 'local experiment evidence'
        assert not (tmp_path / 'ready.json').exists()
        assert not (tmp_path / 'endpoints.json').exists()
        for value in (port, port + 1):
            with socket.socket() as probe:
                assert probe.connect_ex(('127.0.0.1', value)) != 0


def test_fatal_vit_compute_failure_exits_the_service(tmp_path):
    port = ports()
    env = environment('vit', port)
    env['MOCK_COMPUTE_FORWARD_EXPRESSION'] = '-1'
    log = tmp_path / 'process.log'
    with log.open('w') as stream:
        child = subprocess.Popen(command(), env=env, cwd=tmp_path, stdout=stream, stderr=stream)
        try:
            wait_healthy(child, port, log)
            classpath = os.pathsep.join([str(ROOT / 'java/engine/target/test-classes'),
                                        str(ROOT / 'java/engine/target/mock-engine-service-1.0-SNAPSHOT-all.jar')])
            subprocess.run([command()[0], '-cp', classpath, 'org.flexlb.mock.engine.EngineFailureProbe',
                            f'127.0.0.1:{port + 1}'], check=True, capture_output=True, timeout=10)
            child.wait(timeout=8)
            assert child.returncode not in (0, 143)
            assert 'V forward failed' in log.read_text()
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
