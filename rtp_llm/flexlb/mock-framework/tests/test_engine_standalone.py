"""Whale starts the same packaged entry for one P/D/V engine, without a host manager."""
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.request import urlopen

import pytest


@pytest.mark.parametrize('role', ['prefill', 'decode', 'vit'])
def test_one_engine_per_process_uses_the_common_jar_without_a_manager(tmp_path, role):
    root = Path(__file__).resolve().parents[1]
    java = str(Path(os.environ['JAVA_HOME']) / 'bin/java')
    output = tmp_path / 'output'
    output.mkdir()
    config = tmp_path / 'config.json'
    values = {'compute': {'forward': {'expression': '1'}}}
    component = {'config': values}
    if role == 'vit':
        values['hidden_size'] = 4
    else:
        component['engine'] = 'rtp'
        values.update(seq_size_per_block=16, max_seq_len=4096, test_block_num=9, reserve_block_ratio=0)
    config.write_text(json.dumps({'id': role + '-0', 'component_role': role, 'component': component}))
    ready = output / 'ready.json'
    with socket.socket() as first, socket.socket() as second:
        first.bind(('127.0.0.1', 0))
        second.bind(('127.0.0.1', 0))
        rpc_port, control_port = first.getsockname()[1], second.getsockname()[1]
    env = {key: value for key, value in os.environ.items() if not key.startswith('MOCK_')}
    env.update(MOCK_CONFIG_FILE=str(config), MOCK_OUTPUT_DIR=str(output), MOCK_READY_FILE=str(ready),
        MOCK_RPC_PORT=str(rpc_port), MOCK_CONTROL_PORT=str(control_port),
        MOCK_BIND_HOST='0.0.0.0', MOCK_ADVERTISE_HOST='127.0.0.1')
    command = [java, '-XX:ActiveProcessorCount=2', '-jar',
        str(root / 'java/host/target/mock-engine-service-1.0-SNAPSHOT-all.jar')]
    with (output / 'process.log').open('w') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 15
            while not ready.exists():
                assert child.poll() is None, (output / 'process.log').read_text()
                assert time.monotonic() < deadline
                time.sleep(.02)
            endpoint = json.loads((output / 'endpoints.json').read_text())
            effective = json.loads((output / 'effective-config.json').read_text())
            assert effective['id'] == endpoint['id'] == role + '-0'
            assert endpoint['role'] == role
            assert endpoint['rpc'] == f'127.0.0.1:{rpc_port}'
            assert endpoint['control'] == f'http://127.0.0.1:{control_port}'
            with urlopen(endpoint['control'] + '/health', timeout=2) as response:
                assert json.load(response)['interfaces'] == []
            with urlopen(endpoint['metrics'], timeout=2) as response:
                assert response.status == 200
        finally:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        assert not ready.exists()
    # A replacement must use a fresh artifact directory, for every engine role.
    endpoints_before = (output / 'endpoints.json').read_bytes()
    retried = subprocess.run(command, env=env, capture_output=True, timeout=15)
    assert retried.returncode != 0
    assert (output / 'endpoints.json').read_bytes() == endpoints_before
    assert not ready.exists()
    for port in (rpc_port, control_port):
        with socket.socket() as probe:
            assert probe.connect_ex(('127.0.0.1', port)) != 0
