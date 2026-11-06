from mock_framework.execution.test.helpers import FRAMEWORK
from pathlib import Path

import pytest

from mock_framework.config import load_yaml
from mock_framework.execution.step.registry import Registry

ROOT = FRAMEWORK


@pytest.fixture
def registry():
    return Registry.standard()


@pytest.fixture
def case_data():
    return load_yaml(ROOT / 'python/mock_framework/config/test/fixtures/compile_case.yaml')


@pytest.fixture
def compile_data(tmp_path, registry):
    import yaml
    from mock_framework.config import compile_case
    def compile_(data, *, interfaces=None):
        path = tmp_path / 'case.yaml'
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding='utf-8')
        return compile_case(path, interfaces or registry, shared_root=tmp_path / 'shared')
    return compile_


@pytest.fixture
def native_kvcm(tmp_path):
    """Owned native-protocol test process; never contacts a real deployment."""
    import json
    import os
    import subprocess
    import time
    processes = []

    def start():
        ready = tmp_path / 'kvcm-ready.json'
        log = (tmp_path / 'kvcm.log').open('w')
        classpath = os.pathsep.join([str(ROOT / 'java/client/target/test-classes'),
            str(ROOT / 'java/engine/target/mock-engine-service-1.0-SNAPSHOT-all.jar')])
        process = subprocess.Popen([str(Path(os.environ['JAVA_HOME']) / 'bin/java'), '-cp', classpath,
            'org.flexlb.mock.client.KvcmMetaProbe', str(ready)], stdout=log, stderr=log)
        processes.append((process, log))
        deadline = time.monotonic() + 10
        while not ready.exists():
            if process.poll() is not None or time.monotonic() >= deadline:
                raise AssertionError((tmp_path / 'kvcm.log').read_text())
            time.sleep(.02)
        return json.loads(ready.read_text())

    yield start
    for process, log in processes:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        log.close()
