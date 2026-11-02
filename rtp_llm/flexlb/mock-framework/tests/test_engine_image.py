"""Validate the image context contract without requiring a container daemon."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def packager():
    spec = importlib.util.spec_from_file_location('package_engine', ROOT / 'scripts/package-engine.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packaging_rejects_a_different_source_before_compiling(monkeypatch):
    monkeypatch.setenv('EXPECTED_SOURCE_SHA', '0' * 40)
    with pytest.raises(SystemExit, match='selected CI source SHA'):
        packager().package()


def test_context_contains_only_service_payload_and_provenance(tmp_path, monkeypatch):
    module = packager()
    monkeypatch.setattr(module, 'FRAMEWORK', tmp_path)
    monkeypatch.delenv('EXPECTED_SOURCE_SHA', raising=False)
    monkeypatch.setenv('CI_RECIPE_SHA', 'a' * 40)
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *args, **kwargs: 'b' * 40 + '\n')
    calls = []
    monkeypatch.setattr(module.subprocess, 'run', lambda args, **kwargs: calls.append(args))
    jar = tmp_path / 'java/host/target/mock-engine-service-1.0-SNAPSHOT-all.jar'
    jar.parent.mkdir(parents=True)
    jar.write_bytes(b'test-artifact')
    (tmp_path / 'docker').mkdir()
    for name in ('Dockerfile', 'whale_start.sh'):
        (tmp_path / 'docker' / name).write_bytes((ROOT / 'docker' / name).read_bytes())
    stale = tmp_path / 'target/image/old-case'
    stale.mkdir(parents=True)
    module.package()
    context = tmp_path / 'target/image'
    assert {p.name for p in context.iterdir()} == {'mock-engine.jar', 'build-manifest.json', 'Dockerfile', 'whale_start.sh'}
    manifest = json.loads((context / 'build-manifest.json').read_text())
    assert manifest['source_sha'] == 'b' * 40 and manifest['recipe_sha'] == 'a' * 40
    assert manifest['tests_executed'] is False
    import hashlib
    assert manifest['jar_sha256'] == hashlib.sha256(b'test-artifact').hexdigest()
    assert calls[0][-3:] == ['java/host', '-am', '-DskipTests']


def test_image_entry_execs_java_with_standard_jvm_options(tmp_path):
    java = tmp_path / 'bin/java'
    java.parent.mkdir()
    java.write_text('#!/bin/sh\nprintf "%s\\n" "$$" "$MOCK_BIND_HOST" "$JAVA_TOOL_OPTIONS" "$@"\n')
    java.chmod(0o755)
    env = dict(os.environ, JAVA_HOME=str(tmp_path), JAVA_TOOL_OPTIONS='-Xmx512m')
    env.pop('MOCK_BIND_HOST', None)
    child = subprocess.Popen(['bash', str(ROOT / 'docker/whale_start.sh')], env=env, stdout=subprocess.PIPE, text=True)
    output, _ = child.communicate(timeout=5)
    assert child.returncode == 0
    assert output.splitlines() == [str(child.pid), '0.0.0.0', '-Xmx512m', '-jar', '/home/admin/mock-engine/mock-engine.jar']
    jar = ROOT / 'java/host/target/mock-engine-service-1.0-SNAPSHOT-all.jar'
    with zipfile.ZipFile(jar) as packaged:
        assert b'Main-Class: org.flexlb.mock.host.EngineMain' in packaged.read('META-INF/MANIFEST.MF')
