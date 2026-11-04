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


@pytest.mark.parametrize('internal', [False, True])
def test_context_contains_only_service_payload_and_provenance(tmp_path, monkeypatch, internal):
    module = packager()
    framework = tmp_path / 'repo/rtp_llm/flexlb/mock-framework'
    monkeypatch.setattr(module, 'FRAMEWORK', framework)
    monkeypatch.delenv('EXPECTED_SOURCE_SHA', raising=False)
    monkeypatch.setenv('CI_RECIPE_SHA', 'a' * 40)
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *args, **kwargs: 'b' * 40 + '\n')
    calls = []
    monkeypatch.setattr(module.subprocess, 'run', lambda args, **kwargs: calls.append(args))
    jar = framework / 'java/engine/target/mock-engine-service-1.0-SNAPSHOT-all.jar'
    jar.parent.mkdir(parents=True)
    with zipfile.ZipFile(jar, 'w') as archive:
        archive.writestr('META-INF/MANIFEST.MF', 'Manifest-Version: 1.0\n')
        if internal:
            archive.writestr('META-INF/services/org.flexlb.mock.metrics.MetricReporter', 'org.flexlb.mock.metrics.KMonitorReporter\n')
            archive.writestr('org/flexlb/mock/metrics/KMonitorReporter.class', b'fixture')
            archive.writestr('com/taobao/kmonitor/KMonitorFactory.class', b'fixture')
    if internal:
        marker = framework.parents[2] / 'internal_source/java/kmonitor/pom.xml'
        marker.parent.mkdir(parents=True)
        marker.write_text('<project/>')
    verified = []
    monkeypatch.setattr(module, 'check_service', lambda jar, backend: verified.append(backend))
    (framework / 'docker').mkdir()
    for name in ('Dockerfile', 'whale_start.sh'):
        (framework / 'docker' / name).write_bytes((ROOT / 'docker' / name).read_bytes())
    stale = framework / 'target/image/old-case'
    stale.mkdir(parents=True)
    module.package()
    context = framework / 'target/image'
    assert {p.name for p in context.iterdir()} == {'mock-engine.jar', 'build-manifest.json', 'Dockerfile', 'whale_start.sh'}
    manifest = json.loads((context / 'build-manifest.json').read_text())
    assert manifest['source_sha'] == 'b' * 40 and manifest['recipe_sha'] == 'a' * 40
    assert manifest['tests_executed'] is False
    import hashlib
    assert manifest['jar_sha256'] == hashlib.sha256(jar.read_bytes()).hexdigest()
    assert manifest['metrics_backend'] == ('kmonitor' if internal else 'prometheus')
    assert verified == [manifest['metrics_backend']]
    if internal:
        assert calls[0][-1] == '-Pinternal-monitoring'
    else:
        assert calls[0][-3:] == ['java/engine', '-am', '-DskipTests']


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
    jar = ROOT / 'java/engine/target/mock-engine-service-1.0-SNAPSHOT-all.jar'
    with zipfile.ZipFile(jar) as packaged:
        assert b'Main-Class: org.flexlb.mock.engine.EngineMain' in packaged.read('META-INF/MANIFEST.MF')
