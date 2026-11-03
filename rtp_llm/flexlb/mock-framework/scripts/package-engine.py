#!/usr/bin/env python3
"""Compile one immutable Engine JAR and prepare the standard CI Docker context."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import socket
import tempfile
import time
import urllib.error
import urllib.request

FRAMEWORK = Path(__file__).resolve().parents[1]


def check_service(jar, backend):
    """Exercise the shaded gRPC runtime through the actual entry point, without remote services."""
    java = Path(os.environ['JAVA_HOME']) / 'bin/java' if os.environ.get('JAVA_HOME') else 'java'
    for _ in range(100):
        with socket.socket() as http, socket.socket() as rpc:
            http.bind(('127.0.0.1', 0))
            port = http.getsockname()[1]
            if port > 65530:
                continue
            try:
                rpc.bind(('127.0.0.1', port + 1))
            except OSError:
                continue
            break
    else:
        raise SystemExit('Cannot allocate Engine startup check ports')
    with tempfile.TemporaryDirectory(prefix='mock-package-check-') as directory:
        model = Path(directory)
        env = {key: os.environ[key] for key in ('PATH', 'HOME', 'LANG', 'TMPDIR') if key in os.environ}
        env.update(ROLE_TYPE='PREFILL',
                   START_PORT=str(port), MOCK_BIND_HOST='127.0.0.1', MOCK_ADVERTISE_HOST='127.0.0.1',
                   TEST_BLOCK_NUM='17', SEQ_SIZE_PER_BLOCK='128', MAX_SEQ_LEN='2048',
                   MOCK_COMPUTE_FORWARD_EXPRESSION='1', KV_CACHE_EVENT_PUBLISHER_TYPE='none',
                   ENABLE_REMOTE_CACHE='false')
        with (model / 'startup.log').open('w+') as log:
            process = subprocess.Popen([str(java), '-jar', str(jar)], env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                until = time.monotonic() + 15
                while process.poll() is None and time.monotonic() < until:
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as response:
                            log.flush()
                            records = [json.loads(line) for line in (model / 'startup.log').read_text().splitlines(keepends=True) if line.startswith('{') and line.endswith('\n')]
                            owned = any(record.get('event') == 'engine_started' and
                                        record.get('endpoints', {}).get('rpc') == f'127.0.0.1:{port + 1}' for record in records)
                            if response.status == 200 and owned and process.poll() is None:
                                try:
                                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=1) as metrics:
                                        if backend != 'prometheus' or metrics.status != 200:
                                            raise SystemExit('Packaged Engine enabled an unexpected metric backend')
                                except urllib.error.HTTPError as error:
                                    if backend != 'kmonitor' or error.code != 404:
                                        raise
                                print(f'Packaged Engine startup verified: metrics={backend}', flush=True)
                                return
                    except (urllib.error.URLError, TimeoutError):
                        time.sleep(0.1)
                log.seek(0)
                raise SystemExit('Packaged Engine failed startup:\n' + log.read()[-6000:])
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


def package():
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=FRAMEWORK, text=True).strip()
    expected = os.environ.get('EXPECTED_SOURCE_SHA')
    if expected is not None and source != expected:
        raise SystemExit('Engine source differs from the selected CI source SHA')
    recipe = os.environ.get('CI_RECIPE_SHA')
    if recipe is not None and (len(recipe) != 40 or any(c not in '0123456789abcdef' for c in recipe)):
        raise SystemExit('CI_RECIPE_SHA must be a full Git SHA')
    internal_monitoring = (FRAMEWORK.parents[2] / 'internal_source/java/kmonitor/pom.xml').is_file()
    profiles = []
    if internal_monitoring:
        profiles.append('internal-monitoring')
    build_args = ['-P' + ','.join(profiles)] if profiles else []
    subprocess.run([str(FRAMEWORK / 'scripts/build.sh'), '-pl', 'java/host', '-am', '-DskipTests', *build_args],
                   cwd=FRAMEWORK, check=True)
    jar = FRAMEWORK / 'java/host/target/mock-engine-service-1.0-SNAPSHOT-all.jar'
    with zipfile.ZipFile(jar) as archive:
        reporter = 'META-INF/services/org.flexlb.mock.metrics.MetricReporter'
        implementation = 'org.flexlb.mock.metrics.KMonitorReporter'
        if internal_monitoring:
            required = (reporter, implementation.replace('.', '/') + '.class', 'com/taobao/kmonitor/KMonitorFactory.class')
            if any(name not in archive.namelist() for name in required):
                raise SystemExit('Internal Engine package is missing KMonitor reporting')
            if archive.read(reporter).decode().strip() != implementation:
                raise SystemExit('Unexpected Engine metric reporter')
        elif reporter in archive.namelist():
            raise SystemExit('Public Engine package contains stale internal monitoring; clean the build targets')
    backend = 'kmonitor' if internal_monitoring else 'prometheus'
    check_service(jar, backend)
    # Recreate only the owned context; never include a checkout or local Case artifacts.
    context = FRAMEWORK / 'target/image'
    if context.exists():
        shutil.rmtree(context)
    context.mkdir(parents=True)
    shutil.copyfile(jar, context / 'mock-engine.jar')
    for name in ('Dockerfile', 'whale_start.sh'):
        shutil.copyfile(FRAMEWORK / 'docker' / name, context / name)
    manifest = dict(schema_version=1, source_sha=source, recipe_sha=recipe,
                    jar_sha256=hashlib.sha256(jar.read_bytes()).hexdigest(), tests_executed=False,
                    metrics_backend=backend, service_startup_verified=True)
    (context / 'build-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)
    print(f'Docker context: {context}', flush=True)


if __name__ == '__main__':
    package()
