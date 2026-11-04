from mock_framework.execution.test.helpers import FRAMEWORK
"""Exercise provider discovery and exclusive push collection in the packaged Engine."""
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import zipfile

import pytest

ROOT = FRAMEWORK
JAR = ROOT / 'java/engine/target/mock-engine-service-1.0-SNAPSHOT-all.jar'
JAVA = Path(os.environ['JAVA_HOME']) / 'bin/java' if os.environ.get('JAVA_HOME') else Path(shutil.which('java') or '/usr/bin/java')
JAVAC = JAVA.with_name('javac')


@pytest.fixture(scope='module')
def push_provider(tmp_path_factory):
    directory = tmp_path_factory.mktemp('engine-push-provider')
    with zipfile.ZipFile(JAR) as archive:
        if 'META-INF/services/org.flexlb.mock.metrics.MetricReporter' in archive.namelist():
            pytest.skip('This fixture verifies the public artifact with a synthetic provider, not an internal SDK image')
    source = directory / 'FixtureReporter.java'
    source.write_text('''
import org.flexlb.mock.metrics.MetricReporter;
import io.prometheus.client.Collector.MetricFamilySamples;
import java.util.*;
import java.nio.file.*;
public class FixtureReporter implements MetricReporter {
    private Path destination;
    public void initialize(Map<String,String> env) { destination = Path.of(env.get("REPORT_CAPTURE")); }
    public long intervalMillis() { return 50; }
    public void report(List<MetricFamilySamples> snapshot) {
        var text = new StringBuilder();
        for (var family : snapshot) for (var sample : family.samples)
            text.append(sample.name).append(sample.labelNames).append(sample.labelValues).append("\\n");
        try { Files.writeString(destination, text); }
        catch (java.io.IOException error) { throw new java.io.UncheckedIOException(error); }
    }
}
''')
    subprocess.run([str(JAVAC), '-cp', str(JAR), str(source)], check=True, capture_output=True, text=True)
    service = directory / 'META-INF/services/org.flexlb.mock.metrics.MetricReporter'
    service.parent.mkdir(parents=True)
    service.write_text('FixtureReporter\n')
    return directory


def listener_pair():
    for _ in range(20):
        with socket.socket() as http, socket.socket() as rpc:
            http.bind(('127.0.0.1', 0))
            port = http.getsockname()[1]
            if port >= 65535:
                continue
            try:
                rpc.bind(('127.0.0.1', port + 1))
            except OSError:
                continue
            return port
    raise AssertionError('Cannot allocate Engine listener pair')


@pytest.mark.parametrize('role', ['PREFILL', 'DECODE', 'VIT'])
def test_provider_disables_pull_and_receives_role_metrics(tmp_path, push_provider, role):
    port = listener_pair()
    capture = tmp_path / 'reported.txt'
    env = {key: os.environ[key] for key in ('PATH', 'HOME', 'LANG', 'TMPDIR') if key in os.environ}
    env.update(ROLE_TYPE=role, PATCH_SIZE='14', DOWNSAMPLE_RATIO='3', MIN_PIXELS='0', MAX_IMAGE_TOKENS='1024', HIDDEN_SIZE='4',
               START_PORT=str(port), MOCK_BIND_HOST='127.0.0.1', MOCK_ADVERTISE_HOST='127.0.0.1',
               TEST_BLOCK_NUM='17', SEQ_SIZE_PER_BLOCK='128', MAX_SEQ_LEN='2048',
               MOCK_COMPUTE_FORWARD_EXPRESSION='1', KV_CACHE_EVENT_PUBLISHER_TYPE='none',
               REPORT_CAPTURE=str(capture), kmonitorTags='test_scope^push-test')
    with (tmp_path / 'engine.log').open('w+') as log:
        process = subprocess.Popen([str(JAVA), '-cp', os.pathsep.join((str(push_provider), str(JAR))),
                                    'org.flexlb.mock.engine.EngineMain'], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=.5) as health:
                        if health.status == 200 and capture.exists() and 'push-test' in capture.read_text():
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(.05)
            else:
                log.seek(0)
                pytest.fail('Engine push reporter failed:\n' + log.read())
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=1)
            assert error.value.code == 404
            expected = 'py_rtp_vit_qps' if role == 'VIT' else 'rtp_llm_wait_stream_size'
            assert expected in capture.read_text()
            log.seek(0)
            assert '"metrics_backend":"kmonitor"' in log.read()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
