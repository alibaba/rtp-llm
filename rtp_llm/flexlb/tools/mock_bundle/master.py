"""Resolve independent Mock/ViT Pods into the release's local discovery file.

Whale publishes one healthy control endpoint per Pod. Expand the frozen logical
P/D ports, rather than registering hundreds of artificial VIP instances. Only
Pod discovery differs; Master scheduling and worker RPCs remain the release code.
"""
import ipaddress
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import threading
import time
import urllib.parse
import urllib.request


def get(url):
    with urllib.request.urlopen(url, timeout=3) as response:
        data = response.read(1 << 20)
    return data


def vip_hosts(domain, env):
    servers = get(env['MOCK_JMENV_URL']).decode().splitlines()
    errors = []
    for server in servers:
        try:
            ipaddress.IPv4Address(server.strip())
            query = urllib.parse.urlencode({'dom':domain, 'qps':0, 'udpPort':0, 'encoding':'GBK'})
            response = json.loads(get('http://' + server.strip() + ':80/vipserver/api/srvIPXT?' + query))
            hosts = response['hosts']
            return sorted({(str(ipaddress.IPv4Address(h['ip'])), int(h['port']))
                           for h in hosts if h['valid']})
        except Exception as error:
            errors.append(type(error).__name__)
    raise RuntimeError('VIP discovery failed: ' + ','.join(errors))


def expand(mock_hosts, vit_hosts, env):
    p, d = int(env['MOCK_PREFILL_COUNT']), int(env['MOCK_DECODE_COUNT'])
    if p < 0 or d < 0 or p + d == 0:
        raise ValueError('invalid logical worker counts')
    result = {env['MOCK_PREFILL_DOMAIN']: [], env['MOCK_DECODE_DOMAIN']: [], env['MOCK_VIT_DOMAIN']: []}
    for ip, control_port in mock_hosts:
        if not 0 < control_port < control_port + p + d <= 65535:
            raise ValueError('invalid logical worker port range')
        # Mock HTTP control = first P gRPC - 1. Master discovery consumes HTTP
        # ports and applies its release +1 conversion to reach each gRPC server.
        result[env['MOCK_PREFILL_DOMAIN']].extend(f'{ip}:{control_port + i}' for i in range(p))
        result[env['MOCK_DECODE_DOMAIN']].extend(f'{ip}:{control_port + p + i}' for i in range(d))
    result[env['MOCK_VIT_DOMAIN']] = [f'{ip}:{port}' for ip, port in vit_hosts]
    return result


def atomic_write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True))
    temporary.replace(path)


def main():
    env = os.environ.copy()
    if env.get('RTP_LLM_MOCK_BUNDLE') != '1' or env.get('MOCK_BUNDLE_ROLE') != 'master':
        raise ValueError('release test Master requires explicit mock boundary')
    runtime = Path(env['MOCK_BUNDLE_RUN_DIR'])
    runtime.mkdir(parents=True, exist_ok=True)
    file = runtime / 'discovery.json'
    config = json.loads(env['MODEL_SERVICE_CONFIG'])
    config.pop('hosts', None)
    config['discovery_file'] = str(file)
    env['MODEL_SERVICE_CONFIG'] = json.dumps(config)
    def resolve():
        return expand(vip_hosts(env['MOCK_CONTROL_DOMAIN'], env), vip_hosts(env['MOCK_VIT_DOMAIN'], env), env)
    deadline = time.monotonic() + 180
    while True:
        try:
            snapshot = resolve()
            if not all(snapshot.values()):
                raise RuntimeError('waiting for Mock and ViT Pods')
            atomic_write(file, snapshot)
            break
        except Exception as error:
            if time.monotonic() >= deadline:
                raise RuntimeError('Mock discovery startup timed out') from error
            time.sleep(2)
    stop = threading.Event()
    def refresh():
        last_ok = time.monotonic()
        while not stop.wait(2):
            try:
                atomic_write(file, resolve())
                last_ok = time.monotonic()
            except Exception as error:
                print('Mock discovery refresh failed: ' + type(error).__name__, flush=True)
                if time.monotonic() - last_ok > 30:
                    atomic_write(file, {key: [] for key in snapshot})
    thread = threading.Thread(target=refresh, daemon=True)
    thread.start()
    java = str(Path(env.get('JAVA_HOME', '/opt/taobao/java')) / 'bin/java')
    jar = Path(env.get('MOCK_BUNDLE_JAR_DIR', str(Path(__file__).parent / 'jars'))) / 'master.jar'
    args = [java, *shlex.split(env.get('MOCK_MASTER_JAVA_OPTS', '-Xms2g -Xmx64g -XX:MaxDirectMemorySize=8g')),
            '-jar', str(jar), '--server.port=' + env.get('START_PORT', '7001'),
            '--management.server.port=' + str(int(env.get('START_PORT', '7001')) + 1)]
    child = subprocess.Popen(args, env=env)
    def terminate(signum, frame):
        child.send_signal(signum)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, terminate)
    try:
        raise SystemExit(child.wait())
    finally:
        stop.set()
        thread.join(timeout=4)


if __name__ == '__main__':
    main()
