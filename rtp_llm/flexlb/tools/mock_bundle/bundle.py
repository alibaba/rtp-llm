"""Frozen release launcher: one independent P/D Mock JVM, never a Master."""
import ipaddress
import json
import os
from pathlib import Path
import shlex
import socket


def command(env):
    if env.get('RTP_LLM_MOCK_BUNDLE') != '1' or env.get('MOCK_BUNDLE_ROLE') != 'mock':
        raise ValueError('release launcher requires RTP_LLM_MOCK_BUNDLE=1 MOCK_BUNDLE_ROLE=mock')
    def number(key, default, minimum=1):
        value = int(env.get(key, default))
        if value < minimum:
            raise ValueError(key + ' is out of range')
        return value
    counts = [number('MOCK_PREFILL_COUNT', 40, 0), number('MOCK_DECODE_COUNT', 160, 0)]
    port = number('MOCK_GRPC_BASE_PORT', 29130)
    if sum(counts) == 0 or port < 2 or port + sum(counts) - 1 > 65535:
        raise ValueError('invalid Mock count or port range')
    host = env.get('POD_IP') or socket.gethostbyname(socket.gethostname())
    address = ipaddress.ip_address(host)
    if address.version != 4 or address.is_loopback or address.is_unspecified:
        raise ValueError('POD_IP must be an advertised IPv4 address')
    root = Path(env.get('MOCK_BUNDLE_RUN_DIR', '/home/admin/ai-whale/mock'))
    root.mkdir(parents=True, exist_ok=True)
    master = json.loads(env['FLEXLB_CONFIG'])
    performance = json.loads(env.get('MOCK_PERFORMANCE_CONFIG_JSON', '{}'))
    if not isinstance(master, dict) or not isinstance(performance, dict):
        raise ValueError('Master and performance config must be JSON objects')
    for name, value in [('master-config.json', master), ('performance.json', performance)]:
        (root / name).write_text(json.dumps(value))
    args = [str(Path(env.get('JAVA_HOME', '/opt/taobao/java')) / 'bin/java'),
            *shlex.split(env.get('MOCK_JAVA_OPTS', '-Xms512m -Xmx4g')),
            '-jar', str(Path(env.get('MOCK_BUNDLE_JAR_DIR', str(Path(__file__).parent / 'jars'))) / 'mock.jar'),
            '--host', host, '--unique-engine-ips', 'false',
            '--n-prefill', str(counts[0]), '--n-decode', str(counts[1]),
            '--base-grpc-port', str(port)]
    # Old CLI uses cache-blocks for the actual pool capacity, not a key count.
    for flag, key, default in [
        ('prefill-block-size', 'MOCK_PREFILL_BLOCK_SIZE', 512),
        ('decode-block-size', 'MOCK_DECODE_BLOCK_SIZE', 128),
        ('prefill-cache-blocks', 'MOCK_PREFILL_KV_POOL_BLOCKS', 45083),
        ('decode-cache-blocks', 'MOCK_DECODE_KV_POOL_BLOCKS', 932802),
        ('decode-max-concurrency', 'MOCK_DECODE_MAX_CONCURRENCY', 128),
        ('event-loop-threads', 'MOCK_EVENT_LOOP_THREADS', 16),
        ('completion-threads', 'MOCK_COMPLETION_THREADS', 16)]:
        args.extend(['--' + flag, str(number(key, default))])
    for flag, name in [('endpoint-file', 'endpoints.json'), ('discovery-file', 'discovery.json'),
                       ('performance', 'performance.json'), ('master-config', 'master-config.json')]:
        args.extend(['--' + flag, str(root / name)])
    return args


if __name__ == '__main__':
    args = command(os.environ)
    os.execv(args[0], args)
