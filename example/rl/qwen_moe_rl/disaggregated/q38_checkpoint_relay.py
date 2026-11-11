"""Bounded read-only source ranges from a mounted Qwen3.8 checkpoint.

The listener is scoped to a designated container IP, known training peers,
and an exact finite range plan. A receiver checks the declared range, the
length and its SHA256 before using the bytes. No whole-shard copy is needed.
"""

import argparse
import base64
from dataclasses import dataclass
import hashlib
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, HTTPServer
import ipaddress
import json
import os
from pathlib import Path
import re
import stat
import time


REVISION = '207bd685a7e3696cfaff12ded7c6a7ea0f88c996'
CONFIG_SHA256 = '4e3819548967e319ab435d044a3a331dbe3b078590ce822e9d74b79430533987'
INDEX_SHA256 = 'e36c40d4e99b2714fff821218a0433bda2dec46afdb1ebee8ce96ced997928ee'
MAX_RANGE_BYTES = 32 * 1024**2
MODEL = '/mnt/fuse/Qwen/Qwen3.8-2.4T-A95B'
METADATA = {'config.json', 'model.safetensors.index.json', 'tokenizer.json',
            'tokenizer_config.json', 'generation_config.json', 'chat_template.jinja',
            'merges.txt', 'vocab.json'}
SHARD = re.compile(r'model-[0-9]{5}-of-00213\.safetensors\Z')


@dataclass(frozen=True)
class Range:
    name: str
    size: int
    offset: int
    length: int

    def __post_init__(self):
        if type(self.name) is not str or (self.name not in METADATA and not SHARD.fullmatch(self.name)):
            raise ValueError('unexpected checkpoint file')
        if any(type(v) is not int for v in (self.size, self.offset, self.length)):
            raise ValueError('range coordinates must be integers')
        if not (0 <= self.offset < self.size and 0 < self.length <= MAX_RANGE_BYTES
                and self.offset + self.length <= self.size):
            raise ValueError('range is outside the object or the per-read budget')

    @property
    def path(self):
        return f'/{REVISION}/{self.name}'

    @property
    def request_header(self):
        return f'bytes={self.offset}-{self.offset+self.length-1}'

    @property
    def content_range(self):
        return f'bytes {self.offset}-{self.offset+self.length-1}/{self.size}'


def read_local(root, selected):
    with open(Path(root) / selected.name, 'rb', buffering=0, opener=_readonly) as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != selected.size:
            raise ValueError('checkpoint file size or type changed')
        source.seek(selected.offset)
        data = source.read(selected.length)
        after = os.fstat(source.fileno())
        if (len(data) != selected.length or
                (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns)):
            raise ValueError('checkpoint range was short or changed during the read')
        return data


def _readonly(path, flags):
    return os.open(path, flags | os.O_NOFOLLOW)


def verify_identity(root):
    for name, expected in (('config.json', CONFIG_SHA256), ('model.safetensors.index.json', INDEX_SHA256)):
        p = Path(root) / name
        selected = Range(name, p.stat().st_size, 0, p.stat().st_size)
        if hashlib.sha256(read_local(root, selected)).hexdigest() != expected:
            raise ValueError(f'pinned model {name} SHA256 mismatch')


def _private_ip(value):
    addr = ipaddress.IPv4Address(value)
    if not addr.is_private or addr.is_unspecified or addr.is_multicast or addr.is_loopback:
        raise ValueError('the relay must use an explicit private container IP')
    return str(addr)


def serve(root, host, port, plan, *, seconds=50):
    """Serve a finite plan; verify that every requested range was sent once."""
    host = _private_ip(host)
    pending = {}
    for peer, choices in plan.items():
        peer = _private_ip(peer)
        for r in choices:
            key = (peer, r.path, r.request_header)
            if key in pending:
                raise ValueError('duplicate peer/range in plan')
            pending[key] = r
    if not pending or not 1 <= seconds <= 3600:
        raise ValueError('an explicit nonempty plan and a finite deadline are required')
    receipts = []
    expires = time.monotonic() + seconds

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            self.request.settimeout(max(0.1, min(8, expires-time.monotonic())))
            super().setup()

        def log_message(self, *args):
            pass

        def do_GET(self):
            requested = self.headers.get_all('Range', [])
            if len(requested) != 1:
                self.send_error(416)
                return
            key = (self.client_address[0], self.path, requested[0])
            selected = pending.get(key)
            if selected is None or time.monotonic() >= expires:
                self.send_error(404)
                return
            try:
                data = read_local(root, selected)
            except (OSError, ValueError):
                self.send_error(409)
                return
            sha = hashlib.sha256(data).hexdigest()
            self.send_response(206)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Content-Range', selected.content_range)
            self.send_header('X-Checkpoint-SHA256', sha)
            self.end_headers()
            self.wfile.write(data)
            del pending[key]
            receipts.append(dict(peer=key[0], name=selected.name, offset=selected.offset,
                                 length=len(data), sha256=sha))

    with HTTPServer((host, port), Handler) as server:
        server.timeout = 0.25
        while pending and time.monotonic() < expires:
            server.handle_request()
    if pending:
        raise TimeoutError(f'{len(pending)} planned checkpoint ranges were not sent')
    return receipts


def fetch(host, port, selected, *, local_ip, timeout=15):
    connection = HTTPConnection(_private_ip(host), port, timeout=timeout,
                                source_address=(_private_ip(local_ip), 0))
    try:
        connection.request('GET', selected.path, headers={'Range': selected.request_header})
        response = connection.getresponse()
        if (response.status != 206 or response.getheader('Content-Range') != selected.content_range
                or response.getheader('Content-Length') != str(selected.length)):
            raise ValueError(f'unexpected checkpoint range response (HTTP {response.status})')
        data = response.read(selected.length)
        sha = hashlib.sha256(data).hexdigest()
        if len(data) != selected.length or sha != response.getheader('X-Checkpoint-SHA256'):
            raise ValueError('checkpoint range was truncated or failed integrity verification')
        return data, sha
    finally:
        connection.close()


def stage_metadata(root, files, official):
    """Create a metadata-only snapshot after checking the official object IDs."""
    if set(files) != METADATA or set(official) != METADATA:
        raise ValueError('exactly the tokenizer/config metadata may be staged')
    checks = {}
    for name, data in files.items():
        entry = official[name]
        if len(data) != entry['size']:
            raise ValueError(f'incorrect official metadata size: {name}')
        digest = hashlib.sha256(data).hexdigest()
        if entry.get('sha256'):
            valid = digest == entry['sha256']
        else:
            blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
            valid = blob == entry['blob_id']
        if not valid:
            raise ValueError(f'incorrect official metadata identity: {name}')
        checks[name] = {'size':len(data), 'sha256':digest}
    root = Path(root)
    if not root.is_absolute():
        raise ValueError('the destination must be a fresh absolute directory')
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    for name,data in files.items():
        with (root/name).open('xb') as out:
            out.write(data)
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != checks[name]['sha256']:
            raise ValueError(f'metadata write did not preserve verified bytes: {name}')
    verify_identity(root)
    with (root/'METADATA-READY.json').open('x') as out:
        json.dump({'revision':REVISION, 'weights_present':False, 'metadata':checks},out,indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('role', choices=('serve', 'fetch'))
    ap.add_argument('host')
    ap.add_argument('port', type=int)
    ap.add_argument('plan_b64')
    ap.add_argument('--local-ip')
    ap.add_argument('--root', default=MODEL)
    ap.add_argument('--seconds', type=int, default=50)
    ap.add_argument('--stage-metadata')
    ap.add_argument('--official-objects-b64')
    args = ap.parse_args()
    raw = json.loads(base64.b64decode(args.plan_b64, validate=True))
    plan = {peer: [Range(**item) for item in entries] for peer, entries in raw.items()}
    if args.role == 'serve':
        if args.stage_metadata or args.official_objects_b64:
            ap.error('metadata staging is receiver-only')
        verify_identity(args.root)
        receipts = serve(args.root, args.host, args.port, plan, seconds=args.seconds)
    else:
        receipts = []
        metadata = {}
        expires = time.monotonic() + args.seconds
        for selected in plan[args.local_ip]:
            while True:
                try:
                    data, sha = fetch(args.host, args.port, selected, local_ip=args.local_ip)
                    break
                except ConnectionRefusedError:
                    if time.monotonic() >= expires:
                        raise
                    time.sleep(0.5)
            if selected.name == 'config.json' and sha != CONFIG_SHA256:
                raise ValueError('received config is not the pinned model')
            if selected.name == 'model.safetensors.index.json' and sha != INDEX_SHA256:
                raise ValueError('received index is not the pinned model')
            receipts.append(dict(peer=args.local_ip, name=selected.name, offset=selected.offset,
                                 length=len(data), sha256=sha))
            if args.stage_metadata:
                if (selected.name not in METADATA or selected.name in metadata
                        or selected.offset != 0 or selected.length != selected.size):
                    raise ValueError('staging requires complete metadata objects only')
                metadata[selected.name] = data
            del data
        if args.stage_metadata:
            if not args.official_objects_b64:
                ap.error('official metadata object identities are required')
            objects = json.loads(base64.b64decode(args.official_objects_b64, validate=True))
            stage_metadata(args.stage_metadata, metadata, objects)
    print('Q38_CHECKPOINT_RELAY ' + json.dumps(receipts, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
