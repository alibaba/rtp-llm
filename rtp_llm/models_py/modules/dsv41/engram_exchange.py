# SPDX-License-Identifier: Apache-2.0
"""One-shot static-cohort metadata exchange. No weights, CUDA or inference RPC.

One authenticated persistent connection per remote instance representative.
All payloads are bounded JSON. No pickle, discovery, election, or live reshard.
"""
from __future__ import annotations
import hashlib
import http.client
import json
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import UUID

MAX_MESSAGE = 4 << 20


class ExchangeError(RuntimeError):
    pass


def encode(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(data) > MAX_MESSAGE:
        raise ExchangeError('MESSAGE_TOO_LARGE')
    return data


def decode(data):
    from .engram_bootstrap import _json
    if len(data) > MAX_MESSAGE:
        raise ExchangeError('MESSAGE_TOO_LARGE')
    value = _json(data)
    if type(value) is not dict:
        raise ExchangeError('INVALID_MESSAGE')
    return value


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_group(group):
    """Public immutable startup contract; no credentials or local filesystem paths."""
    from .engram_shard_config import validate_count
    if type(group) is not dict or set(group) != {'generation', 'model_revision', 'members'}:
        raise ExchangeError('INVALID_GROUP')
    try:
        if str(UUID(group['generation'])) != group['generation']:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise ExchangeError('INVALID_GENERATION') from None
    if type(group['model_revision']) is not str or not 0 < len(group['model_revision']) <= 256:
        raise ExchangeError('INVALID_REVISION')
    members = group['members']
    if type(members) is not list or not 1 <= len(members) <= 64:
        raise ExchangeError('INVALID_MEMBERS')
    ids, hosts, indices = set(), set(), []
    for member in members:
        if type(member) is not dict or set(member) != {'instance_id', 'world_size', 'shards'}:
            raise ExchangeError('INVALID_MEMBER')
        ident = member['instance_id']
        if type(ident) is not str or not 0 < len(ident) <= 128 or ident in ids:
            raise ExchangeError('INVALID_INSTANCE_ID')
        ids.add(ident)
        if type(member['world_size']) is not int or not 1 <= member['world_size'] <= 4096:
            raise ExchangeError('INVALID_WORLD_SIZE')
        if type(member['shards']) is not list or not member['shards']:
            raise ExchangeError('EVERY_INSTANCE_MUST_HOLD_SHARDS')
        publishers = set()
        for shard in member['shards']:
            if type(shard) is not dict or set(shard) != {'index', 'host_id', 'publisher_rank'}:
                raise ExchangeError('INVALID_SHARD')
            if type(shard['index']) is not int or shard['index'] < 0:
                raise ExchangeError('INVALID_SHARD_INDEX')
            if type(shard['host_id']) is not str or not 0 < len(shard['host_id']) <= 256 or shard['host_id'] in hosts:
                raise ExchangeError('DUPLICATE_OR_INVALID_HOST')
            rank = shard['publisher_rank']
            if type(rank) is not int or not 0 <= rank < member['world_size'] or rank in publishers:
                raise ExchangeError('INVALID_PUBLISHER_RANK')
            hosts.add(shard['host_id']); publishers.add(rank); indices.append(shard['index'])
    validate_count(len(indices))
    if sorted(indices) != list(range(len(indices))):
        raise ExchangeError('INCOMPLETE_SHARD_SET')
    return digest(encode(group))


def validate_submission(group, instance_id, documents):
    members = {m['instance_id']: m for m in group['members']}
    if instance_id not in members or type(documents) is not list:
        raise ExchangeError('UNKNOWN_INSTANCE')
    expected = {s['index']: s['host_id'] for s in members[instance_id]['shards']}
    if len(documents) != len(expected):
        raise ExchangeError('INCOMPLETE_INSTANCE_SHARDS')
    seen = set()
    for doc in documents:
        if type(doc) is not dict or type(doc.get('shard')) is not int:
            raise ExchangeError('INVALID_DESCRIPTOR')
        index = doc['shard']
        if (index in seen or index not in expected or doc.get('host_id') != expected[index]
                or doc.get('generation') != group['generation']
                or doc.get('model_revision') != group['model_revision']):
            raise ExchangeError('DESCRIPTOR_IDENTITY_MISMATCH')
        seen.add(index)


class Cohort:
    """Static startup barrier; never retire physical backing on timeout/disconnect."""
    def __init__(self, group, build, timeout):
        self.group = decode(encode(group))
        self.group_digest = validate_group(group)
        self.members = {m['instance_id'] for m in group['members']}
        self.build = build
        self.deadline = time.monotonic() + timeout
        self.condition = threading.Condition()
        self.documents = {}
        self.payload = None
        self.error = None
        self.acks = set()
        self.building = False

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ExchangeError('COHORT_TIMEOUT')
        return remaining

    def submit(self, instance_id, group_digest, documents):
        self.remaining()
        if group_digest != self.group_digest:
            raise ExchangeError('GROUP_MISMATCH')
        validate_submission(self.group, instance_id, documents)
        encoded = encode({'documents': documents})
        with self.condition:
            previous = self.documents.get(instance_id)
            if previous is not None and previous != encoded:
                raise ExchangeError('CONFLICTING_REPUBLICATION')
            self.documents[instance_id] = encoded
            if len(self.documents) == len(self.members) and not self.building:
                self.building = True
                all_docs = [d for ident in sorted(self.documents)
                            for d in decode(self.documents[ident])['documents']]
                def assemble():
                    try:
                        result = self.build(all_docs)
                        payload = encode({'group_digest': self.group_digest, 'package': result})
                    except Exception:
                        with self.condition:
                            self.error = 'ASSEMBLY_FAILED'
                            self.condition.notify_all()
                    else:
                        with self.condition:
                            self.payload = payload
                            self.condition.notify_all()
                threading.Thread(target=assemble, daemon=True).start()
            while self.payload is None and self.error is None:
                self.condition.wait(self.remaining())
            if self.error:
                raise ExchangeError(self.error)
            return self.payload

    def ack(self, instance_id, payload_digest):
        with self.condition:
            self.remaining()
            if instance_id not in self.documents or self.payload is None or digest(self.payload) != payload_digest:
                raise ExchangeError('ACK_MISMATCH')
            self.acks.add(instance_id)
            self.condition.notify_all()

    def wait_acknowledged(self):
        with self.condition:
            while self.acks != self.members:
                self.condition.wait(self.remaining())


class ExchangeServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True

    def __init__(self, address, cohort, context, fingerprints):
        if context is not None:
            if not isinstance(context, ssl.SSLContext) or context.verify_mode != ssl.CERT_REQUIRED:
                raise ExchangeError('MUTUAL_TLS_REQUIRED')
            if not fingerprints:
                raise ExchangeError('MUTUAL_TLS_REQUIRED')
        self.cohort, self.context = cohort, context
        self.fingerprints = fingerprints or {}
        self.connections = 0
        self._slots = threading.BoundedSemaphore(len(cohort.members) + 2)
        super().__init__(address, ExchangeHandler)

    def get_request(self):
        raw, address = super().get_request()
        try:
            raw.settimeout(min(5, self.cohort.remaining()))
            if self.context is None:
                self.connections += 1
                return raw, address
            secure = self.context.wrap_socket(raw, server_side=True)
            secure.settimeout(self.cohort.remaining())
            self.connections += 1
            return secure, address
        except Exception as error:
            raw.close()
            # HTTPServer handles OSError from accept/handshake without leaking a
            # traceback or leaving the accepted descriptor open at the deadline.
            raise OSError('EXCHANGE_ACCEPT_FAILED') from error

    def process_request(self, request, address):
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        super().process_request(request, address)

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self._slots.release()

    def handle_error(self, request, client_address):
        # Never log opaque capabilities or TLS peer details via tracebacks.
        pass


class ExchangeHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *args):
        pass

    def respond(self, status, payload):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        if self.close_connection:
            self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def do_POST(self):
        try:
            lengths = self.headers.get_all('Content-Length', [])
            if len(lengths) != 1 or self.headers.get('Transfer-Encoding'):
                raise ExchangeError('INVALID_FRAME')
            length = int(lengths[0])
            if not 0 < length <= MAX_MESSAGE:
                raise ExchangeError('INVALID_FRAME')
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ExchangeError('TRUNCATED_FRAME')
            request = decode(raw)
            ident = request.get('instance_id')
            # Plain-HTTP mode has no peer certificate; the group digest still
            # binds membership, but identity is self-asserted on a trusted link.
            if self.server.context is not None:
                cert_digest = digest(self.connection.getpeercert(binary_form=True))
                if self.server.fingerprints.get(ident) != cert_digest:
                    raise ExchangeError('UNAUTHORIZED_INSTANCE')
            if self.path == '/v1/exchange' and set(request) == {'instance_id', 'group_digest', 'documents'}:
                payload = self.server.cohort.submit(ident, request['group_digest'], request['documents'])
                self.respond(200, payload)
            elif self.path == '/v1/ack' and set(request) == {'instance_id', 'payload_digest'}:
                self.server.cohort.ack(ident, request['payload_digest'])
                self.server.cohort.wait_acknowledged()
                self.close_connection = True
                self.respond(200, b'{"acknowledged":true}')
            else:
                raise ExchangeError('INVALID_METHOD')
        except Exception as error:
            self.close_connection = True
            code = str(error) if isinstance(error, ExchangeError) else 'INVALID_REQUEST'
            try:
                self.respond(400, encode({'error': code}))
            except (OSError, ValueError):
                pass


def tls_context(tls, server=False):
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH if server else ssl.Purpose.SERVER_AUTH,
                                         cafile=tls['ca_file'])
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(tls['cert_file'], tls['key_file'])
    context.verify_mode = ssl.CERT_REQUIRED
    return context


def remote_exchange(address, context, instance_id, group_digest, documents, deadline, verify, server_fingerprint=None):
    """One connection, one payload transfer, one receipt ACK; no steady-state use."""
    connection = None
    while connection is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ExchangeError('CONNECT_TIMEOUT')
        if context is None:
            candidate = http.client.HTTPConnection(address[0], address[1],
                                                   timeout=min(remaining, 5))
        else:
            candidate = http.client.HTTPSConnection(address[0], address[1], context=context,
                                                    timeout=min(remaining, 5))
        try:
            candidate.connect()
            if (context is not None and server_fingerprint is not None
                    and digest(candidate.sock.getpeercert(binary_form=True)) != server_fingerprint):
                candidate.close()
                raise ExchangeError('COORDINATOR_CERTIFICATE_MISMATCH')
            candidate.sock.settimeout(max(0.001, deadline - time.monotonic()))
            connection = candidate
        except ssl.SSLError:
            candidate.close()
            raise ExchangeError('TLS_AUTHENTICATION_FAILED') from None
        except OSError:
            candidate.close()
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    try:
        def call(path, value):
            connection.request('POST', path, encode(value), {'Content-Type': 'application/json'})
            response = connection.getresponse()
            length = response.getheader('Content-Length', '')
            if not length.isdecimal() or not 0 < int(length) <= MAX_MESSAGE:
                raise ExchangeError('INVALID_RESPONSE_LENGTH')
            data = response.read(int(length) + 1)
            if response.status != 200 or len(data) != int(length):
                raise ExchangeError('REMOTE_EXCHANGE_FAILED')
            return data
        payload = call('/v1/exchange', dict(instance_id=instance_id, group_digest=group_digest, documents=documents))
        decoded = decode(payload)
        if decoded.get('group_digest') != group_digest:
            raise ExchangeError('GROUP_MISMATCH')
        verify(decoded['package'])
        call('/v1/ack', dict(instance_id=instance_id, payload_digest=digest(payload)))
        return decoded['package']
    finally:
        connection.close()
