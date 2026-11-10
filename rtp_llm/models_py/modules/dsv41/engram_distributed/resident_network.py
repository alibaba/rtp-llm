# SPDX-License-Identifier: Apache-2.0
"""Authenticated, read-only Owner discovery without a shared filesystem.

Each Owner serves its immutable descriptor to TCP peers in the C2 gang.
This read-only exchange relies on the platform's gang network isolation.
A random request challenge binds the reply to a fresh liveness probe.
"""

from __future__ import annotations

import fcntl
import hashlib
import ipaddress
import json
import logging
import os
import secrets
import socket
import socketserver
import stat
import struct
import tempfile
import threading
import time
from pathlib import Path
from uuid import NAMESPACE_OID, uuid5

from . import config
from .exchange import MAX_MESSAGE, ExchangeError, decode, encode

TRANSPORT_ENV = "RTP_ENGRAM_RESIDENT_TRANSPORT"
STATE_ENV = "RTP_ENGRAM_RESIDENT_STATE_DIR"
PORT_ENV = "RTP_ENGRAM_RESIDENT_PORT"
DEFAULT_PORT = 29484
_PROTOCOL = 2
_REQUEST_LIMIT = 4096
_RPC_TIMEOUT = 2.0
_LOGGER = logging.getLogger(__name__)


def _private(info, kind):
    if not kind(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise ExchangeError("UNSAFE_RESIDENT_LOCAL_STATE")


def _owner_port(env):
    port = env.get(PORT_ENV, str(DEFAULT_PORT))
    if not port.isascii() or not port.isdecimal() or not 1 <= int(port) <= 65535:
        raise config.ShardConfigError("INVALID_RESIDENT_PORT")
    return int(port)


def c2_topology(env):
    """Use the C2 gang projection as the only authority for keyless peers."""
    annotations = Path(env.get(config.ANNOTATION_ENV, config.DEFAULT_ANNOTATION))
    try:
        parts = config.read_parts(annotations)
        labels = dict(
            line.split("=", 1)
            for line in annotations.with_name("labels").read_text().splitlines()
            if "=" in line
        )
        version = json.loads(labels["gang-version-hash"])
        local_part = json.loads(labels["app.c2.io/gang-part-name"])
        if (
            not isinstance(version, str)
            or not version
            or not isinstance(local_part, str)
        ):
            raise ValueError()
        if [config.part_number(name) for name, _, _ in parts] != list(
            range(len(parts))
        ):
            raise ValueError()
        if len({config.family(name) for name, _, _ in parts}) != 1:
            raise ValueError()
        members = []
        for name, ip, _ in parts:
            address = ipaddress.IPv4Address(ip)
            if address.is_unspecified or address.is_multicast:
                raise ValueError()
            members.append((name, str(address)))
        [local_ip] = [
            ip
            for name, ip in members
            if name.replace("_", "-") == local_part.replace("_", "-")
        ]
    except (KeyError, OSError, ValueError):
        raise config.ShardConfigError("RESIDENT_C2_GANG_METADATA_INVALID") from None
    count = config.ShardConfig.from_environ(env).shard_count
    if len(members) != count:
        raise config.ShardConfigError("RESIDENT_ENDPOINT_COUNT_MISMATCH")
    port = _owner_port(env)
    peers = tuple((ip, port) for _, ip in members)
    identity = encode(["rtp-engram-c2-v1", version, members]).decode()
    return str(uuid5(NAMESPACE_OID, identity)), peers, local_ip


def _recv(sock, length, deadline):
    data = bytearray()
    while len(data) < length:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError()
        sock.settimeout(remaining)
        block = sock.recv(length - len(data))
        if not block:
            raise ConnectionError("OWNER_CONNECTION_CLOSED")
        data.extend(block)
    return bytes(data)


def _read_frame(sock, limit, deadline):
    size = struct.unpack("!I", _recv(sock, 4, deadline))[0]
    if not 0 < size <= limit:
        raise ExchangeError("INVALID_RESIDENT_FRAME")
    return _recv(sock, size, deadline)


def _send_frame(sock, payload):
    if len(payload) > MAX_MESSAGE:
        raise ExchangeError("MESSAGE_TOO_LARGE")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 64

    def __init__(self, address, owner):
        self.owner = owner
        self.slots = threading.BoundedSemaphore(16)
        super().__init__(address, _Handler)

    def verify_request(self, request, client_address):
        if client_address[0] in self.owner.allowed_ips:
            return True
        self.owner.rejected("outside_c2_gang")
        return False

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def handle_error(self, *_):
        # Do not log descriptor contents or handles on request failures.
        self.owner.rejected("internal")


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        owner = self.server.owner
        self.request.settimeout(_RPC_TIMEOUT)
        try:
            payload = _read_frame(
                self.request, _REQUEST_LIMIT, time.monotonic() + _RPC_TIMEOUT
            )
            message = decode(payload)
            expected = {"protocol", "kind", "generation", "revision", "shard", "nonce"}
            if (
                set(message) != expected
                or type(message["protocol"]) is not int
                or message["protocol"] != _PROTOCOL
                or message["kind"] != "get"
                or message["generation"] != owner.generation
                or message["revision"] != owner.revision
                or type(message["shard"]) is not int
                or message["shard"] != owner.shard
                or type(message["nonce"]) is not str
                or len(message["nonce"]) != 64
            ):
                owner.rejected("contract")
                return
            with owner.lock:
                document = owner.document
            reply = dict(
                protocol=_PROTOCOL,
                kind="reply",
                nonce=message["nonce"],
                generation=owner.generation,
                revision=owner.revision,
                shard=owner.shard,
                status="ready" if document is not None else "loading",
                document=document,
            )
            _send_frame(self.request, encode(reply))
        except (ValueError, ExchangeError):
            owner.rejected("framing")
        except OSError:
            return


class OwnerEndpoint:
    def __init__(self, generation, shard):
        self.generation = generation.generation
        self.revision = generation.revision
        self.shard = shard
        network = generation.network
        self.allowed_ips = frozenset(ip for ip, _ in network.peers)
        if network.peers[shard][0] != network.source_ip:
            raise config.ShardConfigError("RESIDENT_C2_OWNER_HOST_MISMATCH")
        self.document = None
        self.published = False
        self.lock = threading.Lock()
        self.rejections = 0
        self.last_rejection_log = float("-inf")
        self.server = _Server(generation.network.peers[shard], self)
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
        )

    def __enter__(self):
        self.thread.start()
        _LOGGER.info(
            "engram event=resident_endpoint_started generation=%s shard=%d pid=%d endpoint=%s:%d access=c2_gang",
            self.generation,
            self.shard,
            os.getpid(),
            *self.server.server_address,
        )
        return self

    def rejected(self, reason):
        with self.lock:
            self.rejections += 1
            now = time.monotonic()
            if now - self.last_rejection_log < 30:
                return
            self.last_rejection_log = now
            count = self.rejections
        _LOGGER.warning(
            "engram event=resident_request_rejected generation=%s shard=%d pid=%d reason=%s rejected_total=%d",
            self.generation,
            self.shard,
            os.getpid(),
            reason,
            count,
        )

    def publish(self, document):
        # Copy and bound the publication, then hold it immutably in Owner RAM.
        raw = encode(document)
        if len(raw) > MAX_MESSAGE - 1024:
            raise ExchangeError("MESSAGE_TOO_LARGE")
        document = decode(raw)
        if (
            document.get("generation") != self.generation
            or document.get("model_revision") != self.revision
            or document.get("shard") != self.shard
        ):
            raise ExchangeError("RESIDENT_ENGRAM_DESCRIPTOR_MISMATCH")
        with self.lock:
            if self.published:
                raise ExchangeError("CONFLICTING_REPUBLICATION")
            self.document = document
            self.published = True
        _LOGGER.info(
            "engram event=resident_descriptor_ready generation=%s shard=%d pid=%d descriptor_sha256=%s",
            self.generation,
            self.shard,
            os.getpid(),
            hashlib.sha256(encode(document)).hexdigest(),
        )

    def withdraw(self):
        with self.lock:
            self.document = None

    def __exit__(self, *_):
        self.withdraw()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        _LOGGER.info(
            "engram event=resident_endpoint_stopped generation=%s shard=%d pid=%d",
            self.generation,
            self.shard,
            os.getpid(),
        )


class OwnerClients:
    def __init__(self, env, generation, revision, directory):
        c2_generation, self.peers, self.source_ip = c2_topology(env)
        if c2_generation != generation:
            raise config.ShardConfigError("RESIDENT_C2_GANG_IDENTITY_MISMATCH")
        self.generation, self.revision = generation, revision
        self.directory = directory

    def request(self, shard, timeout):
        nonce = secrets.token_hex(32)
        deadline = time.monotonic() + timeout
        message = dict(
            protocol=_PROTOCOL,
            kind="get",
            generation=self.generation,
            revision=self.revision,
            shard=shard,
            nonce=nonce,
        )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.bind((self.source_ip, 0))
            sock.connect(self.peers[shard])
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError()
            sock.settimeout(remaining)
            _send_frame(sock, encode(message))
            reply = decode(_read_frame(sock, MAX_MESSAGE, deadline))
        if (
            set(reply)
            != {
                "protocol",
                "kind",
                "generation",
                "revision",
                "shard",
                "nonce",
                "status",
                "document",
            }
            or any(
                reply.get(k) != message[k]
                for k in ("protocol", "generation", "revision", "shard", "nonce")
            )
            or type(reply["protocol"]) is not int
            or type(reply["shard"]) is not int
            or reply["kind"] != "reply"
            or reply["status"] not in ("ready", "loading")
        ):
            raise ExchangeError("RESIDENT_OWNER_RESPONSE_MISMATCH")
        if reply["status"] == "loading":
            raise ConnectionError("RESIDENT_OWNER_NOT_READY")
        document = reply["document"]
        if (
            type(document) is not dict
            or document.get("generation") != self.generation
            or document.get("model_revision") != self.revision
            or document.get("shard") != shard
        ):
            raise ExchangeError("RESIDENT_ENGRAM_DESCRIPTOR_MISMATCH")
        return document

    def load(self, deadline):
        started = time.monotonic()
        last_log = float("-inf")
        retries = 0
        while True:
            documents = []
            try:
                for shard in range(len(self.peers)):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError()
                    documents.append(self.request(shard, min(_RPC_TIMEOUT, remaining)))
                _LOGGER.info(
                    "engram event=resident_owners_connected generation=%s pid=%d shards=%d retries=%d duration_s=%.3f",
                    self.generation,
                    os.getpid(),
                    len(documents),
                    retries,
                    time.monotonic() - started,
                )
                return documents
            except OSError as error:
                retries += 1
                now = time.monotonic()
                if now - last_log >= 10:
                    last_log = now
                    _LOGGER.warning(
                        "engram event=resident_owner_wait generation=%s pid=%d shard=%d endpoint=%s:%d retries=%d error_class=%s",
                        self.generation,
                        os.getpid(),
                        len(documents),
                        *self.peers[len(documents)],
                        retries,
                        "timeout" if isinstance(error, TimeoutError) else "connection",
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _LOGGER.error(
                        "engram event=resident_owner_timeout generation=%s pid=%d shard=%d duration_s=%.3f",
                        self.generation,
                        os.getpid(),
                        len(documents),
                        time.monotonic() - started,
                    )
                    raise ExchangeError("RESIDENT_ENGRAM_STARTUP_TIMEOUT") from None
                time.sleep(min(0.25, remaining))
            except ExchangeError as error:
                _LOGGER.error(
                    "engram event=resident_owner_validation_failed generation=%s pid=%d shard=%d code=%s",
                    self.generation,
                    os.getpid(),
                    len(documents),
                    error.code,
                )
                raise

    def pin(self, documents):
        """Private local hashes survive worker exits; never overwrite old pins."""
        self.directory.mkdir(mode=0o700, exist_ok=True)
        _private(self.directory.lstat(), stat.S_ISDIR)
        fd = os.open(
            self.directory / "pins.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
        )
        with os.fdopen(fd, "r+b") as f:
            _private(os.fstat(f.fileno()), stat.S_ISREG)
            fcntl.flock(f, fcntl.LOCK_EX)
            pins = encode(
                {
                    str(i): hashlib.sha256(encode(doc)).hexdigest()
                    for i, doc in enumerate(documents)
                }
            )
            path = self.directory / "pins"
            try:
                with os.fdopen(
                    os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb"
                ) as saved:
                    _private(os.fstat(saved.fileno()), stat.S_ISREG)
                    previous = saved.read(8193)
                if previous != pins:
                    _LOGGER.error(
                        "engram event=resident_owner_allocation_changed generation=%s pid=%d",
                        self.generation,
                        os.getpid(),
                    )
                    raise ExchangeError("RESIDENT_OWNER_ALLOCATION_CHANGED")
            except FileNotFoundError:
                with tempfile.NamedTemporaryFile(
                    dir=self.directory, delete=False
                ) as target:
                    temporary = Path(target.name)
                    try:
                        target.write(pins)
                        target.flush()
                        os.fsync(target.fileno())
                        os.replace(temporary, path)
                    finally:
                        temporary.unlink(missing_ok=True)
        _LOGGER.info(
            "engram event=resident_owner_identity_verified generation=%s pid=%d shards=%d",
            self.generation,
            os.getpid(),
            len(documents),
        )
