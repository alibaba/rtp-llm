# SPDX-License-Identifier: Apache-2.0
"""Authenticated, read-only Owner discovery without a shared filesystem.

Each Owner serves its own immutable descriptor. Requests and responses use
Fernet authenticated encryption with a per-generation key provisioned into a
private LOCAL file on each member. No descriptor or key is logged. A random
request challenge prevents an old response from passing a fresh liveness probe.
There is no central coordinator or mutation RPC.
"""

from __future__ import annotations

import fcntl
import hashlib
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

from . import config
from .exchange import MAX_MESSAGE, ExchangeError, decode, encode

TRANSPORT_ENV = "RTP_ENGRAM_RESIDENT_TRANSPORT"
STATE_ENV = "RTP_ENGRAM_RESIDENT_STATE_DIR"
KEY_ENV = "RTP_ENGRAM_RESIDENT_KEY_FILE"
ENDPOINTS_ENV = "RTP_ENGRAM_RESIDENT_ENDPOINTS"
PORT_ENV = "RTP_ENGRAM_RESIDENT_PORT"
DEFAULT_PORT = 29484
_MAX_FRAME = 2 * MAX_MESSAGE
_REQUEST_LIMIT = 4096
_RPC_TIMEOUT = 2.0
_TTL = 30
_LOGGER = logging.getLogger(__name__)


def network_requested(env):
    transport = env.get(TRANSPORT_ENV, "file")
    if transport not in ("file", "tcp"):
        raise config.ShardConfigError("INVALID_RESIDENT_TRANSPORT")
    return transport == "tcp"


def _private(info, kind):
    if not kind(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise ExchangeError("UNSAFE_RESIDENT_LOCAL_STATE")


def _cipher(env):
    from cryptography.fernet import Fernet

    path = Path(env.get(KEY_ENV, ""))
    if not path.is_absolute():
        raise config.ShardConfigError("RESIDENT_KEY_FILE_REQUIRED")
    with os.fdopen(
        os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb"
    ) as f:
        _private(os.fstat(f.fileno()), stat.S_ISREG)
        key = f.read(129).strip()
    if len(key) != 44:
        raise config.ShardConfigError("INVALID_RESIDENT_KEY")
    try:
        return Fernet(key)
    except (ValueError, TypeError):
        raise config.ShardConfigError("INVALID_RESIDENT_KEY") from None


def endpoints(env, count):
    port = env.get(PORT_ENV, str(DEFAULT_PORT))
    if not port.isascii() or not port.isdecimal() or not 1 <= int(port) <= 65535:
        raise config.ShardConfigError("INVALID_RESIDENT_PORT")
    if ENDPOINTS_ENV in env:
        peers = config._parse_members(env[ENDPOINTS_ENV], port)
    else:
        peers = config._cohort(env) or config._ganginfo_cohort(env)
        if peers is None:
            candidates = []
            for host in sorted(config._local_addresses()):
                try:
                    found = config._annotation_cohort(env, [host])
                except config.ShardConfigError as error:
                    if str(error) != "LOCAL_HOST_NOT_IN_GANGINFO":
                        raise
                    continue
                if found and found not in candidates:
                    candidates.append(found)
            if len(candidates) != 1:
                raise config.ShardConfigError("RESIDENT_ENDPOINTS_REQUIRED")
            peers = candidates[0]
        # Worker rendezvous ports belong to the workers. Owner ports must live
        # independently and are identical on all hosts unless explicitly given.
        peers = [(host, int(port)) for host, _ in peers]
    if len(peers) != count:
        raise config.ShardConfigError("RESIDENT_ENDPOINT_COUNT_MISMATCH")
    return tuple(peers)


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
    if len(payload) > _MAX_FRAME:
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
        # Never let an untrusted request print a decrypted descriptor or key.
        self.owner.rejected("internal")


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        from cryptography.fernet import InvalidToken

        owner = self.server.owner
        self.request.settimeout(_RPC_TIMEOUT)
        try:
            message = decode(
                owner.cipher.decrypt(
                    _read_frame(
                        self.request, _REQUEST_LIMIT, time.monotonic() + _RPC_TIMEOUT
                    ),
                    ttl=_TTL,
                )
            )
            expected = {"protocol", "kind", "generation", "revision", "shard", "nonce"}
            if (
                set(message) != expected
                or type(message["protocol"]) is not int
                or message["protocol"] != 1
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
                protocol=1,
                kind="reply",
                nonce=message["nonce"],
                generation=owner.generation,
                revision=owner.revision,
                shard=owner.shard,
                status="ready" if document is not None else "loading",
                document=document,
            )
            _send_frame(self.request, owner.cipher.encrypt(encode(reply)))
        except InvalidToken:
            owner.rejected("authentication")
        except (ValueError, ExchangeError):
            owner.rejected("framing")
        except OSError:
            return


class OwnerEndpoint:
    def __init__(self, generation, shard):
        self.generation = generation.generation
        self.revision = generation.revision
        self.shard = shard
        self.cipher = generation.network.cipher
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
            "engram event=resident_endpoint_started generation=%s shard=%d pid=%d endpoint=%s:%d",
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
    def __init__(self, env, generation, revision, count, directory):
        self.peers = endpoints(env, count)
        self.cipher = _cipher(env)
        self.generation, self.revision = generation, revision
        self.directory = directory

    def request(self, shard, timeout):
        from cryptography.fernet import InvalidToken

        nonce = secrets.token_hex(32)
        deadline = time.monotonic() + timeout
        message = dict(
            protocol=1,
            kind="get",
            generation=self.generation,
            revision=self.revision,
            shard=shard,
            nonce=nonce,
        )
        with socket.create_connection(self.peers[shard], timeout=timeout) as sock:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError()
            sock.settimeout(remaining)
            _send_frame(sock, self.cipher.encrypt(encode(message)))
            try:
                reply = decode(
                    self.cipher.decrypt(
                        _read_frame(sock, _MAX_FRAME, deadline), ttl=_TTL
                    )
                )
            except InvalidToken:
                raise ExchangeError("RESIDENT_OWNER_AUTH_FAILED") from None
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
