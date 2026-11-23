"""One in-memory cache of completed downloads per ViT service."""

import hashlib
import logging
import multiprocessing
import os
import socket
import socketserver
import struct
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from multiprocessing.util import Finalize

# Local cache access is optional. Bound the whole socket exchange, including
# connect and transfer, so a stalled cache owner cannot stall an HTTP download.
_CACHE_TIMEOUT = 0.1
_CACHE_WORKERS = 16
_CACHE_START_TIMEOUT = 20
# One request per private Unix connection: key and byte count (-1 for GET).
_REQUEST = struct.Struct("!32sq")
_SIZE = struct.Struct("!q")


def _set_timeout(sock, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("URL cache deadline expired")
    sock.settimeout(remaining)


def _read(stream, size, sock, deadline):
    # read1 performs at most one socket read, so a slow sender cannot restart
    # the deadline for every fragment. An interrupted PUT never becomes visible.
    data = BytesIO()
    while data.tell() < size:
        _set_timeout(sock, deadline)
        chunk = stream.read1(min(size - data.tell(), 1024 * 1024))
        if not chunk:
            raise EOFError
        data.write(chunk)
    return data.getvalue()


class _ByteStore:
    def __init__(self, capacity, *, replace_existing=False):
        self.replace_existing = replace_existing
        self.items = OrderedDict()
        self.lock = threading.Lock()
        self.resize(capacity)

    def resize(self, capacity):
        with self.lock:
            self.capacity = max(0, capacity)
            while len(self.items) > self.capacity:
                self.items.popitem(last=False)

    def get(self, key):
        if not self.lock.acquire(blocking=False):
            return None
        try:
            data = self.items.get(key)
            if data is not None:
                self.items.move_to_end(key)
            return data
        finally:
            self.lock.release()

    def put(self, key, data):
        if not self.lock.acquire(blocking=False):
            return
        try:
            if self.capacity and (self.replace_existing or key not in self.items):
                self.items[key] = data
                self.items.move_to_end(key)
                if len(self.items) > self.capacity:
                    # Drop the last reference after releasing the index lock.
                    _ = self.items.popitem(last=False)
        finally:
            self.lock.release()


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        deadline = time.monotonic() + _CACHE_TIMEOUT
        try:
            capacity = self.server.capacity.value
            if capacity != self.server.store.capacity:
                self.server.store.resize(capacity)
            key, size = _REQUEST.unpack(
                _read(self.rfile, _REQUEST.size, self.request, deadline)
            )
            if size == -1:
                data = self.server.store.get(key)
                _set_timeout(self.request, deadline)
                self.request.sendall(_SIZE.pack(-1 if data is None else len(data)))
                if data:
                    _set_timeout(self.request, deadline)
                    self.request.sendall(data)
            elif size >= 0:
                data = _read(self.rfile, size, self.request, deadline)
                self.server.store.put(key, data)
                _set_timeout(self.request, deadline)
                self.request.sendall(_SIZE.pack(0))
        except (EOFError, OSError):
            # Worker exited, timed out, or cancelled the connection.
            return


class _Server(socketserver.UnixStreamServer):
    request_queue_size = 128

    def __init__(self, path, capacity, replace_existing):
        self.capacity = capacity
        self.store = _ByteStore(capacity.value, replace_existing=replace_existing)
        super().__init__(path, _Handler)
        self._available = threading.BoundedSemaphore(_CACHE_WORKERS)
        self._workers = ThreadPoolExecutor(_CACHE_WORKERS, thread_name_prefix="vit-url")

    def process_request(self, request, client_address):
        # The download cache is optional. A full worker pool should cause a
        # cache bypass instead of queueing work after the caller's deadline.
        if not self._available.acquire(blocking=False):
            self.shutdown_request(request)
            return
        self._workers.submit(self._serve_request, request, client_address)

    def _serve_request(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)
            self._available.release()

    def server_close(self):
        super().server_close()
        self._workers.shutdown()


def _serve_cache(path, capacity, replace_existing, ready):
    # The ViT request process must not share its GIL with socket handling,
    # transient request allocations, or collection in the download cache.
    with _Server(path, capacity, replace_existing) as server:
        with ready:
            ready.send_bytes(b"")
        server.serve_forever(poll_interval=0.05)


def _stop_cache(process, directory):
    if process is not None:
        if process.pid is not None and process.is_alive():
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
        if process.pid is not None:
            process.join()
        process.close()
    directory.cleanup()


class URLDataCache:
    def __init__(self, cache_size, *, replace_existing=False):
        self._store = _ByteStore(cache_size, replace_existing=replace_existing)
        self._capacity = cache_size
        self._address = None
        self._shared_capacity = None
        self._finalizer = None

    def share(self, capacity):
        """Pass just the socket path to every engine and its replacement workers."""
        self.resize_cache(capacity)
        if self._address is None and capacity > 0:
            ctx = multiprocessing.get_context("spawn")
            directory = process = None
            try:
                directory = tempfile.TemporaryDirectory(prefix="vit-url-", dir="/tmp")
                path = os.path.join(directory.name, "cache.sock")
                shared_capacity = ctx.Value("q", capacity)
                reader, writer = ctx.Pipe(duplex=False)
                with reader, writer:
                    process = ctx.Process(
                        name="vit-url-cache",
                        target=_serve_cache,
                        args=(path, shared_capacity, self._store.replace_existing, writer),
                        daemon=True,
                    )
                    process.start()
                    writer.close()
                    if not reader.poll(_CACHE_START_TIMEOUT):
                        raise TimeoutError("shared URL cache did not start")
                    reader.recv_bytes()
            except (EOFError, OSError, RuntimeError):
                if directory is not None:
                    _stop_cache(process, directory)
                logging.exception("Shared URL cache unavailable; bypassing cache")
                self._address = ""
            else:
                self._shared_capacity = shared_capacity
                self._address = path
                self._finalizer = Finalize(
                    self, _stop_cache, args=(process, directory), exitpriority=1
                )
        return self._address if capacity > 0 else None

    def close(self):
        if self._finalizer is not None:
            self._finalizer()
            self._finalizer = None
            self._address = ""
            self._shared_capacity = None

    def attach(self, handle, capacity):
        self._address = handle
        self.resize_cache(capacity)

    def resize_cache(self, capacity):
        self._capacity = capacity
        self._store.resize(capacity)
        if self._shared_capacity is not None:
            self._shared_capacity.value = max(0, capacity)

    def _exchange(self, key, data, deadline):
        end = time.monotonic() + _CACHE_TIMEOUT
        if deadline is not None:
            end = min(end, deadline)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            _set_timeout(sock, end)
            sock.connect(self._address)
            _set_timeout(sock, end)
            sock.sendall(_REQUEST.pack(key, -1 if data is None else len(data)))
            if data:
                _set_timeout(sock, end)
                sock.sendall(data)
            with sock.makefile("rb") as stream:
                (size,) = _SIZE.unpack(_read(stream, _SIZE.size, sock, end))
                return _read(stream, size, sock, end) if size >= 0 else None

    def check_cache(self, key, deadline=None):
        if self._address is None:
            data = self._store.get(key)
        else:
            data = self._exchange(key, None, deadline)
        return None if data is None else BytesIO(data)

    def insert_cache(self, key, source, deadline=None):
        if self._address is None:
            self._store.put(key, source.getvalue())
        else:
            self._exchange(key, source.getvalue(), deadline)

    def get_or_load(self, url, headers, load, deadline=None):
        if self._capacity <= 0 or self._address == "":
            return load(), False
        # The same URL can return different bytes under different credentials.
        key = hashlib.sha256(headers.encode() + b"\0" + url.encode()).digest()
        try:
            cached = self.check_cache(key, deadline)
        except (EOFError, OSError):
            # Bypass publication too when the owner was unreachable on lookup.
            return load(), False
        if cached is not None:
            return cached, True
        source = load()
        try:
            self.insert_cache(key, source, deadline)
        except (EOFError, OSError):
            # The caller already owns a complete independent download.
            pass
        return source, False
