"""One in-memory cache of completed downloads per ViT service."""

import hashlib
import logging
import os
import socket
import socketserver
import struct
import tempfile
import threading
import time
from collections import OrderedDict
from io import BytesIO
from multiprocessing.util import Finalize

# Local cache access is optional. Bound the whole socket exchange, including
# connect and transfer, so a stalled cache owner cannot stall an HTTP download.
_CACHE_TIMEOUT = 0.1
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
    def __init__(self, capacity):
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
            if self.capacity and key not in self.items:
                self.items[key] = data
                if len(self.items) > self.capacity:
                    # Drop the last reference after releasing the index lock.
                    _ = self.items.popitem(last=False)
        finally:
            self.lock.release()


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        deadline = time.monotonic() + _CACHE_TIMEOUT
        try:
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


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    request_queue_size = 128

    def __init__(self, path, store):
        self.store = store
        super().__init__(path, _Handler)


def _stop_server(server, directory):
    server.shutdown()
    server.server_close()
    directory.cleanup()


class URLDataCache:
    def __init__(self, cache_size):
        self._store = _ByteStore(cache_size)
        self._capacity = cache_size
        self._address = None
        self._owner_pid = None

    def share(self, capacity):
        """Pass just the socket path to every engine and its replacement workers."""
        self.resize_cache(capacity)
        if self._address is None and capacity > 0:
            try:
                directory = tempfile.TemporaryDirectory(prefix="vit-url-", dir="/tmp")
                path = os.path.join(directory.name, "cache.sock")
                server = _Server(path, self._store)
                threading.Thread(
                    target=server.serve_forever, args=(0.05,), daemon=True
                ).start()
            except (OSError, RuntimeError):
                logging.exception("Shared URL cache unavailable; bypassing cache")
                self._address = ""
            else:
                self._address = path
                self._owner_pid = os.getpid()
                Finalize(self, _stop_server, args=(server, directory), exitpriority=1)
        return self._address if capacity > 0 else None

    def attach(self, handle, capacity):
        self._address = handle
        self.resize_cache(capacity)

    def resize_cache(self, capacity):
        self._capacity = capacity
        self._store.resize(capacity)

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
        if self._address is None or self._owner_pid == os.getpid():
            data = self._store.get(key)
        else:
            data = self._exchange(key, None, deadline)
        return None if data is None else BytesIO(data)

    def insert_cache(self, key, source, deadline=None):
        if self._address is None or self._owner_pid == os.getpid():
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
