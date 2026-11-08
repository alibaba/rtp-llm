"""Prefix-cached incremental tokenization for append-only prompt streams.

Agentic workloads resend the full growing session history on every step, so the
frontend re-tokenizes prompts whose 93-95% prefix is byte-identical to the
previous step's prompt. This module caches per-session tokenization state and
splices the cached ids with tail encodes cut at true pre-tokenizer chunk
boundaries, where BPE merges never cross (verified exact on the served
tokenizer family). Every failure path (cold start, session remap, prefix-hash
mismatch, eviction, internal error) falls back to the exact full encode, and an
optional sampled shadow verification reports mismatches and fails safe to the
full-encode result.

Workers additionally share cached entries with their in-pod peers over unix
sockets (peer cache-entry sharing). Frontends bind one SO_REUSEPORT socket per
process, so per-request connections land on a uniformly random worker and a
per-process-only cache sees each session once every ~N steps — its local entry
is then N steps stale and the splice degenerates into near-full re-encodes.
Sharing fixes the freshness, not the dispatch: unless the local entry is fresh
(cut within one append of the prompt end), the worker issues ONE GET to the
pod's storage daemon — a tiny detached process spawned by a lazy bind race
that holds the freshest announced entry per session key (max-cut-wins,
byte-capped FIFO) and serves it from its own memory in ~0.5-1.5ms regardless
of any worker's load, keeping worker GILs entirely out of the fetch path.
Every cut advance is announced to the daemon fire-and-forget; a cut-only
LOOKUP broadcast sweep over the peer worker sockets (which keep their serve
threads) is the daemon-down fallback. The winning entry's prefix hash is
verified locally before use and spliced exactly like a local hit, so every
request sees a one-step-lagged cut at any worker count. Any sharing error,
timeout, NACK or verification failure degrades to the stale local entry when
one exists and otherwise to the full-encode miss path; serving and fetching
never touch the request-serving code path beyond the cache call.

Disabled by default; enable with RTP_TOKENIZE_CACHE_ENABLE=1 (plus optional
RTP_TOKENIZE_CACHE_MAX_BYTES, RTP_TOKENIZE_CACHE_SHADOW_RATE and
RTP_TOKENIZE_CACHE_SHARE_DIR; an explicitly empty share dir disables peer
sharing while keeping the local cache).
"""

import array
import atexit
import errno
import hashlib
import logging
import os
import random
import select
import socket
import struct
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple

try:
    from rtp_llm.metrics import AccMetrics, GaugeMetrics
    from rtp_llm.metrics import kmonitor as _kmonitor
except ImportError:  # keep this module importable in minimal test sandboxes
    AccMetrics = None  # type: ignore[assignment]
    GaugeMetrics = None  # type: ignore[assignment]
    _kmonitor = None

_ENV_ENABLE = "RTP_TOKENIZE_CACHE_ENABLE"
_ENV_MAX_BYTES = "RTP_TOKENIZE_CACHE_MAX_BYTES"
_ENV_SHADOW_RATE = "RTP_TOKENIZE_CACHE_SHADOW_RATE"
_ENV_SHARE_DIR = "RTP_TOKENIZE_CACHE_SHARE_DIR"
_ENV_SHARE_TIMEOUT_MS = "RTP_TOKENIZE_CACHE_SHARE_TIMEOUT_MS"
_ENV_SHARE_SOFT_WINDOW_MS = "RTP_TOKENIZE_CACHE_SHARE_SOFT_WINDOW_MS"

_DEFAULT_MAX_BYTES = 128 * 1024 * 1024
_DEFAULT_SHARE_DIR = "/tmp/rtp_tokcache_share"
# Sessions are bucketed by the hash of the first _KEY_CHARS characters (the
# stable session header). Prompts shorter than the window still work but miss
# on every append (exact full-encode fallback), which is fine: the mechanism
# targets the >100KB agentic prompts it was designed for.
_KEY_CHARS = 256
_COMPAT_PROBE = "rtp-llm tokenize cache compatibility probe 12345"
_TRUTHY = ("1", "true", "yes", "on")

# Peer sharing wire protocol: every frame is a big-endian u32 length followed
# by that many payload bytes. Requests are exactly one op byte + 64 hex chars
# (the session key): b"L" LOOKUP (cut-only reply) or b"F" FETCH (full entry).
# Responses are b"N" (not held), b"H" + u32 cut (LOOKUP hold), or b"E" + u32
# cut + 64-byte prefix_sha + raw native-endian int32 id bytes (same host only).
_SHARE_PREFIX = "worker_"
_SHARE_SUFFIX = ".sock"
_REQ_LEN = 65
_RESP_HOLD_LEN = 5
_SHARE_MAX_FRAME = 32 * 1024 * 1024
# Overall LOOKUP+FETCH deadline. Sized above the worst GIL-held burst of a
# busy peer's request thread (e40 bench: C-level materializations ~50-150ms
# at 490k-token chains) yet far below any full-encode miss cost (~0.4-2.9s).
_SHARE_FETCH_TIMEOUT_S = 0.2
_SHARE_SERVE_TIMEOUT_S = 0.2
# Once at least one hold is known, the LOOKUP sweep stops after an ADAPTIVE
# soft window instead of waiting out slow peers to the deadline: splicing
# from the best hold in hand costs at most one extra ~6.5% tail (~4-8ms),
# far less than a deadline cliff. The window guarantees at least
# _SHARE_SOFT_WINDOW_S, expands with measured peer responsiveness
# (_SHARE_SOFT_WINDOW_K x first-hold arrival — patient under load, when
# holds arrive late but are worth collecting, and snappy when idle), and is
# capped at _SHARE_SOFT_WINDOW_CAP_S. With zero holds the sweep still waits
# the full deadline (waiting 200ms beats a 0.4-2.9s full encode).
_SHARE_SOFT_WINDOW_S = 0.004
_SHARE_SOFT_WINDOW_K = 3.0
_SHARE_SOFT_WINDOW_CAP_S = 0.040
# Trailing window for the bounded _conservative_cut scan (see its docstring).
_CUT_SUFFIX_CHARS = 16384
# Per-pod directory daemon: one detached process holding key -> (owner_path,
# cut), so discovery of the FRESHEST holder costs ~0.5ms and never waits on a
# busy worker's GIL (the e40 saturation runs proved time-based broadcast
# truncation systematically drops the busy owner's hold). Lazy double-fork
# spawn race from start_share(); every failure mode falls back to the
# broadcast sweep, then to stale local, then to the full-encode failsafe.
_DIR_SOCK_NAME = "directory.sock"
_DIR_MAP_CAP = 100000
_DIR_STORE_CAP_BYTES = 256 * 1024 * 1024
_DIR_IDLE_TTL_S = 300.0
_DIR_CONNECT_TIMEOUT_S = 0.05
_DIR_LOOKUP_TIMEOUT_S = 0.1
_DIR_ANNOUNCE_TIMEOUT_S = 0.2
_DIR_RESPAWN_COOLDOWN_S = 5.0
_DIR_QUIT = object()
# A local entry whose cut covers at least this fraction of the prompt is fresh
# enough (tail within one step; the frozen append pattern grows ~6.5%/step) to
# use WITHOUT asking peers. Staler local entries compete with peer holds by
# cut, so the freshest verified state always wins the splice.
_FRESH_FRAC = 0.92


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """Read exactly ``n`` bytes; ``None`` on clean EOF (timeouts raise)."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _dir_handle_frame(payload: bytes, directory: Dict[bytes, tuple], stats: dict):
    """Pure directory frame handler: response bytes, None, or _DIR_QUIT.

    STORAGE-daemon protocol (Candidate A, E12-fetch-topology-design-note):
    ``P`` key64 u32cut sha64 ids_bytes -> silent max-cut-wins PUT of the FULL
        entry; byte-capped FIFO eviction (announces refresh recency).
    ``G`` key64 -> ``E`` u32cut sha64 ids_bytes | ``N`` -- served from daemon
        memory in ~0.5-1.5ms (e38 primitives): the worker GIL is entirely out
        of the fetch path. The response body is byte-identical to the worker
        FETCH ENTRY frame, so consumers reuse the same verified parser.
    ``Q`` -> daemon shutdown (tests/admin).
    Anything else is ignored (the connection stays usable).
    """
    op = payload[:1]
    if op == b"P" and len(payload) >= 133:
        key = payload[1:65]
        (cut,) = struct.unpack(">I", payload[65:69])
        sha = payload[69:133]
        ids = payload[133:]
        old = directory.get(key)
        if old is None or cut >= old[0]:
            if old is not None:
                stats["bytes"] -= old[2]
                del directory[key]
            directory[key] = (cut, sha + ids, len(ids))
            stats["bytes"] += len(ids)
            _dir_evict(directory, stats)
        return None
    if op == b"G" and len(payload) == 65:
        hit = directory.get(payload[1:])
        if hit is None:
            return struct.pack(">I", 1) + b"N"
        cut, blob, _ = hit
        body = b"E" + struct.pack(">I", cut) + blob
        return struct.pack(">I", len(body)) + body
    if op == b"Q":
        return _DIR_QUIT
    return None


def _dir_evict(directory: Dict[bytes, tuple], stats: dict) -> None:
    """FIFO-evict (by announce order) until both caps hold."""
    while len(directory) > 1 and (
        stats["bytes"] > _DIR_STORE_CAP_BYTES or len(directory) > _DIR_MAP_CAP
    ):
        oldest = next(iter(directory))
        stats["bytes"] -= directory.pop(oldest)[2]


def _dir_probe_alive(dpath: str) -> bool:
    """True when a daemon answers on ``dpath`` (its file may be stale)."""
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(_DIR_CONNECT_TIMEOUT_S)
    try:
        probe.connect(dpath)
        return True
    except OSError:
        return False
    finally:
        try:
            probe.close()
        except OSError:
            pass


def _dir_bind_race(path: str) -> Optional[socket.socket]:
    """Bind the daemon socket; None when a live daemon wins the race."""
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        candidate = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            candidate.bind(path)
            return candidate
        except OSError:
            candidate.close()
            if _dir_probe_alive(path):
                return None  # a live daemon won; this candidate retires
            try:
                os.unlink(path)  # stale socket file with no listener
            except OSError:
                pass
            time.sleep(0.1)
    return None


def _dir_spawn_candidate(dpath: str) -> None:
    """Double-fork a detached daemon candidate (loser reaped by init).

    child1 detaches (setsid) and forks child2, which closes inherited
    descriptors and runs the daemon; child1 exits at once and is waitpid-
    reaped here, so the daemon reparents to init and outlives this worker.
    """
    try:
        pid = os.fork()
    except OSError:
        return
    if pid == 0:
        _dir_child1(dpath)  # never returns
    try:
        os.waitpid(pid, 0)
    except OSError:
        pass


def _dir_child1(dpath: str) -> None:
    """Detached intermediate child: setsid, fork the daemon, exit at once."""
    try:
        os.setsid()
    except OSError:
        pass
    try:
        pid2 = os.fork()
        if pid2 == 0:
            try:
                os.closerange(3, 4096)
            except OSError:
                pass
            _directory_daemon_main(dpath)  # never returns (os._exit)
        os._exit(0)
    except Exception:
        os._exit(1)


def _dir_accept(sock: socket.socket, conns: Dict[socket.socket, bytearray]) -> None:
    try:
        conn, _ = sock.accept()
    except OSError:
        return
    # blocking-with-timeout (not non-blocking): GET responses carry up to
    # ~2MB entries, and sendall must be able to block briefly until the
    # kernel buffer drains instead of failing with EWOULDBLOCK
    conn.settimeout(_SHARE_SERVE_TIMEOUT_S)
    conns[conn] = bytearray()


def _dir_drop_conn(r: socket.socket, conns: Dict[socket.socket, bytearray]) -> None:
    conns.pop(r, None)
    try:
        r.close()
    except OSError:
        pass


def _dir_conn_frames(
    r: socket.socket,
    conns: Dict[socket.socket, bytearray],
    directory: Dict[bytes, tuple],
    stats: dict,
) -> bool:
    """Drain one client socket's frames; True when QUIT was received."""
    buf = conns.get(r)
    if buf is None:
        return False
    try:
        chunk = r.recv(65536)
    except OSError:
        chunk = b""
    if not chunk:
        _dir_drop_conn(r, conns)
        return False
    buf.extend(chunk)
    while len(buf) >= 4:
        (n,) = struct.unpack(">I", bytes(buf[:4]))
        if n < 1 or n > _SHARE_MAX_FRAME:
            _dir_drop_conn(r, conns)  # garbage frame: drop the connection
            return False
        frame_end = 4 + n
        if len(buf) < frame_end:
            return False  # incomplete frame; wait for more bytes
        payload = bytes(buf[4:frame_end])
        del buf[:frame_end]
        resp = _dir_handle_frame(payload, directory, stats)
        if resp is _DIR_QUIT:
            return True
        if resp:
            try:
                r.sendall(resp)
            except OSError:
                _dir_drop_conn(r, conns)
                return False
    return False


def _dir_shutdown(sock: socket.socket, path: str) -> None:
    try:
        sock.close()
    except OSError:
        pass
    try:
        os.unlink(path)
    except OSError:
        pass
    os._exit(0)


def _directory_daemon_main(path: str) -> None:
    """Serve the pod directory until QUIT or idle-TTL; runs in a detached
    forked child, exits via ``os._exit`` and never returns.

    Fork-safety contract: touches ONLY fresh local data and the stdlib — the
    parent worker may hold arbitrary locks (cache lock, logging) at fork
    time, and only the forking thread survives in the child. No logging, no
    parent object state, no atexit.
    """
    sock = _dir_bind_race(path)
    if sock is None:
        os._exit(0)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    sock.listen(64)
    directory: Dict[bytes, tuple] = {}
    stats = {"bytes": 0}
    conns: Dict[socket.socket, bytearray] = {}
    last_activity = time.monotonic()
    while True:
        try:
            rlist, _, _ = select.select([sock] + list(conns), [], [], 1.0)
        except OSError:
            break
        if not rlist:
            if time.monotonic() - last_activity > _DIR_IDLE_TTL_S:
                break  # idle retirement (tests, pod drain, orphaned daemon)
            continue
        last_activity = time.monotonic()
        for r in rlist:
            if r is sock:
                _dir_accept(sock, conns)
            elif _dir_conn_frames(r, conns, directory, stats):
                _dir_shutdown(sock, path)
    _dir_shutdown(sock, path)


def _dir_quit(share_dir: str) -> None:
    """Best-effort directory-daemon shutdown (tests/admin)."""
    dpath = os.path.join(share_dir, _DIR_SOCK_NAME)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        sock.connect(dpath)
        sock.sendall(struct.pack(">I", 1) + b"Q")
    except OSError:
        pass
    finally:
        try:
            sock.close()
        except OSError:
            pass


class _Entry:
    """Cached tokenization state for one session prefix."""

    __slots__ = ("cut", "ids", "prefix_sha", "nbytes", "stamp")

    def __init__(self, cut: int, ids: "array.array[int]", prefix_sha: str, stamp: int):
        self.cut = cut
        self.ids = ids
        self.prefix_sha = prefix_sha
        # Only the id array and hashes are retained (the prefix text itself is
        # never stored), so entries cost ~4 bytes per cached token.
        self.nbytes = len(ids) * 4 + 128
        self.stamp = stamp


class _ShareServer:
    """Serve one worker's cache entries to in-pod peers over a unix socket.

    One daemon thread per worker process. Every failure mode (bind error,
    per-connection error, malformed frame, slow client) is contained: sharing
    degrades to local-only for this worker and the cache keeps serving exact
    results. Connections are short-lived broadcast fetches; the thread answers
    each request from the cache dict under the cache lock (~0.5ms measured for
    the largest chains).
    """

    def __init__(self, cache: "IncrementalTokenizeCache", path: str):
        self._cache = cache
        self._path = path
        self._sock: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def path(self) -> str:
        return self._path

    def start(self) -> bool:
        sock: Optional[socket.socket] = None
        try:
            parent = os.path.dirname(self._path)
            if parent:
                os.makedirs(parent, mode=0o700, exist_ok=True)
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(self._path)
            try:
                os.chmod(self._path, 0o600)
            except OSError:
                pass
            sock.listen(64)
            sock.settimeout(0.5)  # accept tick so close() joins promptly
        except OSError as e:
            logging.warning(
                f"tokenize cache: share socket unavailable ({e}); staying local-only"
            )
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
            return False
        self._sock = sock
        self._thread = threading.Thread(
            target=self._serve, name="tokcache-share", daemon=True
        )
        self._thread.start()
        atexit.register(self.close)
        return True

    def _serve(self) -> None:
        sock = self._sock
        if sock is None:  # pragma: no cover - start() guards this
            return
        while not self._stop.is_set():
            try:
                conn, _ = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                self._handle(conn)
            except Exception as e:
                logging.debug(f"tokenize cache: share connection error: {e}")
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def _handle(self, conn: socket.socket) -> None:
        conn.settimeout(_SHARE_SERVE_TIMEOUT_S)
        while not self._stop.is_set():
            header = _recv_exact(conn, 4)
            if header is None:
                return  # clean close
            (n,) = struct.unpack(">I", header)
            if n != _REQ_LEN:
                return  # malformed frame; drop the connection
            payload = _recv_exact(conn, n)
            if payload is None or payload[:1] not in (b"L", b"F"):
                return
            response = self._cache._serve_request(
                payload[:1], payload[1:].decode("ascii", "replace")
            )
            conn.sendall(struct.pack(">I", len(response)) + response)

    def close(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        try:
            os.unlink(self._path)
        except OSError:
            pass


class IncrementalTokenizeCache:
    """Splice cached prefix ids with boundary-aligned tails.

    ``tokenizer`` must expose ``encode(text)`` and
    ``encode(text, add_special_tokens=False)`` (the BaseTokenizer contract), and
    its underlying tokenizers object must expose ``pre_tokenizer``. Instances
    are created through :meth:`from_env`, which returns ``None`` whenever the
    cache is disabled or the tokenizer fails the compatibility probes, so the
    caller can keep a plain ``if cache is not None`` gate. When ``share_dir``
    is set, :meth:`start_share` binds this worker's peer socket; fetching and
    serving are strictly best-effort around the exact local semantics.
    """

    def __init__(
        self,
        tokenizer,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        shadow_rate: float = 0.0,
        share_dir: Optional[str] = None,
        share_timeout_s: float = _SHARE_FETCH_TIMEOUT_S,
        share_soft_window_s: float = _SHARE_SOFT_WINDOW_S,
    ):
        self._tokenizer = tokenizer
        self._max_bytes = max(1, int(max_bytes))
        self._shadow_rate = min(max(float(shadow_rate), 0.0), 1.0)
        self._entries: Dict[str, _Entry] = {}
        self._total_bytes = 0
        self._stamp = 0
        self._lock = threading.Lock()
        self._rng = random.Random()
        self._raw = self._resolve_raw(tokenizer)
        self._share_dir = share_dir
        self._share_timeout_s = max(0.005, float(share_timeout_s))
        self._share_soft_window_s = max(0.001, float(share_soft_window_s))
        self._share: Optional[_ShareServer] = None
        self._own_path: Optional[str] = None
        self._dir_conn: Optional[socket.socket] = None
        self._dir_lock = threading.Lock()
        self._dir_spawn_attempt = 0.0

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _resolve_raw(tokenizer):
        base = getattr(tokenizer, "tokenizer", tokenizer)
        for attr in ("backend_tokenizer", "_tokenizer"):
            raw = getattr(base, attr, None)
            if raw is not None and getattr(raw, "pre_tokenizer", None) is not None:
                return raw
        return None

    @classmethod
    def from_env(cls, tokenizer) -> Optional["IncrementalTokenizeCache"]:
        """Build the cache when enabled by env and compatible with tokenizer."""
        if os.environ.get(_ENV_ENABLE, "0").strip().lower() not in _TRUTHY:
            return None
        if tokenizer is None:
            return None
        try:
            max_bytes = int(os.environ.get(_ENV_MAX_BYTES, str(_DEFAULT_MAX_BYTES)))
            shadow_rate = float(os.environ.get(_ENV_SHADOW_RATE, "0"))
            share_timeout_s = (
                int(
                    os.environ.get(
                        _ENV_SHARE_TIMEOUT_MS,
                        str(int(_SHARE_FETCH_TIMEOUT_S * 1000)),
                    )
                )
                / 1000.0
            )
            share_soft_window_s = (
                int(
                    os.environ.get(
                        _ENV_SHARE_SOFT_WINDOW_MS,
                        str(int(_SHARE_SOFT_WINDOW_S * 1000)),
                    )
                )
                / 1000.0
            )
        except ValueError:
            logging.warning("tokenize cache: unparseable env value; staying disabled")
            return None
        share_dir = os.environ.get(_ENV_SHARE_DIR, _DEFAULT_SHARE_DIR).strip() or None
        cache = cls(
            tokenizer,
            max_bytes=max_bytes,
            shadow_rate=shadow_rate,
            share_dir=share_dir,
            share_timeout_s=share_timeout_s,
            share_soft_window_s=share_soft_window_s,
        )
        if not cache.compatible():
            logging.warning(
                "tokenize cache: tokenizer lacks a usable pre_tokenizer or adds "
                "special tokens on default encode; staying disabled"
            )
            return None
        cache.start_share()  # best-effort; failure leaves the cache local-only
        logging.info(
            "tokenize cache enabled: max_bytes=%d shadow_rate=%.3f share_dir=%s",
            cache._max_bytes,
            cache._shadow_rate,
            share_dir,
        )
        return cache

    def compatible(self) -> bool:
        """True when splice semantics are safe for this tokenizer.

        Requires a chunk-yielding pre_tokenizer and a default encode that adds
        no special tokens (so cached-prefix ids and tail ids concatenate
        without embedded specials).
        """
        if self._raw is None:
            return False
        pre = getattr(self._raw, "pre_tokenizer", None)
        if pre is None or not callable(getattr(pre, "pre_tokenize_str", None)):
            return False
        try:
            with_specials = list(self._tokenizer.encode(_COMPAT_PROBE))
            without = list(
                self._tokenizer.encode(_COMPAT_PROBE, add_special_tokens=False)
            )
        except TypeError:
            return False
        return with_specials == without

    # ------------------------------------------------------------ peer sharing

    def start_share(self) -> bool:
        """Bind this worker's share socket and start its serve thread.

        Safe to call once; returns False (local-only) when sharing is disabled
        or the socket cannot be bound. Socket names carry the pid plus a random
        suffix so multiple instances in one process (tests) never collide and
        files left behind by exited workers are unambiguous.
        """
        if self._share_dir is None or self._share is not None:
            return self._share is not None
        name = "{}{}{}{}".format(
            _SHARE_PREFIX,
            os.getpid(),
            uuid.uuid4().hex[:8],
            _SHARE_SUFFIX,
        )
        server = _ShareServer(self, os.path.join(self._share_dir, name))
        if not server.start():
            return False
        self._share = server
        self._own_path = server.path
        logging.info(f"tokenize cache: peer sharing socket at {self._own_path}")
        self._ensure_directory()  # best-effort; sweep fallback works without it
        return True

    def close_share(self) -> None:
        """Stop serving and remove this worker's socket file (idempotent).

        The directory daemon (if this worker spawned it) is deliberately left
        running for its peers; it retires on its idle-TTL when the pod drains.
        """
        self._dir_drop()
        if self._share is not None:
            self._share.close()
            self._share = None
        self._own_path = None

    # --------------------------------------------------------- directory daemon

    def _ensure_directory(self) -> None:
        """Spawn the pod directory daemon if absent (double-fork bind race).

        Rate-limited per worker. The candidate child detaches (setsid + second
        fork), closes inherited descriptors, and re-runs the bind race with
        retries so exactly one path-visible daemon survives; the intermediate
        child is waitpid-reaped here, and the daemon itself reparents to init
        (a losing candidate may linger as a zero-RSS zombie until pod init
        reaps it — bounded, one-time per worker start).
        """
        if self._share_dir is None:
            return
        now = time.monotonic()
        if now - self._dir_spawn_attempt < _DIR_RESPAWN_COOLDOWN_S:
            return
        self._dir_spawn_attempt = now
        dpath = os.path.join(self._share_dir, _DIR_SOCK_NAME)
        if _dir_probe_alive(dpath):
            return  # daemon alive
        _dir_spawn_candidate(dpath)

    def _dir_drop(self) -> None:
        if self._dir_conn is not None:
            try:
                self._dir_conn.close()
            except OSError:
                pass
            self._dir_conn = None

    def _dir_socket(self, timeout: float) -> Optional[socket.socket]:
        """Persistent daemon connection (lazy reconnect); None when down."""
        if self._dir_conn is not None:
            return self._dir_conn
        dpath = os.path.join(self._share_dir, _DIR_SOCK_NAME)  # type: ignore[arg-type]
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(max(0.01, timeout))
        try:
            sock.connect(dpath)
        except OSError:
            try:
                sock.close()
            except OSError:
                pass
            self._ensure_directory()  # rate-limited self-heal
            return None
        self._dir_conn = sock
        return sock

    def _dir_get(self, key: str, deadline: float) -> Optional[bytes]:
        """Storage-directory GET: entry frame payload, or None on NACK.

        Raises on transport errors (the caller drops the connection and falls
        back to the sweep); NACK means no announced entry exists yet. The
        payload is the same wire shape as a worker FETCH response, so it goes
        through the identical locally-verified parser.
        """
        with self._dir_lock:
            sock = self._dir_socket(
                min(_DIR_CONNECT_TIMEOUT_S, max(0.01, deadline - time.monotonic()))
            )
            if sock is None:
                raise ConnectionError("directory unavailable")
            sock.settimeout(
                min(_DIR_LOOKUP_TIMEOUT_S, max(0.01, deadline - time.monotonic()))
            )
            sock.sendall(struct.pack(">I", _REQ_LEN) + b"G" + key.encode("ascii"))
            header = _recv_exact(sock, 4)
            if header is None:
                raise ConnectionError("directory closed")
            (n,) = struct.unpack(">I", header)
            if n > _SHARE_MAX_FRAME:
                raise ValueError("directory frame too large")
            payload = _recv_exact(sock, n)
        if payload is None:
            raise ConnectionError("directory closed mid-frame")
        if payload[:1] == b"N":
            return None
        return payload

    def _dir_announce(self, key: str, cut: int, sha: str, ids_bytes: bytes) -> None:
        """Fire-and-forget storage PUT after any cut advance; never raises.

        Max-cut-wins is enforced daemon-side, so announce order races are
        harmless. A failed announce only means the directory briefly serves
        an older entry (or NACKs) — the sweep fallback and consumer-side
        hash verification keep results exact regardless.
        """
        if self._share_dir is None:
            return
        try:
            body = (
                b"P"
                + key.encode("ascii")
                + struct.pack(">I", cut)
                + sha.encode("ascii")
                + ids_bytes
            )
            with self._dir_lock:
                sock = self._dir_socket(_DIR_CONNECT_TIMEOUT_S)
                if sock is None:
                    return
                sock.settimeout(_DIR_ANNOUNCE_TIMEOUT_S)
                sock.sendall(struct.pack(">I", len(body)) + body)
        except Exception:
            self._dir_drop()

    def _discover_peers(self) -> List[str]:
        try:
            names = os.listdir(self._share_dir)  # type: ignore[arg-type]
        except OSError:
            return []
        peers = []
        for name in names:
            if not (name.startswith(_SHARE_PREFIX) and name.endswith(_SHARE_SUFFIX)):
                continue
            path = os.path.join(self._share_dir, name)  # type: ignore[arg-type]
            if path != self._own_path:
                peers.append(path)
        return peers

    def _serve_request(self, op: bytes, key: str) -> bytes:
        """Build the response frame for one peer LOOKUP/FETCH (serve thread)."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return b"N"
            if op == b"L":
                return b"H" + struct.pack(">I", entry.cut)
            return (
                b"E"
                + struct.pack(">I", entry.cut)
                + entry.prefix_sha.encode("ascii")
                + entry.ids.tobytes()
            )

    def _fetch_best_peer(self, text: str, key: str, local_cut: int) -> Optional[_Entry]:
        """Return a peer entry with a fresher cut than local, else None.

        Daemon-first: one GET to the storage directory (~0.5-1.5ms, served
        from daemon memory — no worker GIL in the fetch path) yields the
        freshest announced entry, verified locally like all peer state. A
        daemon NACK or any daemon/peer failure falls back to the broadcast
        sweep (LOOKUP with the adaptive soft window → best holder → FETCH,
        next-best on a NACK race). Never raises; reports the attempt RT.
        """
        begin = time.monotonic()
        deadline = begin + self._share_timeout_s
        entry = None
        try:
            payload = None
            try:
                payload = self._dir_get(key, deadline)
            except Exception as e:
                logging.debug(f"tokenize cache: directory get failed: {e}")
                self._dir_drop()
            if payload is not None:
                candidate = self._parse_entry(text, payload)
                if candidate is not None and candidate.cut > local_cut:
                    entry = candidate
            if entry is None:
                entry = self._sweep_best_peer(text, key, local_cut, deadline)
        except Exception as e:
            logging.debug(f"tokenize cache: peer lookup failed: {e}")
            entry = None
        try:
            self._report_gauge(
                "TOKENIZE_CACHE_FETCH_RT_METRIC", (time.monotonic() - begin) * 1000.0
            )
        except Exception:
            pass
        return entry

    def _sweep_best_peer(self, text: str, key: str, local_cut: int, deadline: float):
        """Broadcast-sweep fallback: best verified hold fresher than local."""
        holds = self._broadcast_lookups(key, len(text), deadline)
        holds = sorted(
            (hold for hold in holds if hold[0] > local_cut),
            key=lambda hold: -hold[0],
        )
        for _, path in holds:
            candidate = self._fetch_from_peer(path, key, text, deadline)
            if candidate is not None and candidate.cut > local_cut:
                return candidate
        return None

    def _broadcast_lookups(
        self, key: str, text_len: int, deadline: float
    ) -> List[Tuple[int, str]]:
        """Broadcast LOOKUP; collect (cut, path) holds with an early stop.

        Replies are at most 5 bytes, so waiting for every peer costs the
        slowest reply rather than the deadline, and a hold that is already
        fresh enough (tail within one append) ends the sweep immediately.
        Once any hold is known, a short soft window additionally bounds the
        sweep: splicing from the best (possibly stale) hold costs at most one
        extra append of tail, far less than waiting out a busy peer. With
        zero holds the sweep waits the full deadline. Stale socket files
        (exited workers) fail connect and are skipped without touching them.
        """
        peers = self._discover_peers()
        if not peers:
            return []
        request = struct.pack(">I", _REQ_LEN) + b"L" + key.encode("ascii")
        fresh_cut = int(text_len * _FRESH_FRAC)
        connected = self._connect_peers(peers)
        path_of: Dict[socket.socket, str] = {s: p for p, s in connected}
        outbox: Dict[socket.socket, bytes] = {s: request for _, s in connected}
        inbox: Dict[socket.socket, bytearray] = {}
        holds: List[Tuple[int, str]] = []
        sweep_start = time.monotonic()
        first_hold_at: Optional[float] = None
        try:
            while outbox or inbox:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                if holds:
                    # adaptive soft window (see _SHARE_SOFT_WINDOW_S): stop
                    # waiting out slow peers once the window elapsed
                    window = min(
                        max(
                            self._share_soft_window_s,
                            _SHARE_SOFT_WINDOW_K * (first_hold_at - sweep_start),
                        ),
                        _SHARE_SOFT_WINDOW_CAP_S,
                    )
                    elapsed = time.monotonic() - sweep_start
                    if elapsed >= window:
                        break
                    remaining = min(remaining, window - elapsed)
                readable, writable, _ = select.select(
                    list(inbox), list(outbox), [], remaining
                )
                for sock in writable:
                    self._send_lookup(sock, outbox, inbox)
                for sock in readable:
                    cut = self._read_hold(sock, inbox)
                    if cut is None:
                        continue
                    holds.append((cut, path_of[sock]))
                    if first_hold_at is None:
                        first_hold_at = time.monotonic()
                    if cut >= fresh_cut:
                        return holds
            return holds
        finally:
            for _, sock in connected:
                try:
                    sock.close()
                except OSError:
                    pass

    @staticmethod
    def _connect_peers(peers: List[str]) -> List[Tuple[str, socket.socket]]:
        """Non-blocking connect to every peer path; skip unreachable ones."""
        connected = []
        for path in peers:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.setblocking(False)
            err = sock.connect_ex(path)
            if err == 0 or err == errno.EINPROGRESS:
                connected.append((path, sock))
            else:
                sock.close()  # stale socket file or full backlog; skip
        return connected

    @staticmethod
    def _send_lookup(
        sock: socket.socket,
        outbox: Dict[socket.socket, bytes],
        inbox: Dict[socket.socket, bytearray],
    ) -> None:
        """Advance one non-blocking send; move to the readers when done."""
        so_error = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
        if so_error != 0:
            outbox.pop(sock)
            sock.close()
            return
        try:
            sent = sock.send(outbox[sock])
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            outbox.pop(sock)
            sock.close()
            return
        rest = outbox[sock][sent:]
        if rest:
            outbox[sock] = rest
        else:
            del outbox[sock]
            inbox[sock] = bytearray()

    @staticmethod
    def _read_hold(
        sock: socket.socket, inbox: Dict[socket.socket, bytearray]
    ) -> Optional[int]:
        """Consume one reader's bytes; return its hold cut, else None."""
        try:
            chunk = sock.recv(4096)
        except (BlockingIOError, InterruptedError):
            return None
        except OSError:
            inbox.pop(sock)
            sock.close()
            return None
        if not chunk:
            inbox.pop(sock)  # closed without responding
            sock.close()
            return None
        inbox[sock].extend(chunk)
        buf = inbox[sock]
        if len(buf) < 4:
            return None
        (n,) = struct.unpack(">I", bytes(buf[:4]))
        if n > _SHARE_MAX_FRAME:
            inbox.pop(sock)
            sock.close()
            return None
        frame_end = 4 + n
        if len(buf) < frame_end:
            return None
        payload = bytes(buf[4:frame_end])
        inbox.pop(sock)
        sock.close()
        if payload[:1] == b"H" and len(payload) == _RESP_HOLD_LEN:
            (cut,) = struct.unpack(">I", payload[1:])
            return cut
        return None  # NACK or malformed

    def _fetch_from_peer(
        self, path: str, key: str, text: str, deadline: float
    ) -> Optional[_Entry]:
        """Single-peer blocking FETCH; returns the LOCALLY verified entry."""
        remaining = deadline - time.monotonic()
        if remaining <= 0.005:
            return None
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(remaining)
        try:
            sock.connect(path)
            sock.sendall(struct.pack(">I", _REQ_LEN) + b"F" + key.encode("ascii"))
            header = _recv_exact(sock, 4)
            if header is None:
                return None
            (n,) = struct.unpack(">I", header)
            if n > _SHARE_MAX_FRAME:
                return None
            payload = _recv_exact(sock, n)
            if payload is None:
                return None
            return self._parse_entry(text, payload)
        except OSError:
            return None  # timeout, refused or reset: try the next-best holder
        finally:
            try:
                sock.close()
            except OSError:
                pass

    @staticmethod
    def _parse_entry(text: str, payload: bytes) -> Optional[_Entry]:
        """Decode and LOCALLY verify one FETCH response; None unless usable.

        The prefix hash is recomputed over ``text[:cut]`` here, so a corrupt,
        stale or forged entry is rejected exactly like a local hash mismatch
        and the caller falls back to the next-best holder or the miss path.
        """
        if not payload or payload[:1] == b"N":
            return None
        if payload[:1] != b"E" or len(payload) < 69:
            return None
        (cut,) = struct.unpack(">I", payload[1:5])
        prefix_sha = payload[5:69].decode("ascii", "replace")
        ids_bytes = payload[69:]
        if cut <= 0 or cut > len(text) or len(ids_bytes) % 4 != 0:
            return None
        if _sha256_hex(text[:cut]) != prefix_sha:
            logging.warning(
                "tokenize cache: rejecting peer entry with prefix-hash mismatch"
            )
            return None
        ids = array.array("i")
        ids.frombytes(ids_bytes)
        return _Entry(cut, ids, prefix_sha, 0)

    # ------------------------------------------------------------------- api

    def get_token_ids(self, text: str) -> List[int]:
        """Return the exact ``tokenizer.encode(text)`` ids, incrementally.

        The result is always identical to a full encode: cache state is only
        ever consulted after byte-level prefix verification (locally held or
        peer-fetched alike), and any anomaly falls back to (and returns) the
        full encode.
        """
        begin = time.monotonic()
        try:
            ids, key, shadow = self._lookup(text)
        except Exception as e:
            logging.warning(f"tokenize cache: full-encode fallback after {e}")
            try:
                self._report_acc("TOKENIZE_CACHE_MISS_QPS_METRIC")
            except Exception:
                pass
            ids, key, shadow = self._full_encode(text), None, False
        if shadow and key is not None:
            ids = self._shadow_verify(text, ids, key)
        try:
            self._report_gauge(
                "TOKENIZE_CACHE_RT_METRIC", (time.monotonic() - begin) * 1000.0
            )
        except Exception:
            pass
        return ids

    # --------------------------------------------------------------- internals

    def _full_encode(self, text: str) -> List[int]:
        return list(self._tokenizer.encode(text))

    def _encode_tail(self, text: str) -> List[int]:
        if not text:
            return []
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def _conservative_cut(self, text: str) -> int:
        """Largest pre-tokenizer chunk end strictly below len(text), else 0.

        Scanned from the right (offsets are monotone) and, for long texts,
        computed on a bounded trailing window: when the window's rightmost
        chunk end strictly below its own end is found, the window's final
        chunk is also the full text's final chunk (chunk locality — the same
        property the splice relies on), so that boundary IS the true
        conservative cut. An inconclusive window (its final chunk straddles
        the window start) falls back to the full-text scan, so the returned
        value is identical to the full scan by construction — while the
        per-call list materialization shrinks from O(len(text)) tuples
        (~600k on miss-path 1.7MB prompts; ~20-80ms GIL-held) to O(window)
        (~2-4ms), which the e40 bench identified as the dominant serve-thread
        starvation source.
        """
        n = len(text)
        if not text:
            return 0
        if n > _CUT_SUFFIX_CHARS:
            cut = self._suffix_cut(text, n)
            if cut is not None:
                return cut
        return self._full_scan_cut(text, n)

    def _suffix_cut(self, text: str, n: int) -> Optional[int]:
        """Conservative cut from a bounded trailing window, None if unsure.

        Chunk 0 of the window may straddle the window start (its end is then
        not a trustworthy boundary) and is skipped; if no later chunk ends
        strictly inside the window, the window is inconclusive.
        """
        start = n - _CUT_SUFFIX_CHARS
        suffix = text[start:]
        chunks = self._raw.pre_tokenizer.pre_tokenize_str(suffix)
        w = len(suffix)
        for _, offsets in reversed(chunks[1:]):
            end = offsets[1]
            if end < w:
                return n - w + end
        return None

    def _full_scan_cut(self, text: str, n: int) -> int:
        """Original whole-text right scan (fallback; also used for short texts)."""
        chunks = self._raw.pre_tokenizer.pre_tokenize_str(text)
        for _, offsets in reversed(chunks):
            if offsets[1] < n:
                return offsets[1]
        return 0

    @staticmethod
    def _entry_valid(entry: Optional[_Entry], text: str) -> bool:
        return (
            entry is not None
            and entry.cut <= len(text)
            and _sha256_hex(text[: entry.cut]) == entry.prefix_sha
        )

    def _lookup(self, text: str):
        """Return (ids, key, shadow_sampled) for ``text``.

        A FRESH local entry (cut within one append of the prompt end) wins
        outright without any peer traffic. Otherwise the peer-assisted
        resolution picks the freshest verified cut (directory daemon first,
        broadcast sweep as fallback), degrading to a stale local entry or the
        full-encode miss. Every cut advance is announced to the directory
        (outside the lock) so peers always discover the freshest holder.
        Without sharing this is exactly the original local-first behavior.
        """
        key = _sha256_hex(text[:_KEY_CHARS])
        hit = self._try_fresh_local(text, key)
        if hit is None:
            hit = self._try_peer_assisted(text, key)
        if hit is None:
            return self._miss(text, key), key, False
        ids, shadow, snap = hit
        self._dir_announce(key, snap[0], snap[1], snap[2])
        return ids, key, shadow

    @staticmethod
    def _snapshot_locked(entry: _Entry) -> Tuple[int, str, bytes]:
        """Copy-on-write announce payload taken under the cache lock."""
        return entry.cut, entry.prefix_sha, entry.ids.tobytes()

    def _try_fresh_local(self, text: str, key: str):
        """Fresh-local fast path → (ids, shadow, snapshot); None if not fresh."""
        with self._lock:
            local = self._entries.get(key)
            if self._entry_valid(local, text) and local.cut >= int(
                len(text) * _FRESH_FRAC
            ):
                ids, shadow = self._hit_locked(
                    text, local, "TOKENIZE_CACHE_HIT_LOCAL_QPS_METRIC"
                )
                return ids, shadow, self._snapshot_locked(local)
        return None

    def _try_peer_assisted(self, text: str, key: str):
        """Freshest verified cut wins (daemon/sweep fetch, or stale local)."""
        local_cut = -1
        with self._lock:
            local = self._entries.get(key)
            if self._entry_valid(local, text):
                local_cut = local.cut
        fetched: Optional[_Entry] = None
        if self._share_dir is not None:
            fetched = self._fetch_best_peer(text, key, local_cut)
        with self._lock:
            local_now = self._entries.get(key)
            if not self._entry_valid(local_now, text):
                local_now = None
            if fetched is not None and (
                local_now is None or fetched.cut > local_now.cut
            ):
                self._install_locked(key, fetched)
                ids, shadow = self._hit_locked(
                    text, fetched, "TOKENIZE_CACHE_HIT_SHARED_QPS_METRIC"
                )
                return ids, shadow, self._snapshot_locked(fetched)
            if local_now is not None:
                ids, shadow = self._hit_locked(
                    text, local_now, "TOKENIZE_CACHE_HIT_LOCAL_QPS_METRIC"
                )
                return ids, shadow, self._snapshot_locked(local_now)
        return None

    def _hit_locked(self, text: str, entry: _Entry, breakdown_metric: str):
        """Shared hit bookkeeping: splice, refresh LRU, sample, report."""
        ids = self._splice_locked(text, entry)
        self._stamp += 1
        entry.stamp = self._stamp
        shadow = self._shadow_rate > 0 and self._rng.random() < self._shadow_rate
        self._report_acc("TOKENIZE_CACHE_HIT_QPS_METRIC")
        self._report_acc(breakdown_metric)
        return ids, shadow

    def _splice_locked(self, text: str, entry: _Entry) -> List[int]:
        """Hit path: extend the cached prefix with boundary-aligned tails."""
        tail_start = entry.cut
        rel_cut = self._conservative_cut(text[tail_start:])
        new_cut = tail_start + rel_cut
        mid_ids = self._encode_tail(text[tail_start:new_cut])
        end_ids = self._encode_tail(text[new_cut:])
        ids = list(entry.ids)
        ids.extend(mid_ids)
        ids.extend(end_ids)
        if new_cut > entry.cut:
            old_nbytes = entry.nbytes
            entry.ids.extend(array.array("i", mid_ids))
            entry.cut = new_cut
            entry.prefix_sha = _sha256_hex(text[:new_cut])
            entry.nbytes = len(entry.ids) * 4 + 128
            # In-place growth must be accounted too, or the cap check in
            # _evict_locked never sees splice-driven expansion and the byte
            # cap is silently defeated (unbounded worker RSS growth).
            self._total_bytes += entry.nbytes - old_nbytes
            self._evict_locked()
        return ids

    def _install_locked(self, key: str, entry: _Entry) -> None:
        """Account one (new or fetched) entry against the cap, LRU-evicting."""
        old = self._entries.pop(key, None)
        if old is not None:
            self._total_bytes -= old.nbytes
        self._stamp += 1
        entry.stamp = self._stamp
        self._entries[key] = entry
        self._total_bytes += entry.nbytes
        self._evict_locked()

    def _miss(self, text: str, key: str) -> List[int]:
        """Miss path: exact full encode, then cache a verified prefix state."""
        ids = self._full_encode(text)
        self._report_acc("TOKENIZE_CACHE_MISS_QPS_METRIC")
        cut = self._conservative_cut(text)
        if cut <= 0:
            return ids
        ids_cut = self._encode_tail(text[:cut])
        sha = _sha256_hex(text[:cut])
        arr = array.array("i", ids_cut)
        with self._lock:
            self._install_locked(key, _Entry(cut, arr, sha, 0))
        self._dir_announce(key, cut, sha, arr.tobytes())
        return ids

    def _evict_locked(self) -> None:
        if self._total_bytes < 0:
            # Defensive resync: the accumulator must never drift negative; if
            # it ever did, recompute it from the live entries before evicting.
            self._total_bytes = sum(e.nbytes for e in self._entries.values())
        while self._total_bytes > self._max_bytes and len(self._entries) > 1:
            key = min(self._entries, key=lambda k: self._entries[k].stamp)
            self._total_bytes -= self._entries.pop(key).nbytes

    def _shadow_verify(self, text: str, ids: List[int], key: str) -> List[int]:
        full = self._full_encode(text)
        if full == ids:
            return ids
        logging.error(
            "tokenize cache: shadow mismatch detected; returning the full-encode "
            "result and dropping the session entry"
        )
        self._report_acc("TOKENIZE_CACHE_MISMATCH_QPS_METRIC")
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._total_bytes -= entry.nbytes
        return full

    # ---------------------------------------------------------------- metrics

    @staticmethod
    def _report_acc(name: str) -> None:
        if _kmonitor is None or AccMetrics is None:
            return
        try:
            _kmonitor.report(getattr(AccMetrics, name), 1)
        except Exception:
            pass

    @staticmethod
    def _report_gauge(name: str, value: float) -> None:
        if _kmonitor is None or GaugeMetrics is None:
            return
        try:
            _kmonitor.report(getattr(GaugeMetrics, name), value)
        except Exception:
            pass
