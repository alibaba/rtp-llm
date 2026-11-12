"""Per-user store that keeps resident-weight allocations alive by holding their file descriptors.

The engine exports each CUDA VMM allocation as a POSIX FD and hands it here; holding the FD keeps
the physical GPU memory alive after the engine exits, so a restarted engine can map it again.
This file uses only the standard library and is started as ``python -I store.py serve``: it never
loads CUDA or torch. The wire protocol and the published ledger live here as dataclasses and
enums, so both sides speak the same typed shapes instead of indexing bare dicts by string key.

    python store.py list             # what is held, per GPU and role
    python store.py drop [--slot S]  # release one record, or everything
    python store.py stop             # release everything and exit
"""

import argparse
import enum
import fcntl
import json
import os
import resource
import socket
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

PROTOCOL = 4  # 4: SlotRecord no longer carries source digests; stop older stores before upgrading
# SCM_RIGHTS carries at most 253 descriptors per message.
MAX_FDS_PER_FRAME = 200
# One hung client must not block the single-threaded store for every other GPU on this host.
SERVER_CONN_TIMEOUT_S = 30.0
_HEADER = struct.Struct("!Q")
_PEERCRED = struct.Struct("3i")

ScalarValue = Union[int, float, str, bool, None]
ScopeKey = Union[int, str]  # a layer index, or GLOBAL_SCOPE
GLOBAL_SCOPE = "global"
_LAYER_PREFIX = "layer:"

T = TypeVar("T")


class StoreError(RuntimeError):
    """A store-side failure: protocol mismatch, rejected request or undecodable message.

    Deliberately distinct from the plain RuntimeError the CUDA driver raises, so a caller can
    degrade gracefully when the store is gone or sent nonsense without also swallowing its own
    driver errors.
    """


class StoreOp(str, enum.Enum):
    PUBLISH = "publish"
    ATTACH = "attach"
    DROP = "drop"
    LIST = "list"
    EPLB_BEGIN = "eplb_begin"
    EPLB_COMMIT = "eplb_commit"
    RETAIN_FILE = "retain_file"
    STOP = "stop"
    FDS_CONTINUATION = "fds_continuation"  # a frame carrying only descriptors


class AttachStatus(str, enum.Enum):
    HIT = "hit"
    ABSENT = "absent"  # nothing published for this slot
    MISMATCH = "mismatch"  # published for another configuration
    EPLB_DIRTY = "eplb_dirty"  # left mid rearrangement


class WeightOrigin(str, enum.Enum):
    LOADED = "loaded"  # from the loader; a restart puts it back into the weight dict
    INIT = "init"  # from initialize(); a restart hands it back by name


def _decode(what: str, build: Callable[[], T]) -> T:
    """Anything the peer sent that does not decode is a store failure, not a local bug.

    Without this the engine's ``except (OSError, StoreError)`` around attach would not catch a
    damaged record, and a start that should fall back to the checkpoint would crash instead.
    """
    try:
        return build()
    except StoreError:
        raise
    except Exception as error:
        raise StoreError(f"resident weight store: damaged {what}: {error!r}") from None


def encode_scope(scope: ScopeKey) -> str:
    """JSON object keys must be strings; "layer:0" and "global" stay unambiguous either way."""
    return GLOBAL_SCOPE if scope == GLOBAL_SCOPE else f"{_LAYER_PREFIX}{int(scope)}"


def decode_scope(text: object) -> ScopeKey:
    if text == GLOBAL_SCOPE:
        return GLOBAL_SCOPE
    if isinstance(text, str) and text.startswith(_LAYER_PREFIX):
        index = text[len(_LAYER_PREFIX) :]
        if index.isdigit():
            return int(index)
    raise StoreError(f"resident weight store: {text!r} is not a scope key")


@dataclass
class StorageSpec:
    """One published VMM-backed storage: where it lives and what it held when published."""

    allocation_index: int
    offset_bytes: int
    nbytes: int
    checksum: str


@dataclass
class TensorSpec:
    """One tensor view over a StorageSpec, named by the scope and name the model knows it by."""

    scope: ScopeKey
    name: str
    origin: WeightOrigin
    storage_index: int  # index into Manifest.storages; -1 for a zero-byte tensor
    dtype: str
    shape: List[int]
    strides: List[int]
    storage_offset: int
    is_inference: bool

    def to_dict(self) -> dict:
        return dict(
            asdict(self), scope=encode_scope(self.scope), origin=self.origin.value
        )

    @classmethod
    def from_dict(cls, data: dict) -> "TensorSpec":
        return _decode(
            "tensor",
            lambda: cls(
                **dict(
                    data,
                    scope=decode_scope(data["scope"]),
                    origin=WeightOrigin(data["origin"]),
                )
            ),
        )


@dataclass
class Manifest:
    """The whole published ledger: every resident tensor in one flat table.

    ``transform_returned_tensors`` answers, per initialize()-time transform group, whether it
    returned tensors at all -- a degenerate transform returns nothing -- and is kept apart from
    ``init_scalars`` so a group key can never collide with a scalar a model chose to cache
    under the same name.
    """

    allocation_sizes: List[int]
    storages: List[StorageSpec]
    tensors: List[TensorSpec]
    transform_returned_tensors: Dict[ScopeKey, Dict[str, bool]] = field(
        default_factory=dict
    )
    init_scalars: Dict[ScopeKey, Dict[str, ScalarValue]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "allocation_sizes": list(self.allocation_sizes),
            "storages": [asdict(s) for s in self.storages],
            "tensors": [t.to_dict() for t in self.tensors],
            "transform_returned_tensors": {
                encode_scope(k): dict(v)
                for k, v in self.transform_returned_tensors.items()
            },
            "init_scalars": {
                encode_scope(k): dict(v) for k, v in self.init_scalars.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        return _decode(
            "manifest",
            lambda: cls(
                allocation_sizes=[int(size) for size in data["allocation_sizes"]],
                storages=[StorageSpec(**s) for s in data["storages"]],
                tensors=[TensorSpec.from_dict(t) for t in data["tensors"]],
                transform_returned_tensors={
                    decode_scope(k): dict(v)
                    for k, v in data.get("transform_returned_tensors", {}).items()
                },
                init_scalars={
                    decode_scope(k): dict(v)
                    for k, v in data.get("init_scalars", {}).items()
                },
            ),
        )


@dataclass
class SlotRecord:
    """What the store keeps for one slot: "<GPU UUID>/<role>".

    The engine supplies the configuration fingerprint and manifest; the store fills
    ``published_at`` and owns the EPLB fields. Users own weight/code compatibility.
    """

    fingerprint: str  # SHA-256 of fingerprint_text; attach compares it
    fingerprint_text: str  # "field: value" lines, returned on a mismatch
    summary: str  # model type and checkpoint, shown by `list`
    total_bytes: int  # sum of the allocation sizes
    publisher_pid: int
    manifest: Manifest
    published_at: float = 0.0
    eplb_dirty: bool = False  # an EPLB rearrangement began and has not committed
    eplb_placement: Dict[str, List[int]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(asdict(self), manifest=self.manifest.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "SlotRecord":
        return _decode(
            "record",
            lambda: cls(**dict(data, manifest=Manifest.from_dict(data["manifest"]))),
        )


@dataclass
class SlotSummary:
    """What `list` reports for one slot; the manifest is deliberately left out of it."""

    total_bytes: int
    allocations: int
    publisher_pid: int
    published_at: float
    summary: str
    eplb_dirty: bool


@dataclass
class StoreRequest:
    """One request. ``slot_key`` names a model role on one GPU: "<GPU UUID>/<role>"."""

    op: StoreOp
    slot_key: Optional[str] = None
    fingerprint: Optional[str] = None  # attach
    record: Optional[SlotRecord] = None  # publish
    eplb_placement: Optional[Dict[str, List[int]]] = None  # eplb_commit
    file_path: Optional[str] = None  # retain_host_file: the host table
    lock_path: Optional[str] = None  # retain_host_file: its creation lock

    def to_message(self) -> dict:
        message = {
            key: value
            for key, value in asdict(self).items()
            if value is not None and key != "record"
        }
        message["op"] = self.op.value
        if self.record is not None:
            message["record"] = self.record.to_dict()
        return message

    @classmethod
    def from_message(cls, message: dict) -> "StoreRequest":
        op = message.get("op")
        if op not in _OP_VALUES:
            raise StoreError(f"resident weight store: unknown op {op!r}")
        fields = {k: v for k, v in message.items() if k not in ("proto", "nfds", "op")}
        return _decode(
            "request",
            lambda: cls(
                op=StoreOp(op),
                **dict(
                    fields,
                    record=(
                        SlotRecord.from_dict(fields["record"])
                        if fields.get("record") is not None
                        else None
                    ),
                ),
            ),
        )


@dataclass
class StoreReply:
    """The store's answer: each op fills only the fields it has, and ``error`` replaces them all.

    A reply without an error is an accepted request; there is no separate "ok" flag to keep in
    step with the fields around it.
    """

    error: Optional[str] = None
    status: Optional[AttachStatus] = None  # attach
    record: Optional[SlotRecord] = None  # attach hit, or dropping one slot
    published_fingerprint_text: Optional[str] = None  # attach mismatch
    slots: Optional[Dict[str, SlotSummary]] = None  # list
    retained_file_sizes: Optional[Dict[str, int]] = None  # list
    retained: Optional[bool] = None  # retain_host_file

    def to_message(self) -> dict:
        if self.error is not None:
            return {"error": self.error}
        fields = {
            "status": None if self.status is None else self.status.value,
            "record": None if self.record is None else self.record.to_dict(),
            "published_fingerprint_text": self.published_fingerprint_text,
            "slots": (
                None
                if self.slots is None
                else {key: asdict(slot) for key, slot in self.slots.items()}
            ),
            "retained_file_sizes": self.retained_file_sizes,
            "retained": self.retained,
        }
        return {key: value for key, value in fields.items() if value is not None}

    @classmethod
    def from_message(cls, message: dict) -> "StoreReply":
        if "error" in message:
            return cls(error=str(message["error"]))
        return _decode(
            "reply",
            lambda: cls(
                status=(
                    AttachStatus(message["status"]) if "status" in message else None
                ),
                record=(
                    SlotRecord.from_dict(message["record"])
                    if "record" in message
                    else None
                ),
                published_fingerprint_text=message.get("published_fingerprint_text"),
                slots=(
                    {key: SlotSummary(**slot) for key, slot in message["slots"].items()}
                    if "slots" in message
                    else None
                ),
                retained_file_sizes=message.get("retained_file_sizes"),
                retained=message.get("retained"),
            ),
        )


_OP_VALUES = frozenset(op.value for op in StoreOp)


@dataclass
class _PublishedSlot:
    record: SlotRecord
    fds: List[int]  # one per allocation, in Manifest.allocation_sizes order


@dataclass
class _RetainedFile:
    fd: int  # holds LOCK_SH on the host table file
    lock_path: str


@dataclass
class AttachResult:
    """The answer to an attach: a hit carries the record and one FD per allocation.

    On a mismatch the store sends ``published_fingerprint_text`` and deliberately **no record**:
    the published manifest may come from a different field layout, and decoding it on the very
    path that reports the mismatch would turn a clear diagnosis into a parse failure. That escape
    hatch does not replace bumping PROTOCOL -- any wire or ledger field change must.
    """

    status: AttachStatus
    record: Optional[SlotRecord] = None
    allocation_fds: List[int] = field(default_factory=list)
    published_fingerprint_text: str = ""


def default_store_name() -> str:
    return (
        os.environ.get("RTP_LLM_RESIDENT_STORE")
        or f"rtp_llm_resident_weights.{os.getuid()}"
    )


def _address(name: str) -> str:
    # Abstract socket: no file to go stale after SIGKILL, no path length limit from TMPDIR.
    return "\0" + name


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    while n:
        chunk = sock.recv(min(n, 1 << 20))
        if not chunk:
            raise ConnectionError("resident weight store connection closed")
        chunks.append(chunk)
        n -= len(chunk)
    return b"".join(chunks)


def _send_frame(sock: socket.socket, obj: dict, fds=()) -> None:
    payload = json.dumps(obj).encode()
    header = _HEADER.pack(len(payload))
    if fds:
        socket.send_fds(sock, [header], list(fds))
    else:
        sock.sendall(header)
    sock.sendall(payload)


def _close(fds) -> None:
    for fd in fds:
        os.close(fd)


def _recv_frame(sock: socket.socket):
    data, fds, _, _ = socket.recv_fds(sock, _HEADER.size, MAX_FDS_PER_FRAME)
    try:
        if not data:
            raise ConnectionError("resident weight store connection closed")
        if len(data) < _HEADER.size:
            data += _recv_exact(sock, _HEADER.size - len(data))
        (n,) = _HEADER.unpack(data)
        payload = _recv_exact(sock, n)
        return _decode("frame", lambda: json.loads(payload)), list(fds)
    except BaseException:
        _close(fds)
        raise


def send_message(sock: socket.socket, obj: dict, fds=()) -> None:
    fds = list(fds)
    _send_frame(sock, dict(obj, nfds=len(fds)), fds[:MAX_FDS_PER_FRAME])
    for start in range(MAX_FDS_PER_FRAME, len(fds), MAX_FDS_PER_FRAME):
        _send_frame(
            sock,
            {"op": StoreOp.FDS_CONTINUATION.value},
            fds[start : start + MAX_FDS_PER_FRAME],
        )


def recv_message(sock: socket.socket):
    obj, fds = _recv_frame(sock)
    try:
        while len(fds) < obj.get("nfds", 0):
            _, more = _recv_frame(sock)
            fds += more
    except BaseException:
        # A publisher killed between frames: its allocations must not stay pinned here.
        _close(fds)
        raise
    return obj, fds


def request(
    name: str, req: StoreRequest, fds=(), timeout: float = 600.0
) -> Tuple[StoreReply, List[int]]:
    """One request/reply exchange.

    Raises ConnectionRefusedError (an OSError) when no store is running, and StoreError when the
    store rejects the request or sends something undecodable.
    """
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(_address(name))
        send_message(sock, {"proto": PROTOCOL, **req.to_message()}, fds)
        message, reply_fds = recv_message(sock)
    try:
        reply = StoreReply.from_message(message)
        if reply.error is not None:
            raise StoreError(f"resident weight store: {reply.error}")
    except BaseException:
        _close(reply_fds)
        raise
    return reply, reply_fds


# ---- typed client API ------------------------------------------------------------------


def publish_slot(
    store_name: str, slot_key: str, record: SlotRecord, fds: Sequence[int]
) -> None:
    request(
        store_name, StoreRequest(StoreOp.PUBLISH, slot_key=slot_key, record=record), fds
    )


def attach_slot(store_name: str, slot_key: str, fingerprint: str) -> AttachResult:
    reply, fds = request(
        store_name,
        StoreRequest(StoreOp.ATTACH, slot_key=slot_key, fingerprint=fingerprint),
    )
    if reply.status is AttachStatus.HIT and reply.record is not None:
        return AttachResult(
            AttachStatus.HIT, record=reply.record, allocation_fds=list(fds)
        )
    _close(fds)
    if reply.status is None or reply.status is AttachStatus.HIT:
        raise StoreError(
            f"resident weight store: malformed attach reply for {slot_key}"
        )
    return AttachResult(
        reply.status,
        published_fingerprint_text=reply.published_fingerprint_text or "",
    )


def drop_slot(store_name: str, slot_key: Optional[str] = None) -> Optional[SlotRecord]:
    """Release one slot (or every slot) and return the record that was dropped, if any."""
    reply, _ = request(store_name, StoreRequest(StoreOp.DROP, slot_key=slot_key))
    return reply.record


def eplb_begin(store_name: str, slot_key: str) -> None:
    """Mark a rearrangement in flight: until it commits, the record describes stale memory."""
    request(store_name, StoreRequest(StoreOp.EPLB_BEGIN, slot_key=slot_key))


def eplb_commit(
    store_name: str, slot_key: str, eplb_placement: Dict[str, List[int]]
) -> None:
    """End the rearrangement and record the placement the resident memory now holds."""
    request(
        store_name,
        StoreRequest(
            StoreOp.EPLB_COMMIT, slot_key=slot_key, eplb_placement=eplb_placement
        ),
    )


def retain_host_file(store_name: str, file_path: str, lock_path: str) -> bool:
    """Ask the store to hold LOCK_SH on a host table so "the last reader deletes it" cannot fire."""
    reply, _ = request(
        store_name,
        StoreRequest(StoreOp.RETAIN_FILE, file_path=file_path, lock_path=lock_path),
    )
    return bool(reply.retained)


def list_store(store_name: str) -> StoreReply:
    """What the store holds right now, for the operator-facing CLI and for tests."""
    reply, _ = request(store_name, StoreRequest(StoreOp.LIST))
    return reply


def stop_store(store_name: str) -> None:
    request(store_name, StoreRequest(StoreOp.STOP))


def ensure_running(
    name: str, timeout: float = 30.0, owner_pid: Optional[int] = None
) -> None:
    """Start a detached store for ``name`` unless one is already serving."""
    try:
        list_store(name)
        return
    except ConnectionRefusedError:
        pass
    log_path = f"/tmp/{name}.log"
    with open(log_path, "a") as log:
        # Double-fork so the store is reparented, not visible among engine descendants.
        launcher = subprocess.Popen(
            [
                sys.executable,
                "-I",
                os.path.abspath(__file__),
                "serve",
                "--name",
                name,
                "--daemonize",
                *(["--owner-pid", str(owner_pid)] if owner_pid is not None else []),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        # Only the short-lived launcher is our child; reap it before checking readiness.
        try:
            launcher.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise StoreError(
                f"resident weight store launcher timed out; see {log_path}"
            ) from error
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            list_store(name)
            return
        except ConnectionRefusedError:
            time.sleep(0.1)
    raise StoreError(f"resident weight store {name} did not start; see {log_path}")


class _Store:
    def __init__(self):
        self.slots: Dict[str, _PublishedSlot] = {}
        self.retained_files: Dict[str, _RetainedFile] = {}

    def handle(self, message: dict, fds: list) -> Tuple[StoreReply, List[int]]:
        """Takes ownership of ``fds``: a successful publish keeps them, every other path closes them."""
        kept = False
        try:
            if message.get("proto") != PROTOCOL:
                return (
                    StoreReply(
                        error=f"protocol {message.get('proto')} != store protocol "
                        f"{PROTOCOL}; run `store.py stop` and restart"
                    ),
                    [],
                )
            req = StoreRequest.from_message(message)
            if req.op is StoreOp.PUBLISH:
                if req.slot_key is None or req.record is None:
                    raise StoreError("publish needs a slot_key and a record")
                self._publish(req.slot_key, req.record, fds)
                kept = True
                return StoreReply(), []
            handler = _HANDLERS.get(req.op)
            if handler is None:
                return StoreReply(error=f"op {req.op.value!r} takes no answer here"), []
            return handler(self, req)
        except StoreError as error:
            return StoreReply(error=str(error)), []
        finally:
            if not kept:
                _close(fds)

    def _publish(self, slot_key: str, record: SlotRecord, fds: list) -> None:
        record.published_at = time.time()
        record.eplb_dirty = False
        self._release(slot_key)
        self.slots[slot_key] = _PublishedSlot(record, fds)

    def _attach(self, req: StoreRequest) -> Tuple[StoreReply, List[int]]:
        held = self.slots.get(req.slot_key)
        if held is None:
            return StoreReply(status=AttachStatus.ABSENT), []
        if held.record.eplb_dirty:
            return StoreReply(status=AttachStatus.EPLB_DIRTY), []
        if held.record.fingerprint != req.fingerprint:
            # No record here on purpose: see AttachResult.
            return (
                StoreReply(
                    status=AttachStatus.MISMATCH,
                    published_fingerprint_text=held.record.fingerprint_text,
                ),
                [],
            )
        # send_fds duplicates the descriptors for the receiver; the store keeps its own.
        return StoreReply(status=AttachStatus.HIT, record=held.record), held.fds

    def _drop(self, req: StoreRequest) -> Tuple[StoreReply, List[int]]:
        if not req.slot_key:
            self._release_all()
            return StoreReply(), []
        # Verify mode compares a fresh load against the record it took out.
        return StoreReply(record=self._release(req.slot_key)), []

    def _list(self, req: StoreRequest) -> Tuple[StoreReply, List[int]]:
        return (
            StoreReply(
                slots={
                    slot_key: SlotSummary(
                        total_bytes=held.record.total_bytes,
                        allocations=len(held.fds),
                        publisher_pid=held.record.publisher_pid,
                        published_at=held.record.published_at,
                        summary=held.record.summary,
                        eplb_dirty=held.record.eplb_dirty,
                    )
                    for slot_key, held in self.slots.items()
                },
                retained_file_sizes={
                    path: os.fstat(entry.fd).st_size
                    for path, entry in self.retained_files.items()
                },
            ),
            [],
        )

    def _eplb_begin(self, req: StoreRequest) -> Tuple[StoreReply, List[int]]:
        # A record dropped while its engine serves has nothing left to protect.
        slot = self.slots.get(req.slot_key)
        if slot is not None:
            slot.record.eplb_dirty = True
        return StoreReply(), []

    def _eplb_commit(self, req: StoreRequest) -> Tuple[StoreReply, List[int]]:
        slot = self.slots.get(req.slot_key)
        if slot is not None:
            slot.record.eplb_placement.update(req.eplb_placement or {})
            slot.record.eplb_dirty = False
        return StoreReply(), []

    def _retain(self, req: StoreRequest) -> Tuple[StoreReply, List[int]]:
        return (
            StoreReply(retained=self._retain_host_file(req.file_path, req.lock_path)),
            [],
        )

    def _stop(self, req: StoreRequest) -> Tuple[StoreReply, List[int]]:
        return StoreReply(), []

    def _release(self, slot_key: str) -> Optional[SlotRecord]:
        slot = self.slots.pop(slot_key, None)
        if slot is None:
            return None
        _close(slot.fds)
        print(f"{time.ctime()} dropped {slot_key}", flush=True)
        return slot.record

    def _release_all(self) -> None:
        for slot_key in list(self.slots):
            self._release(slot_key)
        for path, entry in self.retained_files.items():
            fcntl.flock(entry.fd, fcntl.LOCK_UN)
            os.close(entry.fd)
            _delete_unused(path, entry.lock_path)
        self.retained_files.clear()

    def _retain_host_file(self, file_path: str, lock_path: str) -> bool:
        if file_path in self.retained_files:
            return True
        try:
            fd = os.open(file_path, os.O_RDONLY)
        except OSError:
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self.retained_files[file_path] = _RetainedFile(fd, lock_path)
        return True


_HANDLERS: Dict[
    StoreOp, Callable[[_Store, StoreRequest], Tuple[StoreReply, List[int]]]
] = {
    StoreOp.ATTACH: _Store._attach,
    StoreOp.DROP: _Store._drop,
    StoreOp.LIST: _Store._list,
    StoreOp.EPLB_BEGIN: _Store._eplb_begin,
    StoreOp.EPLB_COMMIT: _Store._eplb_commit,
    StoreOp.RETAIN_FILE: _Store._retain,
    StoreOp.STOP: _Store._stop,
}


def _delete_unused(path: str, lock: str) -> None:
    """Delete a host table nobody maps, under its creation lock, as its owner would."""
    lock_fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        # A busy lock means an engine is creating or releasing the table; it owns the cleanup.
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # readers hold LOCK_SH
        finally:
            os.close(fd)
        os.unlink(path)
        os.unlink(lock)
    except OSError:
        pass
    finally:
        os.close(lock_fd)


def serve(
    name: str,
    conn_timeout: float = SERVER_CONN_TIMEOUT_S,
    owner_pid: Optional[int] = None,
    daemonize: bool = False,
) -> None:
    if daemonize:
        if os.fork():
            return
        os.setsid()
        if os.fork():
            os._exit(0)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(_address(name))
    except OSError as error:
        print(f"{time.ctime()} store {name} not started: {error}", flush=True)
        return  # another store already serves this name
    server.listen(64)
    owner_identity = _process_identity(owner_pid) if owner_pid is not None else None
    if owner_pid is not None:
        server.settimeout(1.0)
    print(f"{time.ctime()} store {name} serving, pid {os.getpid()}", flush=True)
    store = _Store()
    uid = os.getuid()
    while True:
        if owner_pid is not None and (
            owner_identity is None or _process_identity(owner_pid) != owner_identity
        ):
            server.close()
            store._release_all()
            return
        try:
            conn, _ = server.accept()
        except socket.timeout:
            continue
        with conn:
            # This store is single-threaded and shared by every GPU of this user: a client that
            # connects and then hangs mid-frame would otherwise block all of them forever.
            conn.settimeout(conn_timeout)
            _, peer_uid, _ = _PEERCRED.unpack(
                conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _PEERCRED.size)
            )
            if peer_uid != uid:
                continue
            stopping = False
            try:
                message, fds = recv_message(conn)
                stopping = message.get("op") == StoreOp.STOP.value
                reply, reply_fds = store.handle(message, fds)
                if stopping:
                    # Stop accepting first, so a client that sees the reply also sees us gone.
                    server.close()
                    store._release_all()
                send_message(conn, reply.to_message(), reply_fds)
            except Exception as error:  # one bad client must not take the store down
                print(f"{time.ctime()} request failed: {error!r}", flush=True)
                continue
            if stopping:
                print(f"{time.ctime()} store {name} stopped", flush=True)
                return


def _process_identity(pid: int) -> Optional[str]:
    """Owner start time, excluding exited zombies; PID reuse must not retain a test store."""
    try:
        with open(f"/proc/{pid}/stat") as process:
            fields = process.read().rsplit(")", 1)[1].split()
        return fields[19] if fields[0] != "Z" else None
    except FileNotFoundError:
        return None


def _print_list(reply: StoreReply) -> None:
    slots, files = reply.slots or {}, reply.retained_file_sizes or {}
    if not slots and not files:
        print("store is empty")
    for slot_key, slot in sorted(slots.items()):
        gib = slot.total_bytes / (1 << 30)
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(slot.published_at))
        eplb = " eplb_dirty" if slot.eplb_dirty else ""
        print(
            f"{slot_key}  {gib:.2f} GiB  {slot.allocations} allocations  "
            f"published {when} by pid {slot.publisher_pid}{eplb}"
        )
        print(f"    {slot.summary}")
    for path, size in sorted(files.items()):
        print(f"host table {path}  {size / (1 << 30):.2f} GiB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["serve", "list", "drop", "stop"])
    parser.add_argument("--name", default=default_store_name())
    parser.add_argument("--slot", help="drop only this record (default: everything)")
    parser.add_argument("--owner-pid", type=int)
    parser.add_argument("--daemonize", action="store_true")
    args = parser.parse_args()
    if args.command == "serve":
        serve(args.name, owner_pid=args.owner_pid, daemonize=args.daemonize)
        return
    try:
        if args.command == "list":
            _print_list(list_store(args.name))
        elif args.command == "drop":
            drop_slot(args.name, args.slot)
        else:
            stop_store(args.name)
    except (ConnectionRefusedError, FileNotFoundError):
        print(f"no store is running under {args.name}")
        sys.exit(0)


if __name__ == "__main__":
    main()
