"""Per-user store that keeps resident-weight allocations alive by holding their file descriptors.

The engine exports CUDA VMM allocations and sealed CPU memfds. Holding their FDs keeps
the backing alive after the engine exits, so a restarted engine can map it again.
This file uses only the standard library and is started as ``python -I store.py serve``: it never
loads CUDA or torch. The wire protocol and the published ledger live here as dataclasses and
enums, so both sides speak the same typed shapes instead of indexing bare dicts by string key.

    python store.py list                         # what is held, per GPU and role
    python store.py config [flags]               # update idle/offload settings live
    python store.py offload [--slot S]           # move idle GPU record(s) to host memory now
    python store.py keepalive [--slot S]         # reset current stage's idle timer (GPU or host)
    python store.py drop [--slot S]              # release one record, or everything
    python store.py stop                         # release everything and exit

Where a record's GPU allocations live is the store's decision alone. Two decoupled idle stages
govern retention: an unheld GPU record moves to sealed host memfds after ``offload_after_s``,
and once in host memory (or directly on GPU when ``offload_after_s == -1``) it is dropped after
``idle_ttl_s``. ``-1`` disables either idle timer. A short-lived mover process (mover.py, the
only part that loads CUDA) copies in either direction; the store only swaps FDs.
"""

import argparse
import enum
import fcntl
import json
import math
import os
import resource
import select
import socket
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, TypeVar, Union

# Records follow decoupled GPU/host idle stages and are held over a connection; stop older stores first.
PROTOCOL = 8
# SCM_RIGHTS carries at most 253 descriptors per message.
MAX_FDS_PER_FRAME = 200
# One hung client must not block the single-threaded store for every other GPU on this host.
SERVER_CONN_TIMEOUT_S = 30.0
# Hard bound on one mover's wall time (either direction) and on each mover control exchange.
DEFAULT_MOVE_TIMEOUT_S = 600.0
MOVER_IO_TIMEOUT_S = 10.0
REAP_GRACE_S = 10.0
DISABLED_TIMER = -1.0
DEFAULT_HOST_RESERVE_RATIO = 0.2
DEFAULT_HOST_RESERVE_MIN_GB = 0.0
MAX_HOST_RESERVE_MIN_GB = 1e9
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
    CONFIG = "config"
    OFFLOAD = "offload"
    KEEPALIVE = "keepalive"
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
    MOVING = "moving"  # moving back to the GPU from host memory; ask again shortly
    RESTORE_FAILED = (
        "restore_failed"  # the move back failed; the record stays in host memory
    )


class MemoryKind(str, enum.Enum):
    CPU = "cpu"
    CUDA = "cuda"


@dataclass
class AllocationSpec:
    """Backing kind, allocated bytes and host access intent, never process-local pointers."""

    kind: MemoryKind
    size_bytes: int
    cuda_access_required: bool = False

    def to_dict(self) -> dict:
        return dict(asdict(self), kind=self.kind.value)

    @classmethod
    def from_dict(cls, data: dict) -> "AllocationSpec":
        return cls(**dict(data, kind=MemoryKind(data["kind"])))


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
    """One published resident storage: where it lives and what it held when published."""

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
    device: MemoryKind

    def to_dict(self) -> dict:
        return dict(
            asdict(self),
            scope=encode_scope(self.scope),
            origin=self.origin.value,
            device=self.device.value,
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
                    device=MemoryKind(data["device"]),
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

    allocations: List[AllocationSpec]
    storages: List[StorageSpec]
    tensors: List[TensorSpec]
    transform_returned_tensors: Dict[ScopeKey, Dict[str, bool]] = field(
        default_factory=dict
    )
    init_scalars: Dict[ScopeKey, Dict[str, ScalarValue]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "allocations": [a.to_dict() for a in self.allocations],
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
                allocations=[AllocationSpec.from_dict(a) for a in data["allocations"]],
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
    # GPU memory the publishing start's initialize() added, net of what it freed: measured from
    # handing the loaded model to the engine to publishing. A restart holds it before KV sizing.
    initialize_gpu_growth_bytes: int = 0
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
    cpu_bytes: int
    gpu_bytes: int
    allocations: int
    publisher_pid: int
    published_at: float
    summary: str
    eplb_dirty: bool
    on_host: bool = False  # its GPU allocations are held as host memfds
    holders: int = 0
    idle_s: Optional[float] = (
        None  # since entering the current idle stage; None while held
    )
    moving: Optional[str] = None  # "to_host" or "to_gpu" while a mover runs
    offload_queued: bool = False  # waiting behind an active offload
    last_offload_error: Optional[str] = None  # recent manual offload refusal/failure


@dataclass
class StoreSettings:
    """Store runtime settings; -1 disables an idle timer."""

    offload_after_s: float = DISABLED_TIMER
    idle_ttl_s: float = DISABLED_TIMER
    move_timeout_s: float = DEFAULT_MOVE_TIMEOUT_S
    host_reserve_ratio: float = DEFAULT_HOST_RESERVE_RATIO
    host_reserve_min_gb: float = DEFAULT_HOST_RESERVE_MIN_GB


def _validate_timer(name: str, value: float) -> float:
    if value == DISABLED_TIMER:
        return DISABLED_TIMER
    if not math.isfinite(value) or value <= 0:
        raise StoreError(f"{name} must be > 0 or -1 (disabled), got {value}")
    return float(value)


def validate_settings(settings: StoreSettings) -> StoreSettings:
    """Reject nonsensical timer, timeout or host-reserve values before they take effect."""
    offload = _validate_timer("offload_after_s", settings.offload_after_s)
    ttl = _validate_timer("idle_ttl_s", settings.idle_ttl_s)
    if not math.isfinite(settings.move_timeout_s) or settings.move_timeout_s <= 0:
        raise StoreError(
            f"move_timeout_s must be finite and > 0, got {settings.move_timeout_s}"
        )
    if not math.isfinite(settings.host_reserve_ratio) or not (
        0.0 <= settings.host_reserve_ratio < 1.0
    ):
        raise StoreError(
            f"host_reserve_ratio must be in [0, 1), got {settings.host_reserve_ratio}"
        )
    if not math.isfinite(settings.host_reserve_min_gb) or not (
        0.0 <= settings.host_reserve_min_gb <= MAX_HOST_RESERVE_MIN_GB
    ):
        raise StoreError(
            f"host_reserve_min_gb must be in [0, {MAX_HOST_RESERVE_MIN_GB:g}], "
            f"got {settings.host_reserve_min_gb}"
        )
    return StoreSettings(
        offload_after_s=offload,
        idle_ttl_s=ttl,
        move_timeout_s=float(settings.move_timeout_s),
        host_reserve_ratio=float(settings.host_reserve_ratio),
        host_reserve_min_gb=float(settings.host_reserve_min_gb),
    )


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
    hold: bool = False  # publish / attach hit: keep the request connection as the hold
    expected_generation: Optional[str] = None  # drop: conditional generation guard
    offload_after_s: Optional[float] = None  # config
    idle_ttl_s: Optional[float] = None  # config
    move_timeout_s: Optional[float] = None  # config
    host_reserve_ratio: Optional[float] = None  # config
    host_reserve_min_gb: Optional[float] = None  # config

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
    restore_error: Optional[str] = None  # attach restore_failed
    slots: Optional[Dict[str, SlotSummary]] = None  # list
    settings: Optional[StoreSettings] = None  # list / config / attach moving
    retained_file_sizes: Optional[Dict[str, int]] = None  # list
    retained: Optional[bool] = None  # retain_host_file
    generation: Optional[str] = None  # publish / attach
    affected: Optional[Dict[str, str]] = None  # offload / keepalive
    message: Optional[str] = None  # offload / keepalive

    def to_message(self) -> dict:
        if self.error is not None:
            return {"error": self.error}
        fields = {
            "status": None if self.status is None else self.status.value,
            "record": None if self.record is None else self.record.to_dict(),
            "published_fingerprint_text": self.published_fingerprint_text,
            "restore_error": self.restore_error,
            "settings": None if self.settings is None else asdict(self.settings),
            "slots": (
                None
                if self.slots is None
                else {key: asdict(slot) for key, slot in self.slots.items()}
            ),
            "retained_file_sizes": self.retained_file_sizes,
            "retained": self.retained,
            "generation": self.generation,
            "affected": self.affected,
            "message": self.message,
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
                restore_error=message.get("restore_error"),
                settings=(
                    StoreSettings(**message["settings"])
                    if "settings" in message
                    else None
                ),
                slots=(
                    {key: SlotSummary(**slot) for key, slot in message["slots"].items()}
                    if "slots" in message
                    else None
                ),
                retained_file_sizes=message.get("retained_file_sizes"),
                retained=message.get("retained"),
                generation=message.get("generation"),
                affected=message.get("affected"),
                message=message.get("message"),
            ),
        )


_OP_VALUES = frozenset(op.value for op in StoreOp)


@dataclass
class _Mover:
    to_host: bool
    process: subprocess.Popen
    sock: socket.socket  # the store's end of the pair; the reply carries the new FDs
    deadline: float  # monotonic; past this the store aborts the move
    timeout_s: float = DEFAULT_MOVE_TIMEOUT_S  # move_timeout_s captured at start
    # Automatic idle offloads drop on failure to free the GPU on time; manual ones keep it.
    drop_on_failure: bool = True


@dataclass
class _ReapingMover:
    """Exiting or killed mover tracked until exit so host/GPU resources are fully released."""

    process: subprocess.Popen
    to_host: bool
    gpu: str
    kill_at: Optional[float]  # monotonic; SIGKILL once if still alive past this


@dataclass
class _PublishedSlot:
    record: SlotRecord
    # One per allocation, in Manifest.allocations order. A CUDA allocation's FD is its VMM
    # handle while the record is on the GPU, and a sealed memfd of the same size while on host.
    fds: List[int]
    generation: str = ""
    on_host: bool = False
    holders: Set[socket.socket] = field(default_factory=set)
    idle_since: Optional[float] = None  # monotonic; None while held
    offload_requested: bool = (
        False  # queued by explicit `offload` behind an active offload
    )
    auto_offload_after: float = (
        0.0  # monotonic grace after a manual offload refusal/failure
    )
    last_offload_error: Optional[str] = None
    mover: Optional[_Mover] = None
    restore_error: Optional[str] = None  # reported once, by the next attach


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
    restore_error: str = ""
    hold_sock: Optional[socket.socket] = None
    generation: str = ""
    settings: Optional[StoreSettings] = None


def default_store_name() -> str:
    return (
        os.environ.get("RTP_LLM_RESIDENT_STORE")
        or f"rtp_llm_resident_weights.{os.getuid()}"
    )


def instance_store_name(start_port: int) -> str:
    """Store owned by one serving instance: START_PORT is unique per network namespace."""
    return f"rtp_llm_resident_weights.{os.getuid()}.{int(start_port)}"


def _address(name: str) -> str:
    # Abstract socket: no file to go stale after SIGKILL, no path length limit from TMPDIR.
    return "\0" + name


def _apply_deadline(sock: socket.socket, deadline: Optional[float]) -> None:
    if deadline is None:
        return
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("resident weight store I/O timed out")
    sock.settimeout(remaining)


def _recv_exact(sock: socket.socket, n: int, deadline: Optional[float] = None) -> bytes:
    chunks = []
    while n:
        _apply_deadline(sock, deadline)
        chunk = sock.recv(min(n, 1 << 20))
        if not chunk:
            raise ConnectionError("resident weight store connection closed")
        chunks.append(chunk)
        n -= len(chunk)
    return b"".join(chunks)


def _send_frame(
    sock: socket.socket, obj: dict, fds=(), deadline: Optional[float] = None
) -> None:
    payload = json.dumps(obj).encode()
    header = _HEADER.pack(len(payload))
    _apply_deadline(sock, deadline)
    if fds:
        socket.send_fds(sock, [header], list(fds))
    else:
        sock.sendall(header)
    _apply_deadline(sock, deadline)
    sock.sendall(payload)


def _close(fds) -> None:
    for fd in fds:
        os.close(fd)


def _recv_frame(sock: socket.socket, deadline: Optional[float] = None):
    _apply_deadline(sock, deadline)
    data, fds, _, _ = socket.recv_fds(sock, _HEADER.size, MAX_FDS_PER_FRAME)
    try:
        if not data:
            raise ConnectionError("resident weight store connection closed")
        if len(data) < _HEADER.size:
            data += _recv_exact(sock, _HEADER.size - len(data), deadline=deadline)
        (n,) = _HEADER.unpack(data)
        payload = _recv_exact(sock, n, deadline=deadline)
        return _decode("frame", lambda: json.loads(payload)), list(fds)
    except BaseException:
        _close(fds)
        raise


def send_message(
    sock: socket.socket, obj: dict, fds=(), deadline: Optional[float] = None
) -> None:
    fds = list(fds)
    _send_frame(
        sock, dict(obj, nfds=len(fds)), fds[:MAX_FDS_PER_FRAME], deadline=deadline
    )
    for start in range(MAX_FDS_PER_FRAME, len(fds), MAX_FDS_PER_FRAME):
        _send_frame(
            sock,
            {"op": StoreOp.FDS_CONTINUATION.value},
            fds[start : start + MAX_FDS_PER_FRAME],
            deadline=deadline,
        )
    if deadline is not None and time.monotonic() > deadline:
        raise TimeoutError("resident weight store I/O timed out")


def recv_message(sock: socket.socket, deadline: Optional[float] = None):
    obj, fds = _recv_frame(sock, deadline=deadline)
    try:
        while len(fds) < obj.get("nfds", 0):
            _, more = _recv_frame(sock, deadline=deadline)
            fds += more
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("resident weight store I/O timed out")
    except BaseException:
        # A publisher killed between frames: its allocations must not stay pinned here.
        _close(fds)
        raise
    return obj, fds


def _peer_uid(sock: socket.socket) -> int:
    return _PEERCRED.unpack(
        sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _PEERCRED.size)
    )[1]


def _exchange(
    name: str,
    req: StoreRequest,
    fds=(),
    timeout: float = 600.0,
    keep_open: bool = False,
) -> Tuple[StoreReply, List[int], Optional[socket.socket]]:
    """One request/reply exchange, optionally keeping the connection open as a slot hold."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    kept = False
    reply_fds: List[int] = []
    try:
        sock.settimeout(timeout)
        sock.connect(_address(name))
        # Abstract socket names have no owner permissions. Authenticate before sending FDs.
        if _peer_uid(sock) != os.getuid():
            raise StoreError(
                "resident weight store: peer UID differs from current user"
            )
        send_message(sock, {"proto": PROTOCOL, **req.to_message()}, fds)
        message, reply_fds = recv_message(sock)
        reply = StoreReply.from_message(message)
        if reply.error is not None:
            raise StoreError(f"resident weight store: {reply.error}")
        if keep_open:
            sock.settimeout(None)
            kept = True
            return reply, reply_fds, sock
        return reply, reply_fds, None
    except BaseException:
        _close(reply_fds)
        raise
    finally:
        if not kept:
            sock.close()


def request(
    name: str, req: StoreRequest, fds=(), timeout: float = 600.0
) -> Tuple[StoreReply, List[int]]:
    """One request/reply exchange.

    Raises ConnectionRefusedError (an OSError) when no store is running, and StoreError when the
    store rejects the request or sends something undecodable.
    """
    reply, reply_fds, _ = _exchange(name, req, fds, timeout=timeout, keep_open=False)
    return reply, reply_fds


# ---- typed client API ------------------------------------------------------------------


def publish_slot(
    store_name: str,
    slot_key: str,
    record: SlotRecord,
    fds: Sequence[int],
    hold: bool = False,
) -> Optional[socket.socket]:
    """Publish ``record``; with ``hold=True``, keep and return the connection holding this record."""
    _, _, sock = _exchange(
        store_name,
        StoreRequest(StoreOp.PUBLISH, slot_key=slot_key, record=record, hold=hold),
        fds,
        keep_open=hold,
    )
    return sock


def attach_slot(
    store_name: str, slot_key: str, fingerprint: str, hold: bool = False
) -> AttachResult:
    """Attach ``slot_key``; with ``hold=True``, a HIT keeps and returns the hold connection."""
    reply, fds, sock = _exchange(
        store_name,
        StoreRequest(
            StoreOp.ATTACH, slot_key=slot_key, fingerprint=fingerprint, hold=hold
        ),
        keep_open=hold,
    )
    if reply.status is AttachStatus.HIT and reply.record is not None:
        return AttachResult(
            AttachStatus.HIT,
            record=reply.record,
            allocation_fds=list(fds),
            hold_sock=sock,
            generation=reply.generation or "",
        )
    if sock is not None:
        sock.close()
    _close(fds)
    if reply.status is None or reply.status is AttachStatus.HIT:
        raise StoreError(
            f"resident weight store: malformed attach reply for {slot_key}"
        )
    return AttachResult(
        reply.status,
        published_fingerprint_text=reply.published_fingerprint_text or "",
        restore_error=reply.restore_error or "",
        generation=reply.generation or "",
        settings=reply.settings,
    )


def drop_slot(
    store_name: str,
    slot_key: Optional[str] = None,
    expected_generation: Optional[str] = None,
) -> Optional[SlotRecord]:
    """Release one slot (or every slot) and return the record that was dropped, if any."""
    reply, _ = request(
        store_name,
        StoreRequest(
            StoreOp.DROP,
            slot_key=slot_key,
            expected_generation=expected_generation,
        ),
    )
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


def configure_store(
    store_name: str,
    *,
    offload_after_s: Optional[float] = None,
    idle_ttl_s: Optional[float] = None,
    move_timeout_s: Optional[float] = None,
    host_reserve_ratio: Optional[float] = None,
    host_reserve_min_gb: Optional[float] = None,
) -> StoreSettings:
    """Update a running store's idle/offload settings without restarting it."""
    reply, _ = request(
        store_name,
        StoreRequest(
            StoreOp.CONFIG,
            offload_after_s=offload_after_s,
            idle_ttl_s=idle_ttl_s,
            move_timeout_s=move_timeout_s,
            host_reserve_ratio=host_reserve_ratio,
            host_reserve_min_gb=host_reserve_min_gb,
        ),
    )
    return reply.settings or StoreSettings()


def offload_slots(store_name: str, slot_key: Optional[str] = None) -> StoreReply:
    """Move idle GPU slot(s) to host memory now without waiting for offload_after_s.

    Unlike automatic idle offloads, insufficient host memory refuses the move and keeps the
    GPU record intact instead of dropping it.
    """
    reply, _ = request(store_name, StoreRequest(StoreOp.OFFLOAD, slot_key=slot_key))
    return reply


def keepalive_slots(
    store_name: str,
    slot_key: Optional[str] = None,
) -> StoreReply:
    """Reset the current idle stage's timer (GPU offload timer, or host/GPU TTL timer)."""
    reply, _ = request(
        store_name,
        StoreRequest(StoreOp.KEEPALIVE, slot_key=slot_key),
    )
    return reply


def stop_store(store_name: str) -> None:
    request(store_name, StoreRequest(StoreOp.STOP))


def ensure_running(
    name: str,
    timeout: float = 30.0,
    owner_pid: Optional[int] = None,
    settings: Optional[StoreSettings] = None,
) -> StoreSettings:
    """Start a detached store for ``name`` unless one is already serving; return its settings.

    The store outlives the engine that started it. Only start_server in ``prod`` mode
    stops its own instance store, and only on a clean exit; otherwise the user stops it
    (``store stop``), its idle timers drop the records, or it ends with the container. A store
    already serving keeps its current settings (use ``store config`` to change them live).
    """
    try:
        return list_store(name).settings or StoreSettings()
    except ConnectionRefusedError:
        pass
    validated = validate_settings(settings or StoreSettings())
    # Atomic creation with private permissions also prevents a pre-planted /tmp symlink.
    with tempfile.NamedTemporaryFile(
        mode="a", prefix="rtp-resident-store.", suffix=".log", delete=False
    ) as log:
        log_path = log.name
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
                "--offload-after-s",
                str(validated.offload_after_s),
                "--idle-ttl-s",
                str(validated.idle_ttl_s),
                "--move-timeout-s",
                str(validated.move_timeout_s),
                "--host-reserve-ratio",
                str(validated.host_reserve_ratio),
                "--host-reserve-min-gb",
                str(validated.host_reserve_min_gb),
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
            return list_store(name).settings or StoreSettings()
        except ConnectionRefusedError:
            time.sleep(0.1)
    raise StoreError(f"resident weight store {name} did not start; see {log_path}")


MOVER_ENV = "RTP_LLM_RESIDENT_MOVER"  # overrides the mover command (tests)


_CGROUP = "/sys/fs/cgroup"  # cgroup v2 mount root (patched in tests)
_PROC_CGROUP = "/proc/self/cgroup"
_PROC_MOUNTINFO = "/proc/self/mountinfo"
_PROC_MEMINFO = "/proc/meminfo"
# Linux cgroup v1 represents "unlimited" with PAGE_COUNTER_MAX * PAGE_SIZE (~2^63 - 4096).
_CGROUP_V1_UNLIMITED = 1 << 60


def _cgroup_mounts() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """(fs_root, mount_point) pairs for cgroup v2 and cgroup v1 memory mounts."""
    v2_mounts: List[Tuple[str, str]] = []
    v1_memory_mounts: List[Tuple[str, str]] = []
    with open(_PROC_MOUNTINFO) as f:
        for line in f:
            left, sep, right = line.partition(" - ")
            if not sep:
                continue
            right_fields = right.split()
            if not right_fields:
                continue
            fs_type = right_fields[0]
            if fs_type != "cgroup2" and fs_type != "cgroup":
                continue
            fields = left.split()
            if len(fields) < 5:
                continue
            raw_root = fields[3]
            if ".." in raw_root.split("/"):
                continue  # mount root lies outside this process's cgroup namespace
            fs_root = "/" + raw_root.strip("/")
            mount_point = os.path.normpath(fields[4])
            if fs_type == "cgroup2":
                v2_mounts.append((fs_root, mount_point))
            else:
                super_opts = (
                    right_fields[2].split(",") if len(right_fields) >= 3 else []
                )
                if "memory" in super_opts:
                    v1_memory_mounts.append((fs_root, mount_point))
    return v2_mounts, v1_memory_mounts


def _process_cgroups() -> Tuple[Optional[str], Optional[str]]:
    """(v2_rel, v1_memory_rel) raw paths from ``_PROC_CGROUP``."""
    v2_rel: Optional[str] = None
    v1_memory_rel: Optional[str] = None
    try:
        with open(_PROC_CGROUP) as f:
            for line in f:
                parts = line.strip().split(":", 2)
                if len(parts) != 3:
                    continue
                if parts[0] == "0" and parts[1] == "":
                    v2_rel = parts[2]
                elif parts[0] != "0" and "memory" in parts[1].split(","):
                    v1_memory_rel = parts[2]
    except FileNotFoundError:
        return None, None
    return v2_rel, v1_memory_rel


def _v1_use_hierarchy(cgroup_dir: str) -> bool:
    try:
        with open(os.path.join(cgroup_dir, "memory.use_hierarchy")) as f:
            return f.read().strip() != "0"
    except OSError:
        return True


def _resolve_cgroup_ancestors(
    rel: str,
    mounts: List[Tuple[str, str]],
    *,
    label: str,
    respect_v1_hierarchy: bool = False,
) -> List[str]:
    """Visible ancestor directories across matching mounts for ``rel`` (leaf first)."""
    if ".." in rel.split("/"):
        raise OSError(f"{label} process path {rel!r} lies outside cgroup namespace")
    rel = "/" + rel.strip("/")
    candidates = [
        (m_root, m_point)
        for m_root, m_point in mounts
        if m_root == "/" or rel == m_root or rel.startswith(m_root + "/")
    ]
    if not candidates:
        raise OSError(f"no {label} mount covering process cgroup {rel}")
    # Aggregate visible ancestors across every matching mount (preferring broader mounts
    # first so a full-tree mount is never shadowed by a narrow bind mount).
    candidates.sort(key=lambda item: len(item[0]))
    seen: Set[str] = set()
    ancestors: List[str] = []
    resolved_any = False
    for fs_root, root in candidates:
        sub = (rel if fs_root == "/" else rel[len(fs_root) :]).strip("/")
        dirs = [root]
        cur = root
        for part in sub.split("/"):
            if not part or part == ".":
                continue
            cur = os.path.join(cur, part)
            dirs.append(cur)
        if not os.path.isdir(dirs[-1]):
            continue
        resolved_any = True
        # Walk leaf -> root; in cgroup v1 a parent only charges its child when the
        # parent itself has memory.use_hierarchy != 0.
        for idx in range(len(dirs) - 1, -1, -1):
            d = dirs[idx]
            if idx < len(dirs) - 1 and respect_v1_hierarchy and not _v1_use_hierarchy(d):
                break
            if d not in seen:
                seen.add(d)
                ancestors.append(d)
    if not resolved_any:
        raise OSError(f"no existing {label} directory resolved for {rel}")
    return ancestors


def _cgroup_ancestors() -> Tuple[List[str], List[str]]:
    """Visible (v2_ancestors, v1_memory_ancestors) directories for the memory hierarchy."""
    v2_rel, v1_memory_rel = _process_cgroups()
    if v2_rel is None and v1_memory_rel is None:
        return [], []
    if _CGROUP != "/sys/fs/cgroup" and _PROC_MOUNTINFO == "/proc/self/mountinfo":
        v2_mounts = [("/", os.path.normpath(_CGROUP))]
        v1_mounts: List[Tuple[str, str]] = []
    else:
        v2_mounts, v1_mounts = _cgroup_mounts()
    # A kernel binds the memory controller to at most one hierarchy at a time: when a v1
    # memory entry exists (pure v1 or hybrid kernel), only v1 owns memory accounting.
    if v1_memory_rel is not None:
        return [], _resolve_cgroup_ancestors(
            v1_memory_rel,
            v1_mounts,
            label="cgroup v1 memory",
            respect_v1_hierarchy=True,
        )
    if v2_rel is not None:
        return (
            _resolve_cgroup_ancestors(v2_rel, v2_mounts, label="cgroup2"),
            [],
        )
    return [], []


def _is_group_oom(cgroup_dir: str) -> bool:
    try:
        with open(os.path.join(cgroup_dir, "memory.oom.group")) as f:
            return f.read().strip() == "1"
    except OSError:
        return False


def _read_stat_map(path: str) -> Dict[str, int]:
    try:
        with open(path) as f:
            out: Dict[str, int] = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    out[parts[0]] = int(parts[1])
            return out
    except FileNotFoundError:
        return {}


def _host_memory() -> Tuple[int, int]:
    """(total, available) bytes for this container: the host's, or its tightest cgroup limit.

    /proc/meminfo inside a container shows the host. At a cgroup v1 or v2 limit the memfd's
    pages are charged to the container, and a fallocate past it OOM-kills instead of failing.
    When cgroup v2 ``memory.oom.group`` is enabled, a cgroup OOM kills the entire group
    regardless of ``oom_score_adj``, so use raw headroom (``limit - current``) without
    crediting page cache.
    """
    values = {}
    with open(_PROC_MEMINFO) as meminfo:
        for line in meminfo:
            key, rest = line.split(":", 1)
            values[key] = int(rest.split()[0]) * 1024
    mem_total = values["MemTotal"]
    total, available = mem_total, values["MemAvailable"]
    v2_ancestors, v1_ancestors = _cgroup_ancestors()
    group_oom = any(_is_group_oom(d) for d in v2_ancestors)
    for cgroup_dir in v2_ancestors:
        try:
            with open(os.path.join(cgroup_dir, "memory.max")) as f:
                limit_str = f.read().strip()
        except FileNotFoundError:
            continue  # no cgroup v2 memory controller at this level
        if limit_str == "max":
            continue
        limit = int(limit_str)
        with open(os.path.join(cgroup_dir, "memory.current")) as f:
            current = int(f.read())
        if group_oom:
            reclaimable = 0
        else:
            stat = _read_stat_map(os.path.join(cgroup_dir, "memory.stat"))
            # Page cache other than tmpfs/shmem can be reclaimed to make room.
            if "file" in stat and "shmem" in stat:
                reclaimable = max(0, stat["file"] - stat["shmem"])
            else:
                reclaimable = 0
        total = min(total, limit)
        available = min(available, min(limit, max(0, limit - current + reclaimable)))

    min_visible_v1_limit = _CGROUP_V1_UNLIMITED
    min_visible_v1_memsw_limit = _CGROUP_V1_UNLIMITED
    for cgroup_dir in v1_ancestors:
        # Unlike cgroup2 (which can be mounted without memory), v1_ancestors comes from an
        # explicit memory controller mount, so a missing limit file fails closed.
        with open(os.path.join(cgroup_dir, "memory.limit_in_bytes")) as f:
            limit = int(f.read().strip())
        use_hierarchy = _v1_use_hierarchy(cgroup_dir)
        stat = _read_stat_map(os.path.join(cgroup_dir, "memory.stat"))
        cache_key, shmem_key = (
            ("total_cache", "total_shmem") if use_hierarchy else ("cache", "shmem")
        )
        if cache_key in stat and shmem_key in stat:
            reclaimable = max(0, stat[cache_key] - stat[shmem_key])
        else:
            reclaimable = 0
        if limit < _CGROUP_V1_UNLIMITED:
            min_visible_v1_limit = min(min_visible_v1_limit, limit)
            with open(os.path.join(cgroup_dir, "memory.usage_in_bytes")) as f:
                current = int(f.read().strip())
            total = min(total, limit)
            available = min(
                available, min(limit, max(0, limit - current + reclaimable))
            )
        try:
            with open(os.path.join(cgroup_dir, "memory.memsw.limit_in_bytes")) as f:
                memsw_limit = int(f.read().strip())
        except FileNotFoundError:
            memsw_limit = _CGROUP_V1_UNLIMITED
        if memsw_limit < _CGROUP_V1_UNLIMITED:
            min_visible_v1_memsw_limit = min(min_visible_v1_memsw_limit, memsw_limit)
            with open(os.path.join(cgroup_dir, "memory.memsw.usage_in_bytes")) as f:
                memsw_current = int(f.read().strip())
            total = min(total, memsw_limit)
            available = min(
                available,
                min(memsw_limit, max(0, memsw_limit - memsw_current + reclaimable)),
            )
    if v1_ancestors:
        # Admission sizing covers visible accounting levels (matching the v2 visible-ancestor
        # contract for subtree bind mounts) and fails closed when memory.stat proves an
        # invisible ancestor enforces a tighter limit whose usage cannot be observed.
        leaf_stat = _read_stat_map(os.path.join(v1_ancestors[0], "memory.stat"))
        hier_limit = leaf_stat.get("hierarchical_memory_limit", _CGROUP_V1_UNLIMITED)
        if hier_limit < _CGROUP_V1_UNLIMITED and hier_limit < min_visible_v1_limit:
            raise OSError(
                f"invisible cgroup v1 ancestor enforces tighter memory limit "
                f"({hier_limit} < {min_visible_v1_limit}); cannot observe its usage"
            )
        hier_memsw = leaf_stat.get("hierarchical_memsw_limit", _CGROUP_V1_UNLIMITED)
        if hier_memsw < _CGROUP_V1_UNLIMITED and hier_memsw < min_visible_v1_memsw_limit:
            raise OSError(
                f"invisible cgroup v1 ancestor enforces tighter memsw limit "
                f"({hier_memsw} < {min_visible_v1_memsw_limit}); cannot observe its usage"
            )
    return total, available


def _gpu_of(slot_key: str) -> str:
    uuid = slot_key.split("/", 1)[0]
    return uuid if uuid.startswith(("GPU-", "MIG-")) else f"GPU-{uuid}"


class _Store:
    def __init__(
        self,
        settings: Optional[StoreSettings] = None,
        mover_io_timeout_s: float = MOVER_IO_TIMEOUT_S,
    ):
        self.settings = validate_settings(settings or StoreSettings())
        self._mover_io_timeout_s = mover_io_timeout_s
        self._instance_id = uuid.uuid4().hex
        self._next_generation = 1
        self.slots: Dict[str, _PublishedSlot] = {}
        self.retained_files: Dict[str, _RetainedFile] = {}
        # Finished or aborted mover processes reaped non-blockingly on each tick.
        self._reaping: List[_ReapingMover] = []

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
                generation = self._publish(req.slot_key, req.record, fds)
                kept = True
                return StoreReply(generation=generation), []
            handler = _HANDLERS.get(req.op)
            if handler is None:
                return StoreReply(error=f"op {req.op.value!r} takes no answer here"), []
            return handler(self, req)
        except StoreError as error:
            return StoreReply(error=str(error)), []
        finally:
            if not kept:
                _close(fds)

    def _publish(self, slot_key: str, record: SlotRecord, fds: list) -> str:
        record.published_at = time.time()
        record.eplb_dirty = False
        # Replacing a slot closes the old generation's holds; when the publisher asked for
        # hold=True, serve() registers the publisher's own connection on the new slot.
        self._release(slot_key)
        generation = f"{self._instance_id}:{self._next_generation}"
        self._next_generation += 1
        self.slots[slot_key] = _PublishedSlot(
            record,
            fds,
            generation=generation,
            idle_since=time.monotonic(),
        )
        return generation

    def _attach(self, req: StoreRequest) -> Tuple[StoreReply, List[int]]:
        # Wait until any killed to_gpu mover on this GPU exits and releases its HBM before
        # allowing either a new restore or a cold load (even after DROP/TTL/replacement).
        if req.slot_key and self._has_aborting_gpu_mover(req.slot_key):
            return StoreReply(status=AttachStatus.MOVING, settings=self.settings), []
        held = self.slots.get(req.slot_key)
        if held is None:
            # A killed/exiting to_host mover from a dropped slot still holds its source VMM FDs
            # until exit; wait before returning ABSENT so cold loading does not overlap its HBM.
            if req.slot_key and self._has_reaping_mover_on_gpu(req.slot_key):
                return (
                    StoreReply(status=AttachStatus.MOVING, settings=self.settings),
                    [],
                )
            return StoreReply(status=AttachStatus.ABSENT), []
        if held.record.eplb_dirty or held.record.fingerprint != req.fingerprint:
            if held.mover is not None:
                # Abort any in-flight mover (to_gpu or to_host) on this mismatched/dirty slot
                # and wait until it exits so cold loading cannot overlap its HBM.
                self._abort_mover(req.slot_key, held)
            if req.slot_key and self._has_reaping_mover_on_gpu(req.slot_key):
                return (
                    StoreReply(status=AttachStatus.MOVING, settings=self.settings),
                    [],
                )
            if held.record.eplb_dirty:
                return (
                    StoreReply(
                        status=AttachStatus.EPLB_DIRTY, generation=held.generation
                    ),
                    [],
                )
            # No record here on purpose: see AttachResult.
            return (
                StoreReply(
                    status=AttachStatus.MISMATCH,
                    published_fingerprint_text=held.record.fingerprint_text,
                    generation=held.generation,
                ),
                [],
            )
        if (
            held.on_host
            and req.slot_key
            and self._has_reaping_mover_on_gpu(req.slot_key)
        ):
            return StoreReply(status=AttachStatus.MOVING, settings=self.settings), []
        held.offload_requested = False
        held.auto_offload_after = 0.0
        held.last_offload_error = None
        if not held.holders:
            # Someone wants it: the idle timers restart.
            held.idle_since = time.monotonic()
        if held.mover is not None and held.mover.to_host:
            self._abort_mover(
                req.slot_key, held
            )  # the GPU copy is still intact; hand it out
        if held.on_host:
            if held.restore_error is not None:
                error, held.restore_error = held.restore_error, None
                return (
                    StoreReply(
                        status=AttachStatus.RESTORE_FAILED,
                        restore_error=error,
                        generation=held.generation,
                    ),
                    [],
                )
            if held.mover is None:
                self._start_mover(req.slot_key, held, to_host=False)
                if held.restore_error is not None:
                    error, held.restore_error = held.restore_error, None
                    return (
                        StoreReply(
                            status=AttachStatus.RESTORE_FAILED,
                            restore_error=error,
                            generation=held.generation,
                        ),
                        [],
                    )
            return StoreReply(status=AttachStatus.MOVING, settings=self.settings), []
        # send_fds duplicates the descriptors for the receiver; the store keeps its own.
        return (
            StoreReply(
                status=AttachStatus.HIT,
                record=held.record,
                generation=held.generation,
            ),
            held.fds,
        )

    def add_holder(self, slot_key: str, conn: socket.socket) -> None:
        slot = self.slots[slot_key]
        slot.holders.add(conn)
        slot.idle_since = None
        slot.offload_requested = False
        slot.auto_offload_after = 0.0
        slot.last_offload_error = None

    def holder_closed(self, conn: socket.socket) -> None:
        conn.close()
        for slot in self.slots.values():
            if conn in slot.holders:
                slot.holders.discard(conn)
                if not slot.holders:
                    slot.idle_since = time.monotonic()

    def watched(self) -> List[socket.socket]:
        """Holder connections and mover sockets serve() waits on besides the listener."""
        socks = []
        for slot in self.slots.values():
            socks += slot.holders
            if slot.mover is not None:
                socks.append(slot.mover.sock)
        return socks

    def _track_reaping(
        self,
        process: subprocess.Popen,
        to_host: bool,
        slot_key: str,
        *,
        grace_s: float = 0.0,
    ) -> None:
        if process.poll() is None:
            self._reaping.append(
                _ReapingMover(
                    process=process,
                    to_host=to_host,
                    gpu=_gpu_of(slot_key),
                    kill_at=time.monotonic() + grace_s if grace_s > 0 else None,
                )
            )

    def _reap(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        still_alive: List[_ReapingMover] = []
        for entry in self._reaping:
            if entry.process.poll() is None:
                if entry.kill_at is not None and now >= entry.kill_at:
                    entry.process.kill()
                    entry.kill_at = None
                    if entry.process.poll() is not None:
                        continue
                still_alive.append(entry)
        self._reaping = still_alive

    def _has_active_offload(self) -> bool:
        self._reap()
        return any(
            slot.mover is not None and slot.mover.to_host
            for slot in self.slots.values()
        ) or any(entry.to_host for entry in self._reaping)

    def _has_aborting_gpu_mover(self, slot_key: str) -> bool:
        self._reap()
        gpu = _gpu_of(slot_key)
        return any(not entry.to_host and entry.gpu == gpu for entry in self._reaping)

    def _has_reaping_mover_on_gpu(self, slot_key: str) -> bool:
        self._reap()
        gpu = _gpu_of(slot_key)
        return any(entry.gpu == gpu for entry in self._reaping)

    def tick(self, now: float) -> None:
        """Reap movers, enforce move deadlines, and run idle TTL / offload timers."""
        self._reap(now)
        for slot_key, slot in list(self.slots.items()):
            mover = slot.mover
            if mover is not None and now >= mover.deadline:
                to_host = mover.to_host
                drop_on_failure = mover.drop_on_failure
                timeout_s = mover.timeout_s
                self._abort_mover(slot_key, slot)
                self._mover_failed(
                    slot_key,
                    slot,
                    to_host,
                    f"mover timed out after {timeout_s:g} s",
                    drop_on_failure=drop_on_failure,
                )
        for slot_key, slot in list(self.slots.items()):
            if slot.holders or slot.idle_since is None:
                continue
            idle = now - slot.idle_since
            if slot.on_host:
                if self.settings.idle_ttl_s > 0 and idle >= self.settings.idle_ttl_s:
                    self._release(slot_key, f"idle in host memory {idle:.0f} s")
                continue
            # On GPU: when offload is enabled, offload_after_s governs the GPU stage and
            # idle_ttl_s starts fresh once the record lands in host memory. Only when offload
            # is disabled (-1) does idle_ttl_s drop directly from the GPU.
            if (
                self.settings.offload_after_s <= 0
                and self.settings.idle_ttl_s > 0
                and idle >= self.settings.idle_ttl_s
            ):
                self._release(slot_key, f"idle on GPU {idle:.0f} s")
                continue
            manual = slot.offload_requested
            auto_due = (
                self.settings.offload_after_s > 0
                and idle >= self.settings.offload_after_s
                and now >= slot.auto_offload_after
            )
            if (
                (manual or auto_due)
                and slot.mover is None
                and not self._has_aborting_gpu_mover(slot_key)
                and not self._has_active_offload()
            ):
                slot.offload_requested = False
                # Serialize offloads so each _host_memory() check sees completed allocations
                # and concurrent movers cannot double-commit host memory. A queued manual
                # offload keeps its preserve-GPU semantics even if it crossed offload_after_s.
                self._start_offload(
                    slot_key,
                    slot,
                    drop_on_low_memory=not manual,
                )

    def _gpu_indices(self, slot: _PublishedSlot) -> List[int]:
        return [
            i
            for i, a in enumerate(slot.record.manifest.allocations)
            if a.kind is MemoryKind.CUDA
        ]

    def _host_reserve_bytes(self, total: int) -> int:
        return max(
            int(total * self.settings.host_reserve_ratio),
            int(self.settings.host_reserve_min_gb * (1 << 30)),
        )

    def _defer_auto_offload(self, slot: _PublishedSlot) -> None:
        if self.settings.offload_after_s > 0:
            slot.auto_offload_after = time.monotonic() + self.settings.offload_after_s

    def _start_offload(
        self,
        slot_key: str,
        slot: _PublishedSlot,
        *,
        drop_on_low_memory: bool = True,
    ) -> Optional[str]:
        sizes = [
            slot.record.manifest.allocations[i].size_bytes
            for i in self._gpu_indices(slot)
        ]
        if not sizes:
            slot.on_host = True  # nothing on the GPU to move
            slot.idle_since = time.monotonic()  # start the host-stage TTL clock fresh
            slot.last_offload_error = None
            return None
        try:
            total, available = _host_memory()
            reserve = self._host_reserve_bytes(total)
        except (OSError, ValueError, OverflowError) as error:
            reason = f"could not read host memory limits: {error}"
            if drop_on_low_memory:
                self._release(slot_key, reason)
                return None
            slot.last_offload_error = reason
            self._defer_auto_offload(slot)
            print(
                f"{time.ctime()} manual offload of {slot_key} refused (keeping GPU copy): {reason}",
                flush=True,
            )
            return reason
        needed = sum(sizes)
        if available - needed < reserve:
            reason = (
                f"offload needs {needed / (1 << 30):.2f} GiB, leaving "
                f"{(available - needed) / (1 << 30):.2f} GiB below host reserve "
                f"{reserve / (1 << 30):.2f} GiB "
                f"({available / (1 << 30):.2f} GiB available of {total / (1 << 30):.2f} GiB)"
            )
            if drop_on_low_memory:
                # Releasing the GPU on time matters more than the next start's speed.
                self._release(slot_key, reason)
                return None
            slot.last_offload_error = reason
            self._defer_auto_offload(slot)
            print(
                f"{time.ctime()} manual offload of {slot_key} refused (keeping GPU copy): {reason}",
                flush=True,
            )
            return reason
        slot.last_offload_error = None
        err = self._start_mover(
            slot_key, slot, to_host=True, drop_on_failure=drop_on_low_memory
        )
        return None if drop_on_low_memory else err

    def _start_mover(
        self,
        slot_key: str,
        slot: _PublishedSlot,
        to_host: bool,
        *,
        drop_on_failure: bool = True,
    ) -> Optional[str]:
        indices = self._gpu_indices(slot)
        try:
            ours, theirs = socket.socketpair()
        except OSError as error:
            reason = f"could not create mover socketpair: {error}"
            self._mover_failed(
                slot_key, slot, to_host, reason, drop_on_failure=drop_on_failure
            )
            return reason
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=_gpu_of(slot_key))
        env["RTP_LLM_RESIDENT_MOVER_FD"] = str(theirs.fileno())
        package_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [package_root, env.get("PYTHONPATH")])
        )
        command = os.environ.get(MOVER_ENV)
        argv = (
            [sys.executable, command]
            if command
            else [sys.executable, "-m", "rtp_llm.resident_weights.mover"]
        )
        try:
            process = subprocess.Popen(
                argv,
                env=env,
                pass_fds=[theirs.fileno()],
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as error:
            ours.close()
            theirs.close()
            reason = f"could not start the mover: {error}"
            self._mover_failed(
                slot_key, slot, to_host, reason, drop_on_failure=drop_on_failure
            )
            return reason
        theirs.close()
        sizes = [slot.record.manifest.allocations[i].size_bytes for i in indices]
        try:
            send_message(
                ours,
                {"to_host": to_host, "sizes": sizes},
                [slot.fds[i] for i in indices],
                deadline=time.monotonic() + self._mover_io_timeout_s,
            )
            ours.settimeout(None)
        except (OSError, StoreError) as error:
            ours.close()
            process.kill()
            self._track_reaping(process, to_host, slot_key)
            reason = f"could not send job to mover: {error}"
            self._mover_failed(
                slot_key, slot, to_host, reason, drop_on_failure=drop_on_failure
            )
            return reason
        timeout_s = self.settings.move_timeout_s
        slot.mover = _Mover(
            to_host,
            process,
            ours,
            deadline=time.monotonic() + timeout_s,
            timeout_s=timeout_s,
            drop_on_failure=drop_on_failure,
        )
        print(
            f"{time.ctime()} moving {slot_key} to {'host memory' if to_host else 'the GPU'}: "
            f"{sum(sizes) / (1 << 30):.2f} GiB, mover pid {process.pid}",
            flush=True,
        )
        return None

    def mover_ready(self, sock: socket.socket) -> None:
        """The mover answered, or exited: swap in its FDs, or handle the failure."""
        for slot_key, slot in list(self.slots.items()):
            mover = slot.mover
            if mover is None or mover.sock is not sock:
                continue
            now = time.monotonic()
            fds: List[int] = []
            if now >= mover.deadline:
                error = f"mover timed out after {mover.timeout_s:g} s"
            else:
                try:
                    io_deadline = min(mover.deadline, now + self._mover_io_timeout_s)
                    message, fds = recv_message(sock, deadline=io_deadline)
                    error = message.get("error")
                except (OSError, StoreError) as failure:
                    error = f"mover exited without an answer: {failure}"
                if error is None and time.monotonic() >= mover.deadline:
                    error = f"mover timed out after {mover.timeout_s:g} s"
            slot.mover = None
            sock.close()
            if error is not None:
                mover.process.kill()
                self._track_reaping(mover.process, mover.to_host, slot_key)
            else:
                self._track_reaping(
                    mover.process, mover.to_host, slot_key, grace_s=REAP_GRACE_S
                )
            indices = self._gpu_indices(slot)
            if error is None and len(fds) != len(indices):
                error = (
                    f"mover sent {len(fds)} descriptors for {len(indices)} allocations"
                )
            if error is not None:
                _close(fds)
                self._mover_failed(
                    slot_key,
                    slot,
                    mover.to_host,
                    error,
                    drop_on_failure=mover.drop_on_failure,
                )
                return
            for i, fd in zip(indices, fds):
                os.close(slot.fds[i])
                slot.fds[i] = fd
            slot.on_host = mover.to_host
            if mover.to_host and not slot.holders:
                # Decoupled two-stage retention: start the host-memory TTL clock fresh on arrival.
                slot.idle_since = time.monotonic()
            slot.last_offload_error = None
            print(
                f"{time.ctime()} moved {slot_key} to {'host memory' if mover.to_host else 'the GPU'}",
                flush=True,
            )
            return

    def _mover_failed(
        self,
        slot_key: str,
        slot: _PublishedSlot,
        to_host: bool,
        error: str,
        *,
        drop_on_failure: bool = True,
    ) -> None:
        if to_host:
            if drop_on_failure:
                self._release(slot_key, f"offload failed: {error}")
            else:
                slot.last_offload_error = error
                self._defer_auto_offload(slot)
                print(
                    f"{time.ctime()} manual offload of {slot_key} failed (keeping GPU copy): {error}",
                    flush=True,
                )
        else:
            slot.restore_error = f"moving {slot_key} back to the GPU failed: {error}"
            print(f"{time.ctime()} {slot.restore_error}", flush=True)

    def _abort_mover(self, slot_key: str, slot: _PublishedSlot) -> None:
        mover = slot.mover
        if mover is None:
            return
        slot.mover = None
        mover.process.kill()
        mover.sock.close()  # whatever it built dies with it
        self._track_reaping(mover.process, mover.to_host, slot_key)

    def _drop(self, req: StoreRequest) -> Tuple[StoreReply, List[int]]:
        if not req.slot_key:
            self._release_all()
            return StoreReply(), []
        if req.expected_generation is not None:
            slot = self.slots.get(req.slot_key)
            if slot is None or slot.generation != req.expected_generation:
                return StoreReply(), []
        # Verify mode compares a fresh load against the record it took out.
        return StoreReply(record=self._release(req.slot_key)), []

    def _list(self, req: StoreRequest) -> Tuple[StoreReply, List[int]]:
        return (
            StoreReply(
                slots={
                    slot_key: SlotSummary(
                        total_bytes=held.record.total_bytes,
                        cpu_bytes=sum(
                            a.size_bytes
                            for a in held.record.manifest.allocations
                            if a.kind is MemoryKind.CPU
                        ),
                        gpu_bytes=sum(
                            a.size_bytes
                            for a in held.record.manifest.allocations
                            if a.kind is MemoryKind.CUDA
                        ),
                        allocations=len(held.fds),
                        publisher_pid=held.record.publisher_pid,
                        published_at=held.record.published_at,
                        summary=held.record.summary,
                        eplb_dirty=held.record.eplb_dirty,
                        on_host=held.on_host,
                        holders=len(held.holders),
                        idle_s=(
                            None
                            if held.idle_since is None
                            else time.monotonic() - held.idle_since
                        ),
                        moving=(
                            None
                            if held.mover is None
                            else "to_host" if held.mover.to_host else "to_gpu"
                        ),
                        offload_queued=held.offload_requested,
                        last_offload_error=held.last_offload_error,
                    )
                    for slot_key, held in self.slots.items()
                },
                settings=self.settings,
                retained_file_sizes={
                    path: os.fstat(entry.fd).st_size
                    for path, entry in self.retained_files.items()
                },
            ),
            [],
        )

    def _config(self, req: StoreRequest) -> Tuple[StoreReply, List[int]]:
        candidate = StoreSettings(
            offload_after_s=(
                self.settings.offload_after_s
                if req.offload_after_s is None
                else req.offload_after_s
            ),
            idle_ttl_s=(
                self.settings.idle_ttl_s if req.idle_ttl_s is None else req.idle_ttl_s
            ),
            move_timeout_s=(
                self.settings.move_timeout_s
                if req.move_timeout_s is None
                else req.move_timeout_s
            ),
            host_reserve_ratio=(
                self.settings.host_reserve_ratio
                if req.host_reserve_ratio is None
                else req.host_reserve_ratio
            ),
            host_reserve_min_gb=(
                self.settings.host_reserve_min_gb
                if req.host_reserve_min_gb is None
                else req.host_reserve_min_gb
            ),
        )
        self.settings = validate_settings(candidate)
        print(f"{time.ctime()} updated settings: {self.settings}", flush=True)
        return StoreReply(settings=self.settings), []

    def _offload(self, req: StoreRequest) -> Tuple[StoreReply, List[int]]:
        if req.slot_key:
            if req.slot_key not in self.slots:
                return StoreReply(error=f"slot {req.slot_key} not found"), []
            targets = [req.slot_key]
        else:
            targets = sorted(self.slots)
            if not targets:
                return StoreReply(affected={}, message="store is empty"), []
        affected: Dict[str, str] = {}
        for slot_key in targets:
            slot = self.slots[slot_key]
            if slot.holders:
                affected[slot_key] = f"skipped: in use ({len(slot.holders)} holders)"
            elif slot.on_host:
                affected[slot_key] = "skipped: already in host memory"
            elif slot.mover is not None:
                direction = "to_host" if slot.mover.to_host else "to_gpu"
                affected[slot_key] = f"skipped: already moving {direction}"
            elif self._has_aborting_gpu_mover(slot_key):
                affected[slot_key] = "skipped: previous move still aborting"
            elif not self._has_active_offload():
                refusal = self._start_offload(slot_key, slot, drop_on_low_memory=False)
                if refusal is not None:
                    affected[slot_key] = f"refused: {refusal}"
                elif slot.on_host:
                    affected[slot_key] = "already on host (no GPU allocations)"
                else:
                    affected[slot_key] = "moving to host memory"
            else:
                slot.offload_requested = True
                affected[slot_key] = "queued for host offload"
        return StoreReply(affected=affected), []

    def _keepalive(self, req: StoreRequest) -> Tuple[StoreReply, List[int]]:
        if req.slot_key:
            if req.slot_key not in self.slots:
                return StoreReply(error=f"slot {req.slot_key} not found"), []
            targets = [req.slot_key]
        else:
            targets = sorted(self.slots)
            if not targets:
                return StoreReply(affected={}, message="store is empty"), []
        now = time.monotonic()
        affected: Dict[str, str] = {}
        for slot_key in targets:
            slot = self.slots[slot_key]
            if slot.holders or slot.idle_since is None:
                affected[slot_key] = f"skipped: in use ({len(slot.holders)} holders)"
            elif slot.mover is not None:
                direction = "to_host" if slot.mover.to_host else "to_gpu"
                affected[slot_key] = f"skipped: moving {direction}"
            elif slot.on_host:
                if self.settings.idle_ttl_s <= 0:
                    affected[slot_key] = "skipped: idle_ttl_s is disabled (-1)"
                else:
                    slot.idle_since = now
                    affected[slot_key] = (
                        f"reset host idle timer (expires in {self.settings.idle_ttl_s:g} s)"
                    )
                    print(
                        f"{time.ctime()} keepalive {slot_key}: reset host timer "
                        f"(expires in {self.settings.idle_ttl_s:g} s)",
                        flush=True,
                    )
            else:
                if self.settings.offload_after_s > 0:
                    slot.idle_since = now
                    slot.auto_offload_after = 0.0
                    affected[slot_key] = (
                        f"reset GPU idle timer (offloads in {self.settings.offload_after_s:g} s)"
                    )
                    print(
                        f"{time.ctime()} keepalive {slot_key}: reset GPU timer "
                        f"(offloads in {self.settings.offload_after_s:g} s)",
                        flush=True,
                    )
                elif self.settings.idle_ttl_s > 0:
                    slot.idle_since = now
                    affected[slot_key] = (
                        f"reset GPU idle timer (expires in {self.settings.idle_ttl_s:g} s)"
                    )
                    print(
                        f"{time.ctime()} keepalive {slot_key}: reset GPU timer "
                        f"(expires in {self.settings.idle_ttl_s:g} s)",
                        flush=True,
                    )
                else:
                    affected[slot_key] = "skipped: idle timers are disabled (-1)"
        return StoreReply(affected=affected), []

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

    def _release(self, slot_key: str, why: str = "") -> Optional[SlotRecord]:
        slot = self.slots.pop(slot_key, None)
        if slot is None:
            return None
        self._abort_mover(slot_key, slot)
        for conn in slot.holders:
            conn.close()
        _close(slot.fds)
        print(
            f"{time.ctime()} dropped {slot_key}{': ' + why if why else ''}", flush=True
        )
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
    StoreOp.CONFIG: _Store._config,
    StoreOp.OFFLOAD: _Store._offload,
    StoreOp.KEEPALIVE: _Store._keepalive,
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
    settings: Optional[StoreSettings] = None,
    mover_io_timeout_s: float = MOVER_IO_TIMEOUT_S,
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
    store = _Store(settings, mover_io_timeout_s=mover_io_timeout_s)
    print(
        f"{time.ctime()} store {name} serving, pid {os.getpid()}, {store.settings}",
        flush=True,
    )
    uid = os.getuid()
    while True:
        if owner_pid is not None and (
            owner_identity is None or _process_identity(owner_pid) != owner_identity
        ):
            server.close()
            store._release_all()
            return
        watched = store.watched()
        # Use poll() rather than select(): a multi-GPU store easily holds >1024 allocation FDs,
        # so holder/mover sockets exceed FD_SETSIZE (1024) where select() raises ValueError.
        poller = select.poll()
        by_fd = {server.fileno(): server}
        poller.register(server.fileno(), select.POLLIN)
        for sock in watched:
            fd = sock.fileno()
            by_fd[fd] = sock
            poller.register(fd, select.POLLIN)
        # Wake at least once a second for the idle timers, mover deadlines and the owner check.
        ready = [by_fd[fd] for fd, _ in poller.poll(1000)]
        for sock in ready:
            if sock is server:
                continue
            if sock not in store.watched():
                continue  # released by an earlier event in this batch
            if any(
                sock is slot.mover.sock for slot in store.slots.values() if slot.mover
            ):
                store.mover_ready(sock)
            else:
                store.holder_closed(sock)  # a holder never sends after its hold
        try:
            store.tick(time.monotonic())
        except Exception as error:
            print(f"{time.ctime()} store tick failed: {error!r}", flush=True)
        if server not in ready:
            continue
        conn, _ = server.accept()
        # This store is single-threaded and shared by every GPU of this user: a client that
        # connects and then hangs mid-frame would otherwise block all of them forever.
        conn.settimeout(conn_timeout)
        if _peer_uid(conn) != uid:
            conn.close()
            continue
        stopping = keep = False
        try:
            message, fds = recv_message(conn)
            stopping = message.get("op") == StoreOp.STOP.value
            reply, reply_fds = store.handle(message, fds)
            keep_slot = None
            if reply.error is None and message.get("hold"):
                op = message.get("op")
                if op == StoreOp.PUBLISH.value:
                    keep_slot = message.get("slot_key")
                elif op == StoreOp.ATTACH.value and reply.status is AttachStatus.HIT:
                    keep_slot = message.get("slot_key")
            if keep_slot is not None:
                # Register the hold atomically on the exact published/attached slot before
                # answering, so no tick or replacement can interleave with it.
                store.add_holder(keep_slot, conn)
                keep = True
            if stopping:
                # Stop accepting first, so a client that sees the reply also sees us gone.
                server.close()
                store._release_all()
            try:
                send_message(conn, reply.to_message(), reply_fds)
            except BaseException:
                if keep:
                    store.holder_closed(conn)
                    keep = False
                raise
            if keep:
                conn.settimeout(None)
        except Exception as error:  # one bad client must not take the store down
            print(f"{time.ctime()} request failed: {error!r}", flush=True)
        finally:
            if not keep:
                conn.close()
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


def _format_timer(seconds: float) -> str:
    return "off" if seconds < 0 else f"{seconds:g} s"


def _format_settings(settings: StoreSettings) -> str:
    return (
        f"offload after {_format_timer(settings.offload_after_s)} idle, "
        f"drop after {_format_timer(settings.idle_ttl_s)} idle, "
        f"move timeout {settings.move_timeout_s:g} s, "
        f"host reserve >= max({settings.host_reserve_ratio * 100:g}%, "
        f"{settings.host_reserve_min_gb:g} GiB)"
    )


def _print_list(reply: StoreReply) -> None:
    slots, files = reply.slots or {}, reply.retained_file_sizes or {}
    if not slots and not files:
        print("store is empty")
    if reply.settings is not None:
        print(_format_settings(reply.settings))
    offload = (
        reply.settings.offload_after_s if reply.settings is not None else DISABLED_TIMER
    )
    ttl = reply.settings.idle_ttl_s if reply.settings is not None else DISABLED_TIMER
    for slot_key, slot in sorted(slots.items()):
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(slot.published_at))
        eplb = " eplb_dirty" if slot.eplb_dirty else ""
        where = "host memory" if slot.on_host else "GPU"
        if slot.holders:
            state = f"{slot.holders} holders"
        else:
            idle_s = slot.idle_s or 0.0
            state = f"idle {idle_s:.0f} s"
            if slot.on_host:
                if ttl > 0:
                    remaining = max(0.0, ttl - idle_s)
                    state += f" (expires in {remaining:.0f} s)"
            elif offload > 0:
                remaining = max(0.0, offload - idle_s)
                state += f" (offloads in {remaining:.0f} s)"
            elif ttl > 0:
                remaining = max(0.0, ttl - idle_s)
                state += f" (expires in {remaining:.0f} s)"
        if slot.moving:
            state += f", moving {slot.moving}"
        elif slot.offload_queued:
            state += ", offload queued"
        print(
            f"{slot_key}  GPU weights {slot.gpu_bytes / (1 << 30):.2f} GiB in {where}  "
            f"CPU {slot.cpu_bytes / (1 << 30):.2f} GiB  {slot.allocations} allocations  "
            f"{state}  published {when} by pid {slot.publisher_pid}{eplb}"
        )
        print(f"    {slot.summary}")
        if slot.last_offload_error:
            print(f"    last offload error: {slot.last_offload_error}")
    for path, size in sorted(files.items()):
        print(f"host table {path}  {size / (1 << 30):.2f} GiB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command",
        choices=["serve", "list", "config", "offload", "keepalive", "drop", "stop"],
    )
    parser.add_argument("--name", default=default_store_name())
    parser.add_argument(
        "--slot", help="operate on only this record (default: everything)"
    )
    parser.add_argument("--owner-pid", type=int)
    parser.add_argument("--daemonize", action="store_true")
    parser.add_argument(
        "--offload-after-s",
        type=float,
        default=None,
        help="idle seconds before moving GPU weights to host memory (-1 disables)",
    )
    parser.add_argument(
        "--idle-ttl-s",
        type=float,
        default=None,
        help="idle seconds before dropping a record (-1 disables)",
    )
    parser.add_argument(
        "--move-timeout-s",
        type=float,
        default=None,
        help="hard wall-time bound per mover (seconds)",
    )
    parser.add_argument(
        "--host-reserve-ratio",
        type=float,
        default=None,
        help="minimum fraction of host/cgroup RAM kept free after offload ([0, 1))",
    )
    parser.add_argument(
        "--host-reserve-min-gb",
        type=float,
        default=None,
        help="minimum GiB of host/cgroup RAM kept free after offload (max with ratio)",
    )
    args = parser.parse_args()
    if args.command == "serve":
        defaults = StoreSettings()
        try:
            settings = validate_settings(
                StoreSettings(
                    offload_after_s=(
                        defaults.offload_after_s
                        if args.offload_after_s is None
                        else args.offload_after_s
                    ),
                    idle_ttl_s=(
                        defaults.idle_ttl_s
                        if args.idle_ttl_s is None
                        else args.idle_ttl_s
                    ),
                    move_timeout_s=(
                        defaults.move_timeout_s
                        if args.move_timeout_s is None
                        else args.move_timeout_s
                    ),
                    host_reserve_ratio=(
                        defaults.host_reserve_ratio
                        if args.host_reserve_ratio is None
                        else args.host_reserve_ratio
                    ),
                    host_reserve_min_gb=(
                        defaults.host_reserve_min_gb
                        if args.host_reserve_min_gb is None
                        else args.host_reserve_min_gb
                    ),
                )
            )
        except StoreError as error:
            print(f"error: {error}", file=sys.stderr)
            sys.exit(1)
        serve(
            args.name,
            owner_pid=args.owner_pid,
            daemonize=args.daemonize,
            settings=settings,
        )
        return
    try:
        if args.command == "list":
            _print_list(list_store(args.name))
        elif args.command == "config":
            if all(
                v is None
                for v in (
                    args.offload_after_s,
                    args.idle_ttl_s,
                    args.move_timeout_s,
                    args.host_reserve_ratio,
                    args.host_reserve_min_gb,
                )
            ):
                parser.error("config requires at least one setting flag to update")
            updated = configure_store(
                args.name,
                offload_after_s=args.offload_after_s,
                idle_ttl_s=args.idle_ttl_s,
                move_timeout_s=args.move_timeout_s,
                host_reserve_ratio=args.host_reserve_ratio,
                host_reserve_min_gb=args.host_reserve_min_gb,
            )
            print(_format_settings(updated))
        elif args.command == "offload":
            reply = offload_slots(args.name, args.slot)
            if reply.message:
                print(reply.message)
            failed = False
            for slot_key, status in sorted((reply.affected or {}).items()):
                print(f"{slot_key}: {status}")
                if status.startswith("refused:") or (
                    args.slot and status.startswith("skipped: in use")
                ):
                    failed = True
            if failed:
                sys.exit(1)
        elif args.command == "keepalive":
            reply = keepalive_slots(args.name, args.slot)
            if reply.message:
                print(reply.message)
            for slot_key, status in sorted((reply.affected or {}).items()):
                print(f"{slot_key}: {status}")
        elif args.command == "drop":
            drop_slot(args.name, args.slot)
        else:
            stop_store(args.name)
    except (ConnectionRefusedError, FileNotFoundError):
        print(f"no store is running under {args.name}")
        sys.exit(0)
    except StoreError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
