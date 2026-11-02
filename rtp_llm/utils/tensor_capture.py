"""ExecutionRecorder artifacts for explicitly authorized, eager inference canaries only.

Set RTP_LLM_RECORD_DIR (absolute) and RTP_LLM_RECORD_SESSION to the existing
ExecutionRecorder values. Opt in with enabled=true AND authorized_capture=true;
this authorizes the ENTIRE batch/chunk and its KV history, not individual rows.
Never enable on
unapproved production traffic: activations are sensitive even with anonymous IDs.

Required enabled configuration: layers, ops naming statically registered contracts,
phases selecting "prefill", "decode", or both,
rank_filter="record_owner", and positive max_calls/max_bundle_bytes/
max_total_bytes/max_inflight_bytes. No GPU work or filesystem/thread side effects
when disabled. Instances belong to models, never to a process-global singleton.

Budgets include raw payload and a conservative manifest allowance (not run.json
or interpreter/torch allocator overhead). Four times that bound is reserved for
inflight storage BEFORE copying: snapshots, D2H/contiguity temporaries, encoding
bytes and JSON. Reservations survive dequeue and are released only after owned
storage is discarded. Published and failed write attempts retain their total-byte
charge (incomplete files may remain); cancellation before writing refunds it.
This deliberately overestimates rather than truncating a bundle. Synchronous
producer-stream D2H is intentional; this is diagnostic code. Writer and reader
share a hard 1 GiB aggregate raw payload limit, regardless of configured budgets.
The bytes_published statistic counts tensors plus manifest only for complete
published bundles; it excludes run.json, incomplete files and failed IO bytes.
All named inputs and planned outputs are reserved before copying. Output snapshots
may be staged, but publication requires every declared output. Generic envelopes
use v3 with an independently versioned contract; builtin adapters retain their
legacy artifact formats. Execution identities remain v1.
"""

import hashlib
import json
import logging
import math
import os
import platform
import queue
import re
import stat
import sys
import threading
import traceback
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Optional

import torch

_LOG = logging.getLogger(__name__)
_VERSION = 1  # ExecutionRecorder identity, independent of artifact envelopes.
_ENVELOPE_VERSION = 3
_MAX_PARAMETERS_BYTES = 16384
_MAX_JSON = 1024 * 1024
_MAX_FILE_BYTES = 1024 * 1024 * 1024
_MAX_DIM = 2**31 - 1
_INFLIGHT_FACTOR = 4
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_DTYPES = {
    name: getattr(torch, name)
    for name in (
        "bool",
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    )
    if hasattr(torch, name)
}


def _integer(value, minimum=0, maximum=_MAX_DIM):
    return type(value) is int and minimum <= value <= maximum


def _identifier(value):
    return (
        isinstance(value, str)
        and _IDENTIFIER.fullmatch(value) is not None
        and ".." not in value
    )


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _json_loads(value):
    if len(value) > _MAX_JSON:
        raise ValueError("JSON too large")
    return json.loads(
        value,
        object_pairs_hook=_json_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
    )


def _json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _snapshot_metadata(value, limit):
    """Own JSON metadata without first materializing an unbounded JSON string.

    Count the exact compact, ensure_ascii JSON size while copying containers.
    Immutable scalar references are safe to retain. Checking string length and
    container cardinality before walking also bounds rejection work; depth and
    scalar restrictions prevent cycles, custom serializers and giant integers.
    """
    remaining = limit
    ancestors = set()

    def charge(size):
        nonlocal remaining
        remaining -= size
        if remaining < 0:
            raise ValueError("capture metadata exceeds JSON limit")

    def visit(item, depth):
        if depth > 64:
            raise ValueError("capture metadata is too deep")
        kind = type(item)
        if kind is str:
            charge(2)
            if len(item) > remaining:
                raise ValueError("capture metadata string exceeds JSON limit")
            for character in item:
                code = ord(character)
                if character in ('"', "\\", "\b", "\f", "\n", "\r", "\t"):
                    charge(2)
                elif code < 32 or code > 126:
                    charge(12 if code > 0xFFFF else 6)
                else:
                    charge(1)
            return item
        if item is None:
            charge(4)
            return None
        if kind is bool:
            charge(4 if item else 5)
            return item
        if kind is int:
            if item.bit_length() > 63:
                raise ValueError("capture metadata integer too large")
            charge(len(str(item)))
            return item
        if kind is float:
            if not math.isfinite(item):
                raise ValueError("nonfinite capture metadata")
            charge(len(repr(item)))
            return item
        if kind not in (dict, list, tuple):
            raise ValueError("capture metadata must contain JSON values")
        charge(2)
        if len(item) > remaining:
            raise ValueError("capture metadata container exceeds JSON limit")
        if id(item) in ancestors:
            raise ValueError("cyclic capture metadata")
        ancestors.add(id(item))
        try:
            if kind is dict:
                result = {}
                for index, (key, child) in enumerate(item.items()):
                    if type(key) is not str:
                        raise ValueError("capture metadata keys must be strings")
                    charge(
                        1 + bool(index)
                    )  # Colon and, except for the first item, comma.
                    result[visit(key, depth + 1)] = visit(child, depth + 1)
                return result
            result = []
            for index, child in enumerate(item):
                charge(bool(index))
                result.append(visit(child, depth + 1))
            return result
        finally:
            ancestors.remove(id(item))

    clean = visit(value, 0)
    return clean, limit - remaining


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value):
    if isinstance(value, MappingProxyType):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def _config(path):
    with open(path, "rb") as stream:
        value = _json_loads(stream.read(_MAX_JSON + 1))
    fields = {
        "enabled",
        "authorized_capture",
        "layers",
        "ops",
        "phases",
        "rank_filter",
        "max_calls",
        "max_bundle_bytes",
        "max_total_bytes",
        "max_inflight_bytes",
    }
    if not isinstance(value, dict) or value.keys() - fields:
        raise ValueError("unknown tensor capture configuration field")
    for key in ("enabled", "authorized_capture"):
        if key in value and type(value[key]) is not bool:
            raise ValueError(key + " must be boolean")
    value.setdefault("enabled", False)
    value.setdefault("authorized_capture", False)
    if "layers" in value:
        layers = value["layers"]
        if not isinstance(layers, list) or not all(_integer(x) for x in layers):
            raise ValueError("layers must contain nonnegative integers")
        value["layers"] = sorted(set(layers))
    if "ops" in value:
        value["ops"] = _resolve_ops(value["ops"])
    if "phases" in value:
        phases = value["phases"]
        if (
            not isinstance(phases, list)
            or not phases
            or any(
                not isinstance(phase, str) or phase not in ("prefill", "decode")
                for phase in phases
            )
        ):
            raise ValueError("unsupported phases")
        value["phases"] = [phase for phase in ("prefill", "decode") if phase in phases]
    if "rank_filter" in value and value["rank_filter"] != "record_owner":
        raise ValueError("unsupported rank_filter")
    for key in (
        "max_calls",
        "max_bundle_bytes",
        "max_total_bytes",
        "max_inflight_bytes",
    ):
        if key in value and not _integer(value[key], 1, 2**63 - 1):
            raise ValueError(key + " must be a positive integer")
    if value["enabled"]:
        if not value["authorized_capture"]:
            raise ValueError(
                "authorized_capture=true is required; authorized canaries only"
            )
        if fields - value.keys() or not value["layers"]:
            raise ValueError("enabled capture requires all selectors and budgets")
    return value


def config_enabled(path: str, *, phase=None) -> bool:
    """Validate configuration/phase without a recorder, environment access or GPU work."""
    if phase is not None and phase not in ("prefill", "decode"):
        raise ValueError("unsupported phase")
    config = _config(path)
    return config["enabled"] and (phase is None or phase in config["phases"])


def _identity(source, session, rank):
    if not isinstance(source, dict):
        raise ValueError("invalid execution identity")
    if (
        type(source.get("schema_version")) is not int
        or source["schema_version"] != _VERSION
    ):
        raise ValueError("unsupported identity version")
    if source.get("session_id") != session or source.get("model_role") != "target":
        raise ValueError("session or role mismatch")
    for key in ("owner_id", "replica_id"):
        if not _identifier(source.get(key)):
            raise ValueError("unsafe identity")
    for key in ("world_rank", "dp_rank"):
        if not _integer(source.get(key)):
            raise ValueError("invalid rank")
    if source["world_rank"] != rank:
        raise ValueError("rank mismatch")
    if (
        not isinstance(source.get("exec_id"), str)
        or re.fullmatch(r"[0-9]{1,40}", source["exec_id"]) is None
    ):
        raise ValueError("invalid execution id")
    rows = source.get("requests")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 4096:
        raise ValueError("invalid batch rows")
    requests = []
    phase = None
    q_total = kv_total = 0
    for row in rows:
        if not isinstance(row, dict) or row.get("phase") not in ("prefill", "decode"):
            raise ValueError("invalid request phase")
        if phase is None:
            phase = row["phase"]
        elif phase != row["phase"]:
            raise ValueError("mixed execution phases")
        if type(row.get("is_fake", False)) is not bool or row.get("is_fake", False):
            raise ValueError("fake batch row")
        if (
            not isinstance(row.get("request_id"), str)
            or re.fullmatch(r"r[0-9]{1,40}", row["request_id"]) is None
        ):
            raise ValueError("request id must be recorder-anonymized")
        if not _integer(row.get("q_len"), 1) or not _integer(row.get("kv_len")):
            raise ValueError("invalid sequence lengths")
        if phase == "decode" and row["q_len"] != 1:
            raise ValueError("only ordinary single-query decode is supported")
        clean = {k: row[k] for k in ("phase", "request_id", "q_len", "kv_len")}
        if "sequence_id" in row:
            if not _integer(row["sequence_id"]):
                raise ValueError("invalid sequence id")
            clean["sequence_id"] = row["sequence_id"]
        q_total += row["q_len"]
        kv_total += row["q_len"] + row["kv_len"]
        if q_total > _MAX_DIM or kv_total > _MAX_DIM:
            raise ValueError("batch too large")
        requests.append(clean)
    clean = {
        k: source[k]
        for k in (
            "schema_version",
            "session_id",
            "owner_id",
            "replica_id",
            "world_rank",
            "dp_rank",
            "exec_id",
            "model_role",
        )
    }
    if "server_role" in source:
        if source["server_role"] not in ("PREFILL", "PDFUSION", "DECODE"):
            raise ValueError("invalid server role")
        if (source["server_role"] == "PREFILL" and phase != "prefill") or (
            source["server_role"] == "DECODE" and phase != "decode"
        ):
            raise ValueError("server role and execution phase mismatch")
        clean["server_role"] = source["server_role"]
    if "phase" in source and source["phase"] != phase:
        raise ValueError("explicit execution phase mismatch")
    clean["phase"] = phase
    clean["requests"] = requests
    return clean


def _parameters(parameters):
    if not isinstance(parameters, dict) or len(parameters) > 32:
        raise ValueError("invalid backend parameters")
    clean = {}
    for key, value in parameters.items():
        if not _identifier(key):
            raise ValueError("invalid parameter name")
        if type(value) in (int, bool):
            if type(value) is int and abs(value) > 2**63 - 1:
                raise ValueError("parameter too large")
        elif type(value) is float:
            if not math.isfinite(value):
                raise ValueError("nonfinite parameter")
        elif not _identifier(value):
            raise ValueError("backend parameters must be small scalars")
        clean[key] = value

    if len(_json_bytes(clean)) > _MAX_PARAMETERS_BYTES:
        raise ValueError("backend parameters exceed reserved metadata bound")
    return clean


def _descriptor(tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        raise ValueError("only dense tensors are supported")
    name = str(tensor.dtype).removeprefix("torch.")
    if name not in _DTYPES:
        raise ValueError("unsupported tensor dtype")
    return {
        "dtype": name,
        "shape": list(tensor.shape),
        "original_stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "serialization": "contiguous_row_major",
        "byte_order": "little",
        "nbytes": tensor.numel() * tensor.element_size(),
    }


def _snapshot(tensor):
    # copy=True is essential on CPU too: detach()/contiguous() may alias input.
    return tensor.detach().to(
        device="cpu", copy=True, memory_format=torch.contiguous_format
    )


def _validate_input_snapshots(inputs, tensors, transformed, descriptors):
    # Keep snapshot references out of begin's active exception frame: failed
    # validation/token construction must release storage before refunding quota.
    if tensors.keys() != inputs.keys():
        raise CaptureContractError(
            "snapshot input names differ from declared interface"
        )
    source_storages = {
        tensor.untyped_storage()._cdata
        for tensor in inputs.values()
        if transformed and tensor.device.type == "cpu"
    }
    for name, tensor in tensors.items():
        spec = transformed.get(name)
        if spec is None:
            source = inputs[name]
            spec = TensorSpec(source.shape, source.dtype)
        spec.validate(tensor)
        if (
            tensor.device.type != "cpu"
            or tensor.layout != torch.strided
            or not tensor.is_contiguous()
        ):
            raise CaptureContractError(
                "input snapshot must be dense contiguous CPU storage"
            )
        if name in transformed:
            if (
                tensor.storage_offset() != 0
                or tensor.untyped_storage().nbytes() != spec.nbytes
                or tensor.untyped_storage()._cdata in source_storages
            ):
                raise CaptureContractError(
                    "transformed snapshot must own compact storage"
                )
            descriptors[name] = _descriptor(tensor)


def _encode_tensor(tensor):
    """Raw bit-preserving encoder; also usable for noncontiguous/scalar/empty tensors."""
    desc = _descriptor(tensor)
    cpu = tensor.detach().to(device="cpu").contiguous()
    octets = cpu.reshape(-1).view(torch.uint8).numpy()
    if sys.byteorder != "little" and tensor.element_size() > 1:
        octets = octets.reshape(-1, tensor.element_size())[:, ::-1]
    data = octets.tobytes()
    desc["sha256"] = hashlib.sha256(data).hexdigest()
    return desc, data


def _private_directory(path):
    """Never follow a directory symlink, including existing parent components."""
    path = Path(os.path.abspath(path))
    for part in reversed((path,) + tuple(path.parents)[:-1]):
        try:
            part.mkdir(mode=0o700)
        except FileExistsError:
            pass
        if not stat.S_ISDIR(part.lstat().st_mode):
            raise ValueError("capture directory is not a real directory")
    return path


def _atomic_file(path, data):
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class TensorCaptureRecorder:
    def __init__(self, config_path: str, *, tp_rank: int = 0, world_rank: int = 0):
        self._config = _config(config_path)
        if not _integer(tp_rank) or not _integer(world_rank):
            raise ValueError("invalid recorder rank")
        self._tp_rank, self._world_rank = tp_rank, world_rank
        self._condition = threading.Condition()
        self._counts = Counter()
        self._reasons = Counter()
        self._closed = False
        self._failed = False
        self._thread = None
        self._sink = None
        self._owner = None
        self._pending = 0
        self._inflight = 0
        self._reserved = 0
        self._queue = queue.Queue(maxsize=min(self._config.get("max_calls", 1), 32))
        self._session = self._directory = None
        if self._config["enabled"]:
            self._session = os.environ.get("RTP_LLM_RECORD_SESSION")
            self._directory = os.environ.get("RTP_LLM_RECORD_DIR")
            if (
                not _identifier(self._session)
                or not self._directory
                or not os.path.isabs(self._directory)
            ):
                raise ValueError("valid RTP_LLM_RECORD_DIR/SESSION required")
            if ".." in Path(self._directory).parts or "\x00" in self._directory:
                raise ValueError("unsafe record directory")
            directory = Path(self._directory)
            for part in reversed((directory,) + tuple(directory.parents)[:-1]):
                try:
                    mode = part.lstat().st_mode
                except FileNotFoundError:
                    continue
                if not stat.S_ISDIR(mode):
                    raise ValueError(
                        "record directory must not contain files or symlinks"
                    )

    def note_skip(self, reason: str) -> None:
        if not _identifier(reason) or len(reason) > 64:
            reason = "invalid_reason"
        with self._condition:
            if reason not in self._reasons and len(self._reasons) >= 64:
                reason = "other"
            self._counts["skipped"] += 1
            self._reasons[reason] += 1

    def stats(self) -> dict:
        with self._condition:
            result = {
                key: self._counts[key]
                for key in (
                    "selected",
                    "completed",
                    "skipped",
                    "io_error",
                    "writer_error",
                    "bytes_published",
                )
            }
            result.update(
                inflight_bytes=self._inflight,
                reserved_bytes=self._reserved,
                pending=self._pending,
                queue_size=self._queue.qsize(),
                closed=self._closed,
                failed=self._failed,
                skip_reasons=dict(self._reasons),
            )
            return result

    def _start(self, identity):
        # Only thread creation happens under the lock, never filesystem IO.
        self._owner = identity["owner_id"]
        thread = threading.Thread(
            target=self._writer, name="tensor-capture", daemon=True
        )
        thread.start()
        self._thread = thread

    def _initialize_sink(self):
        root = _private_directory(
            Path(self._directory) / ("owner-" + self._owner) / "tensor-capture"
        )
        self._sink = root / ("sink-" + uuid.uuid4().hex)
        self._sink.mkdir(mode=0o700)
        metadata = {
            "schema_version": _VERSION,
            "format": "rtp_tensor_capture",
            "config": self._config,
            "config_sha256": hashlib.sha256(_json_bytes(self._config)).hexdigest(),
            "session_id": self._session,
            "owner_id": self._owner,
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "torch_git_version": getattr(torch.version, "git_version", None),
            "cuda_build_version": torch.version.cuda,
            "code_version": None,
            "model_provenance": None,
            "gpu": None,
            "inflight_factor": _INFLIGHT_FACTOR,
            "authorization": "explicit_entire_batch_canary",
            "budget_scope": "payload_and_manifest_bound_excluding_run_metadata",
        }
        # Do not probe availability/device_count: those can initialize CUDA.
        if torch.cuda.is_initialized():
            try:
                props = torch.cuda.get_device_properties(torch.cuda.current_device())
                metadata["gpu"] = {
                    "name": props.name,
                    "major": props.major,
                    "minor": props.minor,
                }
            except Exception:
                self.note_skip("gpu_metadata_unavailable")
        _atomic_file(self._sink / "run.json", _json_bytes(metadata))

    def for_execution(
        self, context_json: str, phase=None
    ) -> Optional["TensorCaptureContext"]:
        """Freeze homogeneous identity; an optional producer phase must agree."""
        if not self._config["enabled"]:
            self.note_skip("disabled")
            return None
        if self._tp_rank != 0:
            self.note_skip("non_owner_rank")
            return None
        try:
            source = _json_loads(context_json)
            identity = _identity(source, self._session, self._world_rank)
            if phase is not None and phase != identity["phase"]:
                raise ValueError("producer phase differs from execution identity")
        except Exception:
            self.note_skip("invalid_context")
            return None
        if identity["phase"] not in self._config["phases"]:
            self.note_skip("phase_filter")
            return None
        with self._condition:
            if self._closed or self._failed:
                self.note_skip("closed" if self._closed else "writer_unavailable")
                return None
            if self._owner is not None and self._owner != identity["owner_id"]:
                self.note_skip("owner_mismatch")
                return None
            if self._thread is None:
                try:
                    self._start(identity)
                except Exception:
                    self._failed = True
                    self._counts["writer_error"] += 1
                    self.note_skip("writer_start_failed")
                    return None
        return TensorCaptureContext(self, identity)

    def _reserve(self, reservation):
        size = reservation.size
        with self._condition:
            reason = None
            if self._closed or self._failed:
                reason = "closed" if self._closed else "writer_unavailable"
            elif size > self._config["max_bundle_bytes"]:
                reason = "oversize"
            elif self._counts["selected"] >= self._config["max_calls"]:
                reason = "max_calls"
            elif self._reserved + size > self._config["max_total_bytes"]:
                reason = "total_budget"
            elif (
                self._inflight + _INFLIGHT_FACTOR * size
                > self._config["max_inflight_bytes"]
            ):
                reason = "inflight_budget"
            elif self._queue.full():
                reason = "queue_full"
            if reason:
                self.note_skip(reason)
                return None
            call = self._counts["selected"]
            reserved, inflight, pending = self._reserved, self._inflight, self._pending
            try:
                self._counts["selected"] += 1
                self._reserved += size
                self._inflight += _INFLIGHT_FACTOR * size
                self._pending += 1
                reservation.active = True
                return call
            except BaseException:
                # No other owner can observe a partially acquired reservation.
                self._counts["selected"] = call
                self._reserved, self._inflight, self._pending = (
                    reserved,
                    inflight,
                    pending,
                )
                reservation.active = False
                self._condition.notify_all()
                raise

    def _release(self, size, retain_total=False):
        with self._condition:
            self._inflight -= _INFLIGHT_FACTOR * size
            self._pending -= 1
            if not retain_total:
                self._reserved -= size
            self._condition.notify_all()

    def _enqueue(self, reservation):
        with self._condition:
            if self._closed or self._failed:
                self.note_skip("closed" if self._closed else "writer_unavailable")
                return False
            try:
                self._queue.put_nowait(reservation)
                return True
            except queue.Full:
                self.note_skip("queue_full")
                return False
            finally:
                # Dequeue also holds _condition. Even if put_nowait inserted and
                # then raised, queue membership is authoritative before the writer
                # can consume it. Producer cancellation must not clear queued data.
                with self._queue.mutex:
                    reservation.queued = any(
                        item is reservation for item in self._queue.queue
                    )
                self._condition.notify_all()

    def _write_bundle(self, manifest, tensors, size):
        call_dir = self._sink / ("call-%08d" % manifest["call_id"])
        call_dir.mkdir(mode=0o700)
        written = 0
        for name, tensor in tensors.items():
            desc, data = _encode_tensor(tensor)
            original = manifest["tensors"][name]
            desc.update(
                original_stride=original["original_stride"],
                storage_offset=original["storage_offset"],
                file=name + ".bin",
            )
            manifest["tensors"][name] = desc
            _atomic_file(call_dir / desc["file"], data)
            written += len(data)
            del data
        manifest["complete"] = True
        data = _json_bytes(manifest)
        written += len(data)
        if written > size or len(data) > _MAX_JSON:
            raise ValueError("manifest exceeded reserved bound")
        # Publication point: readers never consider .tmp or orphaned .bin files complete.
        _atomic_file(call_dir / "manifest.json", data)
        return written

    def _writer(self):
        try:
            self._initialize_sink()
        except Exception:
            with self._condition:
                self._failed = True
                self._counts["writer_error"] += 1
        while True:
            with self._condition:
                if (self._closed or self._failed) and self._pending == 0:
                    return
                if not self._condition.wait_for(lambda: not self._queue.empty(), 0.05):
                    continue
                reservation = self._queue.get_nowait()
            manifest, tensors, size = (
                reservation.manifest,
                reservation.tensors,
                reservation.size,
            )
            attempted = False
            try:
                if self._failed:
                    self.note_skip("writer_unavailable")
                    continue
                # Orphaned files still consume disk; never refund a write attempt.
                attempted = True
                written = self._write_bundle(manifest, tensors, size)
                with self._condition:
                    self._counts["completed"] += 1
                    self._counts["bytes_published"] += written
            except Exception:
                with self._condition:
                    self._counts["io_error"] += 1
            finally:
                del tensors, manifest
                reservation.release(writer=True, retain_total=attempted)

    def close(self, timeout: float = 5) -> bool:
        if (
            not isinstance(timeout, (float, int))
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout must be finite and nonnegative")
        with self._condition:
            self._closed = True
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        with self._condition:
            done = self._pending == 0 and (thread is None or not thread.is_alive())
        _LOG.info("Tensor capture close complete=%s stats=%s", done, self.stats())
        return done


class _CaptureReservation:
    """Single owner of snapshot storage and quota, shared by begin/token/writer.

    Producer release is idempotent and cannot affect an accepted queue entry.
    All ownership decisions use the recorder condition, including queue transfer.
    """

    def __init__(self, recorder, size, tensors):
        self.recorder = recorder
        self.size = size
        self.tensors = tensors
        self.manifest = None
        self.active = False
        self.queued = False

    def release(self, *, writer=False, retain_total=False):
        with self.recorder._condition:
            if not self.active or (self.queued and not writer):
                return False
            recorder = self.recorder
            reserved = recorder._reserved - (0 if retain_total else self.size)
            inflight = recorder._inflight - _INFLIGHT_FACTOR * self.size
            pending = recorder._pending - 1
            try:
                self.tensors.clear()
                self.manifest = None
                recorder._release(self.size, retain_total=retain_total)
                self.active = False
                return True
            except BaseException:
                # Complete, rather than retry, a possibly partial refund. The
                # condition remains held across ownership and accounting changes.
                self.tensors.clear()
                self.manifest = None
                recorder._reserved, recorder._inflight, recorder._pending = (
                    reserved,
                    inflight,
                    pending,
                )
                self.active = False
                recorder._condition.notify_all()
                raise


@dataclass(frozen=True)
class TensorSpec:
    """A bounded tensor allocation plan, never an allocation on the source device."""

    shape: tuple
    dtype: torch.dtype
    device: Optional[torch.device] = None

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        if self.device is not None:
            object.__setattr__(self, "device", torch.device(self.device))
        if (
            len(self.shape) > 16
            or not all(_integer(x) for x in self.shape)
            or self.dtype not in _DTYPES.values()
            or math.prod(max(1, x) for x in self.shape) > _MAX_FILE_BYTES
        ):
            raise ValueError("invalid tensor specification")

    @property
    def nbytes(self):
        return (
            math.prod(self.shape)
            * torch.empty((), dtype=self.dtype, device="cpu").element_size()
        )

    def validate(self, tensor):
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != self.shape
            or tensor.dtype != self.dtype
            or (self.device is not None and tensor.device != self.device)
        ):
            raise ValueError("tensor differs from reserved output specification")


class CaptureContractError(ValueError):
    """Adapter rejected input semantics after a budgeted snapshot."""


@dataclass(frozen=True)
class CaptureContract:
    """Static adapter hooks; snapshot hooks may only fill the named owned tensor.

    Validation/sanitization runs under the shared reservation. Metadata is an
    explicit allowlist and cannot replace identity, interface or budget fields.
    New contracts use envelope v3; legacy adapters alone need another envelope.
    """

    name: str
    version: int = 1
    envelope_version: int = _ENVELOPE_VERSION
    aliases: tuple = ()
    selection_group: Optional[str] = None
    metadata_fields: frozenset = frozenset()
    token_type: object = None

    def clean_parameters(self, parameters, outputs):
        return _parameters(parameters)

    def input_specs(self, inputs, identity, metadata):
        """Plan transformed input snapshots without copying or reading GPU values.

        Return a dict of existing input names to exact final CPU TensorSpecs
        (device=None or cpu). Other inputs retain their source descriptors.
        snapshot_inputs must materialize these owned, dense contiguous snapshots
        under the existing reservation; it must not retain the source pool or
        allocate unbounded scratch. Capture-owned peak memory must fit the
        recorder's existing inflight factor. Plans may not resize after reserve.
        """
        return {}

    def snapshot(self, name, tensor, tensors):
        tensors[name] = _snapshot(tensor)

    def snapshot_inputs(self, inputs, tensors, identity, metadata):
        for name, tensor in inputs.items():
            self.snapshot(name, tensor, tensors)

    def error_reason(self, stage):
        return {
            "contract": "invalid_capture_contract",
            "input": "input_snapshot_error",
            "stage_contract": "invalid_output_contract",
            "stage": "output_snapshot_error",
            "state": "invalid_capture_state",
            "output": "output_snapshot_error",
        }[stage]


# Registration is explicit Python code, not executable configuration. Definitions
# are immutable and never replaced while an execution or writer is using them.
_CONTRACTS = {}
_ALIASES = {}
_REGISTRY_LOCK = threading.RLock()
_BUILTINS_LOCK = threading.Lock()
_BUILTINS_READY = False
_PROTECTED_FIELDS = frozenset(
    {
        "schema_version",
        "contract",
        "complete",
        "session_id",
        "owner_id",
        "replica_id",
        "world_rank",
        "dp_rank",
        "exec_id",
        "execution_id",
        "model_role",
        "server_role",
        "requests",
        "call_id",
        "layer",
        "rank",
        "phase",
        "op",
        "inputs",
        "outputs",
        "tensors",
        "backend",
        "parameters",
        "max_calls",
        "max_bundle_bytes",
        "max_total_bytes",
        "max_inflight_bytes",
    }
)


def register_capture_contract(contract):
    """Register a (name, version); versions must share selection semantics."""
    _ensure_builtin_contracts()
    _register_contract(contract)


def _register_contract(contract):
    if (
        not isinstance(contract, CaptureContract)
        or not _identifier(contract.name)
        or not _integer(contract.version, 1)
        or type(contract.envelope_version) is not int
        or contract.envelope_version not in (1, 2, 3)
        or not isinstance(contract.metadata_fields, frozenset)
        or any(not _identifier(x) for x in contract.metadata_fields)
        or contract.metadata_fields & _PROTECTED_FIELDS
        or not isinstance(contract.aliases, tuple)
        or (
            contract.selection_group is not None
            and not _identifier(contract.selection_group)
        )
        or (
            contract.token_type is not None
            and (
                not isinstance(contract.token_type, type)
                or not issubclass(contract.token_type, CaptureToken)
            )
        )
    ):
        raise ValueError("invalid capture contract registration")
    for alias in contract.aliases:
        if (
            not isinstance(alias, tuple)
            or len(alias) < 2
            or any(not _identifier(x) for x in alias)
            or len(set(alias)) != len(alias)
        ):
            raise ValueError("invalid contract selection alias")
    with _REGISTRY_LOCK:
        key = (contract.name, contract.version)
        if key in _CONTRACTS or any(
            alias in _ALIASES and _ALIASES[alias] != contract.name
            for alias in contract.aliases
        ):
            raise ValueError("duplicate capture contract or alias")
        for (name, _), previous in _CONTRACTS.items():
            if name == contract.name and (
                previous.aliases != contract.aliases
                or previous.selection_group != contract.selection_group
            ):
                raise ValueError("inconsistent contract version selection semantics")
        _CONTRACTS[key] = contract
        for alias in contract.aliases:
            _ALIASES[alias] = contract.name


def _ensure_builtin_contracts():
    global _BUILTINS_READY
    with _BUILTINS_LOCK:
        if not _BUILTINS_READY:
            from rtp_llm.utils import tensor_capture_indexer

            tensor_capture_indexer.register_contracts()
            _BUILTINS_READY = True


def _resolve_ops(ops):
    if (
        not isinstance(ops, list)
        or not ops
        or len(ops) > 32
        or any(not _identifier(x) for x in ops)
        or len(set(ops)) != len(ops)
    ):
        raise ValueError("ops must contain unique registered contract names")
    _ensure_builtin_contracts()
    with _REGISTRY_LOCK:
        alias = _ALIASES.get(tuple(ops))
        if alias is not None:
            return [alias]
        contracts = {name: contract for (name, _), contract in _CONTRACTS.items()}
        if any(op not in contracts for op in ops):
            raise ValueError("unknown tensor capture op")
        groups = [
            contracts[op].selection_group
            for op in ops
            if contracts[op].selection_group is not None
        ]
        if len(set(groups)) != len(groups):
            raise ValueError("conflicting capture modes in one selection group")
        return list(ops)


def _contract(op, version=1):
    _ensure_builtin_contracts()
    with _REGISTRY_LOCK:
        if (
            not isinstance(op, str)
            or not _integer(version, 1)
            or (op, version) not in _CONTRACTS
        ):
            raise ValueError("unknown tensor capture op or contract version")
        return _CONTRACTS[(op, version)]


def _validate_envelope(manifest, descriptors):
    contract = manifest.get("contract")
    if (
        not isinstance(contract, dict)
        or set(contract) != {"name", "version"}
        or not _identifier(contract.get("name"))
        or not _integer(contract.get("version"), 1)
        or manifest.get("op") != contract["name"]
        or not _identifier(manifest.get("backend"))
    ):
        raise ValueError("invalid capture contract envelope")
    names = []
    for field in ("inputs", "outputs"):
        values = manifest.get(field)
        if (
            not isinstance(values, list)
            or (field == "outputs" and not values)
            or any(not _identifier(x) for x in values)
        ):
            raise ValueError("invalid named tensor interface")
        names.extend(values)
    if len(names) != len(set(names)) or set(names) != set(descriptors):
        raise ValueError("capture interface differs from tensor descriptors")
    _parameters(manifest.get("parameters"))


class TensorCaptureContext:
    def __init__(self, recorder, identity):
        self._recorder = recorder
        self._identity = _freeze(identity)
        self._identity_bytes = len(_json_bytes(identity))

    @property
    def identity(self):
        return self._identity

    @property
    def phase(self):
        return self._identity["phase"]

    def is_selected(self, op):
        _resolve_ops([op])  # Selection is by name, independent of contract version.
        return op in self._recorder._config.get("ops", ())

    def selected(self, op, layer):
        if not self.is_selected(op):
            self.note_skip("op_filter", layer)
            return False
        if not _integer(layer) or layer not in self._recorder._config["layers"]:
            self.note_skip("layer_filter", layer)
            return False
        return True

    def note_skip(self, reason, layer=None):
        self._recorder.note_skip(reason)

    def _manifest(
        self,
        contract,
        call,
        layer,
        descriptors,
        inputs,
        outputs,
        backend,
        parameters,
        metadata,
    ):
        manifest = _thaw(self._identity)
        manifest.update(
            schema_version=contract.envelope_version,
            complete=False,
            execution_id=manifest["exec_id"],
            call_id=call,
            layer=layer,
            rank=self._recorder._world_rank,
            op=contract.name,
            phase=self.phase,
            backend=backend,
            parameters=parameters,
            tensors=descriptors,
        )
        if contract.envelope_version == _ENVELOPE_VERSION:
            manifest.update(
                contract=dict(name=contract.name, version=contract.version),
                inputs=list(inputs),
                outputs=list(outputs),
            )
        manifest.update(metadata)
        return manifest

    def begin(
        self,
        *,
        op,
        layer,
        inputs,
        outputs,
        backend,
        parameters,
        metadata=None,
        contract_version=1,
    ):
        """Reserve and snapshot inputs; unknown names/versions never fall back."""
        contract = _contract(op, contract_version)
        if not self.selected(op, layer):
            return None
        tensors = {}
        reservation = None
        reason = contract.error_reason("contract")
        try:
            if (
                not isinstance(inputs, dict)
                or not isinstance(outputs, dict)
                or not outputs
            ):
                raise ValueError("capture requires named inputs and planned outputs")
            inputs, outputs = dict(inputs), dict(outputs)
            names = list(inputs) + list(outputs)
            if (
                not 1 <= len(names) <= 32
                or len(set(names)) != len(names)
                or any(not _identifier(name) for name in names)
                or any(not isinstance(spec, TensorSpec) for spec in outputs.values())
                or not _identifier(backend)
            ):
                raise ValueError("invalid capture interface")
            metadata = {} if metadata is None else metadata
            if (
                type(metadata) is not dict
                or len(metadata) > len(contract.metadata_fields)
                or any(
                    key not in contract.metadata_fields or key in _PROTECTED_FIELDS
                    for key in metadata
                )
            ):
                raise ValueError("undeclared or protected capture metadata")
            metadata, metadata_bytes = _snapshot_metadata(metadata, _MAX_JSON)
            parameters = contract.clean_parameters(parameters, outputs)
            parameter_bytes = _json_bytes(parameters)
            if len(parameter_bytes) > _MAX_PARAMETERS_BYTES:
                raise ValueError("parameters exceed reserved metadata bound")
            descriptors = {name: _descriptor(tensor) for name, tensor in inputs.items()}
            # Source pools may exceed the artifact limit, but their descriptors
            # must remain bounded. Only planned packed snapshots are serialized.
            for desc in descriptors.values():
                if (
                    len(desc["shape"]) > 16
                    or any(not _integer(x, 0, 2**63 - 1) for x in desc["shape"])
                    or any(
                        not _integer(x, 0, 2**63 - 1) for x in desc["original_stride"]
                    )
                    or not _integer(desc["storage_offset"], 0, 2**63 - 1)
                ):
                    raise ValueError("invalid source layout")
            transformed = contract.input_specs(inputs, self._identity, metadata)
            if (
                type(transformed) is not dict
                or transformed.keys() - inputs.keys()
                or any(
                    not isinstance(spec, TensorSpec)
                    or (spec.device is not None and spec.device != torch.device("cpu"))
                    for spec in transformed.values()
                )
            ):
                raise ValueError("invalid transformed input specifications")
            transformed = dict(transformed)
            payload = sum(
                transformed[name].nbytes if name in transformed else desc["nbytes"]
                for name, desc in descriptors.items()
            ) + sum(spec.nbytes for spec in outputs.values())
            if payload > _MAX_FILE_BYTES:
                self.note_skip("payload_limit", layer)
                return None
            for name, tensor in inputs.items():
                if name not in transformed:
                    TensorSpec(tensor.shape, tensor.dtype)
            # Do not advertise the source pool as the saved compact operand.
            for name in transformed:
                del descriptors[name]
            # Each bounded descriptor can have 16 shape/stride dimensions. Future
            # stage parameters have a fixed allowance; callers cannot enlarge it.
            size = (
                payload
                + self._identity_bytes
                + metadata_bytes
                + _MAX_PARAMETERS_BYTES
                + 4096
                + 2048 * len(names)
            )
            reservation = _CaptureReservation(self._recorder, size, tensors)
            call = self._recorder._reserve(reservation)
            if call is None:
                return None
            manifest = self._manifest(
                contract,
                call,
                layer,
                descriptors,
                inputs,
                outputs,
                backend,
                parameters,
                metadata,
            )
            reservation.manifest = manifest
            reason = contract.error_reason("input")
            contract.snapshot_inputs(inputs, tensors, self._identity, metadata)
            _validate_input_snapshots(inputs, tensors, transformed, descriptors)
            token_type = contract.token_type or CaptureToken
            return token_type(
                self._recorder, contract, manifest, tensors, reservation, outputs
            )
        except Exception as error:
            if isinstance(error, CaptureContractError):
                reason = contract.error_reason("contract")
            traceback.clear_frames(error.__traceback__)
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            if reservation is not None:
                reservation.release()
            raise
        if reservation is not None:
            reservation.release()
        self.note_skip(reason, layer)
        return None


class CaptureToken:
    """One reservation and publication lifecycle for both direct and staged captures."""

    def __init__(self, recorder, contract, manifest, tensors, reservation, outputs):
        # Begin and partially initialized subclasses share one idempotent owner.
        self._done = True
        self._lock = threading.Lock()
        self._recorder, self._contract = recorder, contract
        self._manifest, self._tensors = manifest, tensors
        self._reservation = reservation
        self._outputs = outputs
        self._done = False

    def _capture(self, outputs, *, final, backend=None, parameters=None):
        with self._lock:
            if self._done:
                self._recorder.note_skip("token_already_finished")
                return False
            queued = False
            reason = self._contract.error_reason(
                "stage_contract" if not final else "output"
            )
            try:
                with self._recorder._condition:
                    if self._recorder._closed or self._recorder._failed:
                        self._cancel_locked(
                            "closed" if self._recorder._closed else "writer_unavailable"
                        )
                        return False
                remaining = self._outputs.keys() - self._tensors.keys()
                if (
                    not isinstance(outputs, dict)
                    or (not outputs and not final)
                    or outputs.keys() - remaining
                    or (final and outputs.keys() != remaining)
                ):
                    self._cancel_locked(self._contract.error_reason("state"))
                    return False
                for name, tensor in outputs.items():
                    self._outputs[name].validate(tensor)
                if backend is not None and not _identifier(backend):
                    raise ValueError("invalid backend")
                if parameters is not None:
                    parameters = self._contract.clean_parameters(
                        parameters, self._outputs
                    )
                    if len(_json_bytes(parameters)) > _MAX_PARAMETERS_BYTES:
                        raise ValueError("parameters exceed reserved metadata bound")
                descriptors = {
                    name: _descriptor(tensor) for name, tensor in outputs.items()
                }
                reason = self._contract.error_reason("output" if final else "stage")
                for name, tensor in outputs.items():
                    self._contract.snapshot(name, tensor, self._tensors)
                with self._recorder._condition:
                    if self._recorder._closed:
                        self._cancel_locked("closed")
                        return False
                self._manifest["tensors"].update(descriptors)
                if backend is not None:
                    self._manifest["backend"] = backend
                if parameters is not None:
                    self._manifest["parameters"] = parameters
                if not final:
                    return True
                queued = self._recorder._enqueue(self._reservation)
                if queued:
                    self._done = True
                    self._tensors = {}
                    self._manifest = None
                    return True
            except Exception as error:
                traceback.clear_frames(error.__traceback__)
            except BaseException as error:
                traceback.clear_frames(error.__traceback__)
                self._cancel_locked(None)
                raise
            self._cancel_locked(reason)
            return False

    def capture_outputs(self, outputs, *, backend=None, parameters=None):
        return self._capture(
            outputs, final=False, backend=backend, parameters=parameters
        )

    def finish(self, outputs):
        return self._capture(outputs, final=True)

    def _cancel_locked(self, reason):
        if self._done:
            return
        self._done = True
        self._reservation.release()
        self._tensors = {}
        self._manifest = None
        if reason is not None:
            self._recorder.note_skip(reason)

    def cancel(self, reason="kernel_error"):
        with self._lock:
            self._cancel_locked(reason)

    def __del__(self):
        if not getattr(self, "_done", True):
            self.cancel("abandoned_token")


def _read_regular(path, limit, exact=None):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size > limit
            or (exact is not None and info.st_size != exact)
        ):
            raise ValueError("invalid capture file size/type")
        data = stream.read(limit + 1)
        if len(data) != info.st_size:
            raise ValueError("capture file changed during read")
        return data


def load_capture(path):
    """Load a complete v1/v2/v3 manifest (or directory), never pickle or source storage.

    Untrusted files are bounded to 1 MiB JSON / 1 GiB aggregate raw tensor bytes;
    tensors are returned CPU-contiguous in logical shape, not original strides.
    """
    path = Path(os.path.abspath(path))
    if path.is_dir():
        path = path / "manifest.json"
    if path.name != "manifest.json" or any(
        p.is_symlink() for p in (path,) + tuple(path.parents)
    ):
        raise ValueError("unsafe manifest path")
    manifest = _json_loads(_read_regular(path, _MAX_JSON))
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] not in (1, 2, _ENVELOPE_VERSION)
        or manifest.get("complete") is not True
    ):
        raise ValueError("incomplete/unsupported capture")
    descriptors = manifest.get("tensors")
    if not isinstance(descriptors, dict) or not 1 <= len(descriptors) <= 32:
        raise ValueError("invalid tensor descriptors")
    if manifest["schema_version"] == _ENVELOPE_VERSION:
        _validate_envelope(manifest, descriptors)
    planned = []
    total = 0
    for name, desc in descriptors.items():
        if (
            not _identifier(name)
            or not isinstance(desc, dict)
            or desc.get("file") != name + ".bin"
        ):
            raise ValueError("unsafe tensor filename")
        dtype_name = desc.get("dtype")
        dtype = _DTYPES.get(dtype_name) if isinstance(dtype_name, str) else None
        shape, stride = desc.get("shape"), desc.get("original_stride")
        if (
            dtype is None
            or not isinstance(shape, list)
            or len(shape) > 16
            or not all(_integer(x) for x in shape)
        ):
            raise ValueError("invalid dtype/shape")
        if (
            not isinstance(stride, list)
            or len(stride) != len(shape)
            or not all(_integer(x, 0, 2**63 - 1) for x in stride)
            or not _integer(desc.get("storage_offset"), 0, 2**63 - 1)
        ):
            raise ValueError("invalid source layout")
        # Even an empty tensor must have a representable logical layout.
        if math.prod(max(1, dimension) for dimension in shape) > _MAX_FILE_BYTES:
            raise ValueError("tensor logical layout exceeds loader limit")
        elements = math.prod(shape)
        expected = elements * torch.empty((), dtype=dtype).element_size()
        if (
            not _integer(desc.get("nbytes"), 0, _MAX_FILE_BYTES)
            or desc["nbytes"] != expected
        ):
            raise ValueError("tensor byte size mismatch")
        if (
            desc.get("serialization") != "contiguous_row_major"
            or desc.get("byte_order") != "little"
        ):
            raise ValueError("unsupported tensor encoding")
        checksum = desc.get("sha256")
        if (
            not isinstance(checksum, str)
            or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
        ):
            raise ValueError("invalid checksum")
        total += expected
        if total > _MAX_FILE_BYTES:
            raise ValueError("capture exceeds loader limit")
        planned.append((name, desc, dtype, expected))
    tensors = {}
    for name, desc, dtype, expected in planned:
        data = _read_regular(path.parent / desc["file"], expected, exact=expected)
        if hashlib.sha256(data).hexdigest() != desc["sha256"]:
            raise ValueError("tensor checksum mismatch")
        if not expected:
            tensors[name] = torch.empty(desc["shape"], dtype=dtype)
        else:
            octets = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            itemsize = torch.empty((), dtype=dtype).element_size()
            if sys.byteorder != "little" and itemsize > 1:
                octets = octets.reshape(-1, itemsize).flip(1).contiguous().reshape(-1)
            tensors[name] = octets.view(dtype).reshape(desc["shape"])
    return manifest, tensors
