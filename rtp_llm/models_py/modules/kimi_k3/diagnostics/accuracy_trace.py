"""Opt-in tensor tracing for Kimi K3 accuracy diagnostics.

The production path pays one ContextVar lookup per explicit trace point and
does not copy tensors unless ``KIMI_K3_TENSOR_DUMP`` is set.  Accuracy runs
intentionally synchronize tensors to CPU so each model-forward artifact is
self-contained and can be compared in another Python environment.

Configuration is one variable so that a run's entire diagnostic setup is
visible on the command line:

    KIMI_K3_TENSOR_DUMP=<dir>[,key=value]...

The bare first field is the output directory; an empty or unset variable
disables tracing entirely.  Recognized keys:

    rank=<int>          only this global rank records; -1 records every rank
    mode=boundary|semantic_full
    forward=<int>       only this model-forward ordinal records
    router=full         keep the full O(T) router rows, not just the last
    token=<int>         only record when the single decode input id matches
    enable_file=<path>  record only while this file exists
    shard_bytes=<int>   safetensors shard ceiling

This replaced eight separate KIMI_K3_ACCURACY_TRACE_* variables whose
combined state nobody could read off a launch command.
"""

from __future__ import annotations

import itertools
import hashlib
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

from safetensors import safe_open
from safetensors.torch import save_file
import torch
import torch.nn.functional as F


TRACE_ARTIFACT_FORMAT = "kimi-k3-tensor-artifact-v1"
DEFAULT_MAX_SHARD_SIZE_BYTES = 3 << 30


def file_sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        while chunk := reader.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_index_path(logical_path: Path) -> Path:
    return logical_path.with_suffix(logical_path.suffix + ".index.json")


def artifact_exists(logical_path: Path) -> bool:
    return logical_path.is_file() or artifact_index_path(logical_path).is_file()


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _plan_shards(
    tensors: Mapping[str, torch.Tensor], max_shard_size_bytes: int
) -> list[list[str]]:
    if max_shard_size_bytes <= 0:
        raise ValueError("max_shard_size_bytes must be positive")
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for name in sorted(tensors):
        tensor_size = _tensor_nbytes(tensors[name])
        if current and current_size + tensor_size > max_shard_size_bytes:
            shards.append(current)
            current = []
            current_size = 0
        current.append(name)
        current_size += tensor_size
    if current:
        shards.append(current)
    return shards


def _shard_name(logical_path: Path, shard: int, count: int) -> str:
    suffix = logical_path.suffix or ".safetensors"
    stem = logical_path.name[: -len(suffix)] if suffix else logical_path.name
    return f"{stem}-{shard:05d}-of-{count:05d}{suffix}"


def save_tensor_artifact(
    tensors: Mapping[str, torch.Tensor],
    logical_path: Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    max_shard_size_bytes: int = DEFAULT_MAX_SHARD_SIZE_BYTES,
) -> Path:
    """Save one checksummed, optionally sharded accuracy artifact."""

    if artifact_exists(logical_path):
        raise FileExistsError(f"tensor artifact already exists: {logical_path}")
    logical_path.parent.mkdir(parents=True, exist_ok=True)
    contiguous = {
        name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()
    }
    tensor_bytes = sum(_tensor_nbytes(tensor) for tensor in contiguous.values())
    common_metadata = {
        "format": TRACE_ARTIFACT_FORMAT,
        **({} if metadata is None else {key: str(value) for key, value in metadata.items()}),
    }
    shards = _plan_shards(contiguous, max_shard_size_bytes)
    if len(shards) <= 1:
        save_file(contiguous, str(logical_path), metadata=common_metadata)
        logical_path.with_suffix(logical_path.suffix + ".sha256").write_text(
            file_sha256(logical_path) + "\n", encoding="utf-8"
        )
        return logical_path

    weight_map: dict[str, str] = {}
    shard_manifest: dict[str, dict[str, Any]] = {}
    for shard_index, names in enumerate(shards, start=1):
        shard_name = _shard_name(logical_path, shard_index, len(shards))
        shard_path = logical_path.parent / shard_name
        shard_tensors = {name: contiguous[name] for name in names}
        save_file(shard_tensors, str(shard_path), metadata=common_metadata)
        for name in names:
            weight_map[name] = shard_name
        shard_manifest[shard_name] = {
            "tensor_count": len(names),
            "tensor_nbytes": sum(_tensor_nbytes(contiguous[name]) for name in names),
            "file_nbytes": shard_path.stat().st_size,
            "sha256": file_sha256(shard_path),
        }

    index = {
        "format": TRACE_ARTIFACT_FORMAT,
        "logical_name": logical_path.name,
        "tensor_count": len(contiguous),
        "tensor_nbytes": tensor_bytes,
        "max_shard_size_bytes": max_shard_size_bytes,
        "metadata": dict(metadata or {}),
        "weight_map": weight_map,
        "shards": shard_manifest,
    }
    index_path = artifact_index_path(logical_path)
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    index_path.with_suffix(index_path.suffix + ".sha256").write_text(
        file_sha256(index_path) + "\n", encoding="utf-8"
    )
    return index_path


class TensorArtifactReader:
    """Read one logical tensor artifact without loading unrelated shards."""

    def __init__(self, logical_path: Path, *, verify: bool = False) -> None:
        self.logical_path = logical_path
        self.index_path = artifact_index_path(logical_path)
        self.index: dict[str, Any] | None = None
        if logical_path.is_file():
            with safe_open(str(logical_path), framework="pt", device="cpu") as handle:
                self.weight_map = {name: logical_path.name for name in handle.keys()}
        elif self.index_path.is_file():
            self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
            if self.index.get("format") != TRACE_ARTIFACT_FORMAT:
                raise ValueError(f"unsupported tensor artifact: {self.index_path}")
            self.weight_map = dict(self.index["weight_map"])
        else:
            raise FileNotFoundError(f"tensor artifact does not exist: {logical_path}")
        if verify:
            errors = self.validate()
            if errors:
                raise ValueError(f"invalid tensor artifact {logical_path}: {errors}")

    def keys(self) -> set[str]:
        return set(self.weight_map)

    def _path_for(self, name: str) -> Path:
        try:
            shard = self.weight_map[name]
        except KeyError as error:
            raise KeyError(f"tensor artifact has no key {name!r}") from error
        return self.logical_path.parent / shard

    def tensor(self, name: str) -> torch.Tensor:
        path = self._path_for(name)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def tensors(self, names: Iterable[str] | None = None) -> dict[str, torch.Tensor]:
        selected = sorted(self.keys() if names is None else set(names))
        by_shard: dict[Path, list[str]] = {}
        for name in selected:
            by_shard.setdefault(self._path_for(name), []).append(name)
        result: dict[str, torch.Tensor] = {}
        for path, shard_names in by_shard.items():
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for name in shard_names:
                    result[name] = handle.get_tensor(name)
        return result

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.index is None:
            checksum_path = self.logical_path.with_suffix(
                self.logical_path.suffix + ".sha256"
            )
            if not checksum_path.is_file():
                errors.append("missing file checksum")
            elif checksum_path.read_text(encoding="utf-8").strip() != file_sha256(
                self.logical_path
            ):
                errors.append("file checksum mismatch")
            try:
                with safe_open(
                    str(self.logical_path), framework="pt", device="cpu"
                ) as handle:
                    if set(handle.keys()) != self.keys():
                        errors.append("single-file key set changed during validation")
            except Exception as error:
                errors.append(f"cannot open single-file artifact: {error}")
            return errors

        checksum_path = self.index_path.with_suffix(self.index_path.suffix + ".sha256")
        if not checksum_path.is_file():
            errors.append("missing index checksum")
        elif checksum_path.read_text(encoding="utf-8").strip() != file_sha256(
            self.index_path
        ):
            errors.append("index checksum mismatch")
        observed: set[str] = set()
        for shard_name, expected in self.index["shards"].items():
            shard_path = self.logical_path.parent / shard_name
            if not shard_path.is_file():
                errors.append(f"missing shard: {shard_name}")
                continue
            if shard_path.stat().st_size != int(expected["file_nbytes"]):
                errors.append(f"file size mismatch: {shard_name}")
            if file_sha256(shard_path) != expected["sha256"]:
                errors.append(f"checksum mismatch: {shard_name}")
                continue
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                shard_keys = set(handle.keys())
            expected_keys = {
                name
                for name, mapped_shard in self.weight_map.items()
                if mapped_shard == shard_name
            }
            if shard_keys != expected_keys:
                errors.append(f"key set mismatch: {shard_name}")
            observed.update(shard_keys)
        if observed != self.keys():
            errors.append("observed key set differs from index")
        return errors

_ACTIVE_RECORDER: ContextVar[Optional["KimiK3AccuracyRecorder"]] = ContextVar(
    "kimi_k3_accuracy_recorder", default=None
)
_FORWARD_COUNTER = itertools.count()

_SPEC_KEYS = frozenset(
    {"rank", "mode", "forward", "router", "token", "enable_file", "shard_bytes"}
)


def prepare_kimi_kda_trace_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: Optional[torch.Tensor] = None,
    *,
    lower_bound: Optional[float] = None,
    scale: Optional[float] = None,
    norm_epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize KDA's fused input transforms only for accuracy tracing."""

    if q.shape != k.shape or q.shape != raw_gate.shape or q.ndim != 4:
        raise ValueError("q, k and raw_gate must share shape [B,T,H,K]")
    if raw_beta.shape != q.shape[:-1]:
        raise ValueError("raw_beta must have shape [B,T,H]")

    heads, key_dim = q.shape[-2:]
    if a_log.shape != (heads,):
        raise ValueError(f"a_log must have shape {(heads,)}, got {tuple(a_log.shape)}")
    if dt_bias is not None and dt_bias.numel() != heads * key_dim:
        raise ValueError(f"dt_bias must contain {heads * key_dim} values")
    if lower_bound is not None and lower_bound >= 0:
        raise ValueError("KDA lower_bound is a negative log-decay bound")

    q_float = q.float()
    k_float = k.float()
    q_float = q_float * torch.rsqrt(
        q_float.square().sum(dim=-1, keepdim=True) + norm_epsilon
    )
    k_float = k_float * torch.rsqrt(
        k_float.square().sum(dim=-1, keepdim=True) + norm_epsilon
    )
    q_float = q_float * (key_dim**-0.5 if scale is None else scale)

    gate_input = raw_gate.float()
    if dt_bias is not None:
        gate_input = gate_input + dt_bias.float().reshape(heads, key_dim)
    rate = a_log.float().exp().reshape(1, 1, heads, 1)
    log_decay = (
        -rate * F.softplus(gate_input)
        if lower_bound is None
        else float(lower_bound) * torch.sigmoid(rate * gate_input)
    )
    return q_float, k_float, log_decay.exp(), raw_beta.float().sigmoid()


def _tensor_dump_spec() -> dict[str, str]:
    """Parse KIMI_K3_TENSOR_DUMP into {"dir": ..., <key>: <value>}."""

    raw = os.environ.get("KIMI_K3_TENSOR_DUMP", "").strip()
    if not raw:
        return {}
    fields = [field.strip() for field in raw.split(",")]
    if not fields[0] or "=" in fields[0]:
        raise ValueError(
            "KIMI_K3_TENSOR_DUMP must start with the output directory, got "
            f"{raw!r}"
        )
    spec = {"dir": fields[0]}
    for field_text in fields[1:]:
        if not field_text:
            continue
        key, separator, value = field_text.partition("=")
        key = key.strip()
        if not separator or key not in _SPEC_KEYS:
            raise ValueError(
                f"unsupported KIMI_K3_TENSOR_DUMP field {field_text!r}; "
                f"expected one of {sorted(_SPEC_KEYS)}"
            )
        spec[key] = value.strip()
    return spec


def _spec_int(spec: dict[str, str], key: str) -> Optional[int]:
    value = spec.get(key)
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(
            f"KIMI_K3_TENSOR_DUMP {key}= must be an integer, got {value!r}"
        ) from error
    return parsed


def tensor_dump_full_router() -> bool:
    """Whether the router trace keeps every token row instead of the last."""

    return _tensor_dump_spec().get("router") == "full"


def tensor_dump_enabled() -> bool:
    """Whether any tensor dump directory is configured for this process."""

    return bool(_tensor_dump_spec())


def accuracy_trace_requested() -> bool:
    """Whether this process was launched with an accuracy-trace request."""

    return bool(os.environ.get("KIMI_K3_TENSOR_DUMP"))


def _global_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    for name in ("RANK", "WORLD_RANK", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return 0


@dataclass
class KimiK3AccuracyRecorder:
    output_dir: Path
    phase: str
    mode: str = "boundary"
    rank: int = 0
    forward_index: int = 0
    input_token_id: Optional[int] = None
    max_shard_size_bytes: int = DEFAULT_MAX_SHARD_SIZE_BYTES
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    token_dims: dict[str, int] = field(default_factory=dict)
    capture_enabled: bool = True

    def record(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        token_dim: Optional[int] = None,
    ) -> None:
        if not self.capture_enabled:
            return
        if name in self.tensors:
            raise KeyError(f"duplicate K3 accuracy trace tensor: {name}")
        value = tensor.detach()
        if self.mode == "boundary" and token_dim is not None:
            if value.shape[token_dim] == 0:
                raise ValueError(f"cannot trace last token of empty tensor {name}")
            index = [slice(None)] * value.ndim
            index[token_dim] = slice(-1, None)
            value = value[tuple(index)]
        if token_dim is not None:
            self.token_dims[name] = int(token_dim)
        self.tensors[name] = value.contiguous().cpu()
        if name == "input_ids" and self.input_token_id is not None:
            flat_input_ids = value.reshape(-1)
            if flat_input_ids.numel() != 1:
                raise ValueError(
                    "KIMI_K3_TENSOR_DUMP token= only supports single-token "
                    f"Decode forwards, got {flat_input_ids.numel()} input tokens"
                )
            if int(flat_input_ids.item()) != self.input_token_id:
                self.capture_enabled = False

    def mark_fake_stream(self, is_fake: bool, device: torch.device) -> None:
        self.record(
            "stream_is_fake",
            torch.tensor(int(is_fake), dtype=torch.uint8, device=device),
        )
        if is_fake:
            # DP collective-only streams must execute the model, but their
            # synthetic activations and full-head cache tensors are neither an
            # accuracy oracle nor worth copying to the host.
            self.capture_enabled = False

    def flush(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            f"{self.phase}-rank{self.rank:03d}-pid{os.getpid()}-"
            f"forward{self.forward_index:06d}"
        )
        tensor_path = self.output_dir / f"{stem}.safetensors"
        metadata = {
            "format": "kimi-k3-accuracy-trace-v1",
            "phase": self.phase,
            "mode": self.mode,
            "rank": self.rank,
            "pid": os.getpid(),
            "forward_index": self.forward_index,
            "tensor_file": tensor_path.name,
            "tensor_count": len(self.tensors),
            "tensors": {
                name: {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                }
                for name, tensor in self.tensors.items()
            },
            "token_dims": self.token_dims,
        }
        artifact_path = save_tensor_artifact(
            self.tensors,
            tensor_path,
            metadata={
                "phase": self.phase,
                "mode": self.mode,
                "rank": self.rank,
                "forward_index": self.forward_index,
            },
            max_shard_size_bytes=self.max_shard_size_bytes,
        )
        metadata["tensor_artifact"] = artifact_path.name
        (self.output_dir / f"{stem}.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return tensor_path


def record_accuracy_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    token_dim: Optional[int] = None,
) -> None:
    recorder = _ACTIVE_RECORDER.get()
    if recorder is not None:
        recorder.record(name, tensor, token_dim=token_dim)


def accuracy_trace_mode() -> Optional[str]:
    recorder = _ACTIVE_RECORDER.get()
    return None if recorder is None or not recorder.capture_enabled else recorder.mode


def accuracy_trace_enabled() -> bool:
    return accuracy_trace_mode() is not None


def mark_accuracy_fake_stream(is_fake: bool, device: torch.device) -> None:
    recorder = _ACTIVE_RECORDER.get()
    if recorder is not None:
        recorder.mark_fake_stream(is_fake, device)


@contextmanager
def kimi_k3_accuracy_trace(phase: str) -> Iterator[Optional[KimiK3AccuracyRecorder]]:
    spec = _tensor_dump_spec()
    if not spec:
        yield None
        return
    enable_file = spec.get("enable_file")
    if enable_file and not Path(enable_file).is_file():
        yield None
        return

    rank = _global_rank()
    requested_rank = _spec_int(spec, "rank")
    if requested_rank is None:
        requested_rank = 0
    if requested_rank >= 0 and rank != requested_rank:
        yield None
        return

    mode = spec.get("mode") or "boundary"
    if mode == "full":
        mode = "semantic_full"
    if mode not in ("boundary", "semantic_full"):
        raise ValueError(
            "KIMI_K3_TENSOR_DUMP mode= must be 'boundary' or 'semantic_full', "
            f"got {mode!r}"
        )
    forward_index = next(_FORWARD_COUNTER)
    requested_forward = _spec_int(spec, "forward")
    if requested_forward is not None:
        if requested_forward < 0:
            raise ValueError(
                "KIMI_K3_TENSOR_DUMP forward= must be non-negative, got "
                f"{requested_forward}"
            )
        if forward_index != requested_forward:
            yield None
            return
    input_token_id = _spec_int(spec, "token")
    if input_token_id is not None and input_token_id < 0:
        raise ValueError(
            f"KIMI_K3_TENSOR_DUMP token= must be non-negative, got {input_token_id}"
        )
    shard_bytes = _spec_int(spec, "shard_bytes")
    recorder = KimiK3AccuracyRecorder(
        output_dir=Path(spec["dir"]),
        phase=phase,
        mode=mode,
        rank=rank,
        forward_index=forward_index,
        input_token_id=input_token_id,
        max_shard_size_bytes=(
            DEFAULT_MAX_SHARD_SIZE_BYTES if shard_bytes is None else shard_bytes
        ),
    )
    token = _ACTIVE_RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE_RECORDER.reset(token)
        recorder.flush()


__all__ = [
    "KimiK3AccuracyRecorder",
    "TensorArtifactReader",
    "artifact_exists",
    "artifact_index_path",
    "accuracy_trace_enabled",
    "accuracy_trace_mode",
    "accuracy_trace_requested",
    "kimi_k3_accuracy_trace",
    "mark_accuracy_fake_stream",
    "prepare_kimi_kda_trace_inputs",
    "record_accuracy_tensor",
    "save_tensor_artifact",
    "tensor_dump_enabled",
    "tensor_dump_full_router",
]
