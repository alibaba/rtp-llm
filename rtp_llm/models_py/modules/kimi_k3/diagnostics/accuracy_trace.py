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
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import torch

from rtp_llm.models_py.modules.kimi_k3.diagnostics.trace_artifact import (
    DEFAULT_MAX_SHARD_SIZE_BYTES,
    save_tensor_artifact,
)

_ACTIVE_RECORDER: ContextVar[Optional["KimiK3AccuracyRecorder"]] = ContextVar(
    "kimi_k3_accuracy_recorder", default=None
)
_FORWARD_COUNTER = itertools.count()

_SPEC_KEYS = frozenset(
    {"rank", "mode", "forward", "router", "token", "enable_file", "shard_bytes"}
)


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
    "accuracy_trace_mode",
    "kimi_k3_accuracy_trace",
    "mark_accuracy_fake_stream",
    "record_accuracy_tensor",
    "tensor_dump_enabled",
    "tensor_dump_full_router",
]
