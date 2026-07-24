"""Temporary opt-in tensor tracing for Kimi K3 dummy-vs-RTP accuracy work.

The production path pays one ContextVar lookup per explicit trace point and
does not copy tensors unless ``KIMI_K3_ACCURACY_TRACE_DIR`` is set.  Accuracy
runs intentionally synchronize tensors to CPU so each model-forward artifact
is self-contained and can be compared in another Python environment.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import itertools
import json
import os
from pathlib import Path
from typing import Iterator, Optional

import torch

from rtp_llm.models_py.modules.kimi_k3.tmp_test.trace_artifact import (
    DEFAULT_MAX_SHARD_SIZE_BYTES,
    save_tensor_artifact,
)


_ACTIVE_RECORDER: ContextVar[Optional["KimiK3AccuracyRecorder"]] = ContextVar(
    "kimi_k3_accuracy_recorder", default=None
)
_FORWARD_COUNTER = itertools.count()


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
            max_shard_size_bytes=int(
                os.environ.get(
                    "KIMI_K3_ACCURACY_TRACE_MAX_SHARD_BYTES",
                    str(DEFAULT_MAX_SHARD_SIZE_BYTES),
                )
            ),
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
    output = os.environ.get("KIMI_K3_ACCURACY_TRACE_DIR")
    if not output:
        yield None
        return
    enable_file = os.environ.get("KIMI_K3_ACCURACY_TRACE_ENABLE_FILE")
    if enable_file and not Path(enable_file).is_file():
        yield None
        return

    rank = _global_rank()
    requested_rank = int(os.environ.get("KIMI_K3_ACCURACY_TRACE_RANK", "0"))
    if requested_rank >= 0 and rank != requested_rank:
        yield None
        return

    mode = os.environ.get("KIMI_K3_ACCURACY_TRACE_MODE", "boundary")
    if mode == "full":
        mode = "semantic_full"
    if mode not in ("boundary", "semantic_full"):
        raise ValueError(
            "KIMI_K3_ACCURACY_TRACE_MODE must be 'boundary' or "
            "'semantic_full', "
            f"got {mode!r}"
        )
    recorder = KimiK3AccuracyRecorder(
        output_dir=Path(output),
        phase=phase,
        mode=mode,
        rank=rank,
        forward_index=next(_FORWARD_COUNTER),
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
]
