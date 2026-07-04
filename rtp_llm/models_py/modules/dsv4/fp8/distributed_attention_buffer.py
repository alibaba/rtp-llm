"""Attention-dedicated CP symmetric-buffer lifecycle.

The fused DSv4 prefill attention op is a peer-symmetric CP collective.  Every
rank must reject unsupported shapes before entering the op, and every rank must
pass the same communication layout into the CUDA side.  This module owns that
small but load-bearing contract; the CUDA op owns the actual exchange/barrier
logic.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist


_BUFFER_CACHE: dict[tuple, "Dsv4CpAttentionBuffer"] = {}
_DEFAULT_ALIGN_BYTES = 256
_DEFAULT_PROTOCOL_BYTES = 4352
_ENV_MAX_TOKENS = "DSV4_CP_DISTRIBUTED_PREFILL_ATTN_MAX_TOKENS_PER_RANK"
_ENV_MEGA_COMPRESSED_TOPK_CAP = "DSV4_CP_DISTRIBUTED_PREFILL_ATTN_COMPRESSED_TOPK_CAP"
_ENV_MEGA_SPLITK_SCRATCH_BYTES = "DSV4_CP_DISTRIBUTED_PREFILL_ATTN_SPLITK_SCRATCH_BYTES"
_ENV_MEGA_CSA_SCORE_SCRATCH_BYTES = "DSV4_CP_DISTRIBUTED_PREFILL_ATTN_CSA_SCORE_SCRATCH_BYTES"
_DEFAULT_MEGA_COMPRESSED_TOPK_CAP = 1024
_DEFAULT_MEGA_GRID_SYNC_BYTES = 1152
_DEFAULT_MEGA_SPLITK_SCRATCH_BYTES = 4 * 1024 * 1024
_SIGNAL_PAD_CHANNEL_BASE = 224
_SIGNAL_PAD_CHANNEL_COUNT = 32


def _align_up(value: int, alignment: int) -> int:
    alignment = int(alignment)
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((int(value) + alignment - 1) // alignment) * alignment


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Dsv4CpAttentionBufferSpec:
    """Capacity key for the attention symmetric buffer.

    The per-rank region has persistent payloads plus a transient scratch
    payload.  Fresh KV and semantic indexer-K must remain readable until the
    attention body finishes, while SWA/compressor exchanges reuse the scratch
    payload during earlier writer phases.
    """

    cp_size: int
    max_tokens_per_rank: int
    batch_cap: int
    swa_bytes_per_token: int
    # The symmetric region stages raw BF16 projection payloads before the op
    # writes the packed 584B / 132B cache rows.  Size for worst-case
    # overlap=True payload widths, not for the packed cache entry sizes.
    main_bytes_per_token: int = 2048
    indexer_bytes_per_token: int = 1024
    csa_compressor_bytes_per_token: int = 1024
    mega_compressed_topk_cap: int = 0
    mega_grid_sync_bytes: int = 0
    mega_splitk_scratch_bytes_per_rank: int = 0
    mega_csa_score_scratch_bytes_per_rank: int = 0
    scratch_bytes_per_rank: int = 0
    protocol_bytes_per_rank: int = _DEFAULT_PROTOCOL_BYTES
    align_bytes: int = _DEFAULT_ALIGN_BYTES
    page_rr: bool = False

    def __post_init__(self) -> None:
        if self.cp_size != 8:
            raise ValueError(f"V0 requires cp_size=8, got {self.cp_size}")
        if self.max_tokens_per_rank <= 0:
            raise ValueError(
                f"max_tokens_per_rank must be positive, got {self.max_tokens_per_rank}"
            )
        if self.batch_cap <= 0:
            raise ValueError(f"batch_cap must be positive, got {self.batch_cap}")
        if self.swa_bytes_per_token <= 0:
            raise ValueError(
                f"swa_bytes_per_token must be positive, got {self.swa_bytes_per_token}"
            )
        if self.main_bytes_per_token <= 0:
            raise ValueError(
                f"main_bytes_per_token must be positive, got {self.main_bytes_per_token}"
            )
        if self.indexer_bytes_per_token <= 0:
            raise ValueError(
                f"indexer_bytes_per_token must be positive, got {self.indexer_bytes_per_token}"
            )
        if self.csa_compressor_bytes_per_token < 0:
            raise ValueError(
                "csa_compressor_bytes_per_token must be non-negative, got "
                f"{self.csa_compressor_bytes_per_token}"
            )
        if self.mega_compressed_topk_cap < 0:
            raise ValueError(
                f"mega_compressed_topk_cap must be non-negative, got {self.mega_compressed_topk_cap}"
            )
        if self.mega_grid_sync_bytes < 0:
            raise ValueError(
                f"mega_grid_sync_bytes must be non-negative, got {self.mega_grid_sync_bytes}"
            )
        if self.mega_splitk_scratch_bytes_per_rank < 0:
            raise ValueError(
                "mega_splitk_scratch_bytes_per_rank must be non-negative, got "
                f"{self.mega_splitk_scratch_bytes_per_rank}"
            )
        if self.mega_csa_score_scratch_bytes_per_rank < 0:
            raise ValueError(
                "mega_csa_score_scratch_bytes_per_rank must be non-negative, got "
                f"{self.mega_csa_score_scratch_bytes_per_rank}"
            )
        if self.page_rr:
            raise ValueError("V0 distributed attention buffer does not support page/RR")

    @property
    def mega_side_effect_scratch_bytes_per_rank(self) -> int:
        if (
            int(self.mega_compressed_topk_cap) <= 0
            and int(self.mega_grid_sync_bytes) <= 0
            and int(self.mega_splitk_scratch_bytes_per_rank) <= 0
            and int(self.mega_csa_score_scratch_bytes_per_rank) <= 0
        ):
            return 0
        token_cap = int(self.max_tokens_per_rank)
        align = int(self.align_bytes)
        swa_stage = _align_up(
            int(self.mega_grid_sync_bytes)
            + token_cap * int(self.swa_bytes_per_token),
            align,
        )
        csa_compressor_stage = _align_up(
            token_cap * int(self.csa_compressor_bytes_per_token),
            align,
        )
        main_compressor_stage = _align_up(
            token_cap * int(self.main_bytes_per_token) * 2,
            align,
        )
        compressed_index_stage = _align_up(
            token_cap * 8,
            align,
        ) + token_cap * max(int(self.mega_compressed_topk_cap), 1) * 4
        side_effect_stage = (
            swa_stage
            + csa_compressor_stage
            + main_compressor_stage
            + compressed_index_stage
        )
        if side_effect_stage == 0:
            return 0
        return (
            side_effect_stage
            + int(self.mega_splitk_scratch_bytes_per_rank)
            + int(self.mega_csa_score_scratch_bytes_per_rank)
        )

    @property
    def stage_bytes_per_rank(self) -> int:
        token_cap = int(self.max_tokens_per_rank)
        persistent_bytes = token_cap * (
            int(self.main_bytes_per_token) + int(self.indexer_bytes_per_token)
        )
        transient_bytes = max(
            token_cap * int(self.swa_bytes_per_token),
            token_cap * int(self.main_bytes_per_token),
            token_cap * int(self.indexer_bytes_per_token),
            self.mega_side_effect_scratch_bytes_per_rank,
            int(self.scratch_bytes_per_rank),
        )
        return persistent_bytes + transient_bytes

    @property
    def per_rank_bytes(self) -> int:
        return _align_up(
            int(self.protocol_bytes_per_rank) + self.stage_bytes_per_rank,
            int(self.align_bytes),
        )

    @property
    def total_bytes(self) -> int:
        return self.per_rank_bytes * int(self.cp_size)

    def validate_request(self, *, tokens_per_rank: int, batch_size: int) -> None:
        tokens_per_rank = int(tokens_per_rank)
        batch_size = int(batch_size)
        if tokens_per_rank < 0:
            raise ValueError(
                f"tokens_per_rank must be non-negative, got {tokens_per_rank}"
            )
        if tokens_per_rank > self.max_tokens_per_rank:
            raise ValueError(
                f"tokens_per_rank={tokens_per_rank} exceeds distributed attention "
                f"buffer capacity {self.max_tokens_per_rank}"
            )
        if batch_size <= 0 or batch_size > self.batch_cap:
            raise ValueError(
                f"batch_size={batch_size} is outside distributed attention buffer "
                f"capacity [1, {self.batch_cap}]"
            )

    def cache_key(self, group_id: int) -> tuple:
        return (
            int(group_id),
            int(self.cp_size),
            int(self.max_tokens_per_rank),
            int(self.batch_cap),
            int(self.swa_bytes_per_token),
            int(self.main_bytes_per_token),
            int(self.indexer_bytes_per_token),
            int(self.csa_compressor_bytes_per_token),
            int(self.mega_compressed_topk_cap),
            int(self.mega_grid_sync_bytes),
            int(self.mega_splitk_scratch_bytes_per_rank),
            int(self.mega_csa_score_scratch_bytes_per_rank),
            int(self.scratch_bytes_per_rank),
            int(self.protocol_bytes_per_rank),
            int(self.align_bytes),
            bool(self.page_rr),
        )


@dataclass
class Dsv4CpAttentionBuffer:
    spec: Dsv4CpAttentionBufferSpec
    communicator: Any = None
    symm_buffer: Optional[torch.Tensor] = None
    symm_handle: Any = None

    @property
    def comm_ptr(self) -> int:
        if self.communicator is None:
            return 0
        return int(getattr(self.communicator, "_communicator_ptr"))

    @property
    def buffer_handle(self) -> int:
        if self.communicator is None:
            return -1
        return int(getattr(self.communicator, "_ub_handle"))

    @property
    def signal_handle(self) -> int:
        if self.communicator is None:
            return -1
        return int(getattr(self.communicator, "_gpu_ptr_handle"))

    @property
    def rank_offsets(self) -> list[int]:
        if self.communicator is None:
            return [
                rank * int(self.spec.per_rank_bytes)
                for rank in range(int(self.spec.cp_size))
            ]
        offsets = getattr(self.communicator, "_rank_offset_lists")
        return [int(x) for x in offsets]

    @property
    def symm_buffer_ptrs_dev(self) -> int:
        if self.symm_handle is None:
            return 0
        return int(getattr(self.symm_handle, "buffer_ptrs_dev", 0))

    @property
    def symm_signal_pad_ptrs_dev(self) -> int:
        if self.symm_handle is None:
            return 0
        return int(getattr(self.symm_handle, "signal_pad_ptrs_dev", 0))

    def validate_request(self, *, tokens_per_rank: int, batch_size: int) -> None:
        self.spec.validate_request(
            tokens_per_rank=tokens_per_rank,
            batch_size=batch_size,
        )

    def op_kwargs(self, *, cp_rank: int) -> dict[str, Any]:
        kwargs = {
            "cp_rank": int(cp_rank),
            "cp_size": int(self.spec.cp_size),
            "comm_ptr": self.comm_ptr,
            "buffer_handle": self.buffer_handle,
            "signal_handle": self.signal_handle,
            "per_rank_buffer_bytes": int(self.spec.per_rank_bytes),
            "rank_offsets": self.rank_offsets,
        }
        if self.symm_buffer is not None and self.symm_handle is not None:
            signal_pad_ptrs_dev = self.symm_signal_pad_ptrs_dev
            if os.environ.get("DSV4_CP_ATTENTION_FORCE_PAYLOAD_BARRIER", "0") == "1":
                signal_pad_ptrs_dev = 0
            kwargs.update(
                {
                    "symm_buffer": self.symm_buffer,
                    "symm_buffer_ptrs_dev": self.symm_buffer_ptrs_dev,
                    "symm_signal_pad_ptrs_dev": signal_pad_ptrs_dev,
                    "symm_handle": self.symm_handle,
                }
            )
        return kwargs


def build_dsv4_cp_attention_buffer_spec(
    *,
    cp_size: int,
    actual_tokens_per_rank: int,
    batch_size: int,
    swa_bytes_per_token: int,
    page_rr: bool,
) -> Dsv4CpAttentionBufferSpec:
    max_tokens = _env_int(_ENV_MAX_TOKENS, max(int(actual_tokens_per_rank), 1))
    mega_topk_cap = _env_int(
        _ENV_MEGA_COMPRESSED_TOPK_CAP,
        _DEFAULT_MEGA_COMPRESSED_TOPK_CAP,
    )
    mega_splitk_scratch = _env_int(
        _ENV_MEGA_SPLITK_SCRATCH_BYTES,
        _DEFAULT_MEGA_SPLITK_SCRATCH_BYTES,
    )
    # CSA score prebuild stores one float score per local row and visible
    # compressed candidate. The CUDA side uses
    # min(max(compressed_region_width, compressed_topk), kMaxCompressedTopK).
    # Approximate compressed_region_width from the rank-consistent token cap so
    # 4k total tokens (514/rank) reserve 1028 candidates, not just the 1024
    # default topK cap.
    csa_score_stride_cap = min(
        max((max(int(max_tokens), 1) * int(cp_size) + 3) // 4, max(int(mega_topk_cap), 1)),
        4096,
    )
    default_csa_score_scratch = max(int(max_tokens), 1) * csa_score_stride_cap * 4
    mega_csa_score_scratch = _env_int(
        _ENV_MEGA_CSA_SCORE_SCRATCH_BYTES,
        default_csa_score_scratch,
    )
    spec = Dsv4CpAttentionBufferSpec(
        cp_size=int(cp_size),
        max_tokens_per_rank=max_tokens,
        batch_cap=max(int(batch_size), 1),
        swa_bytes_per_token=int(swa_bytes_per_token),
        mega_compressed_topk_cap=max(int(mega_topk_cap), 0),
        mega_grid_sync_bytes=_DEFAULT_MEGA_GRID_SYNC_BYTES,
        mega_splitk_scratch_bytes_per_rank=max(int(mega_splitk_scratch), 0),
        mega_csa_score_scratch_bytes_per_rank=max(int(mega_csa_score_scratch), 0),
        page_rr=bool(page_rr),
    )
    spec.validate_request(
        tokens_per_rank=int(actual_tokens_per_rank),
        batch_size=int(batch_size),
    )
    return spec


def get_or_create_dsv4_cp_attention_buffer(
    *,
    group: Any,
    cp_rank: int,
    spec: Dsv4CpAttentionBufferSpec,
    communicator_factory: Optional[Callable[..., Any]] = None,
) -> Dsv4CpAttentionBuffer:
    """Return the process-local attention buffer for ``spec``.

    ``communicator_factory`` is injectable for CPU unit tests and the legacy
    single-node CUDA-IPC prototype.  Production uses PyTorch symmetric memory,
    matching the DeepGEMM/MegaMoE allocation/rendezvous path.  No hot-path
    exchange is entered here; only allocation-time rendezvous and zeroing run.
    """

    group_id = id(group)
    key = spec.cache_key(group_id)
    cached = _BUFFER_CACHE.get(key)
    if cached is not None:
        return cached

    if communicator_factory is None:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed attention symmetric memory requires CUDA")
        local_rank_raw = os.environ.get("LOCAL_RANK")
        if local_rank_raw is not None and local_rank_raw != "":
            local_rank = int(local_rank_raw)
        elif torch.cuda.device_count() > 0:
            local_rank = int(cp_rank) % int(torch.cuda.device_count())
        else:
            local_rank = int(cp_rank)
        torch.cuda.set_device(local_rank)

        import torch.distributed._symmetric_memory as symm_mem

        symm_buffer = symm_mem.empty(
            int(spec.total_bytes),
            dtype=torch.int8,
            device=torch.device(f"cuda:{local_rank}"),
        )
        symm_handle = symm_mem.rendezvous(symm_buffer, group=group)
        symm_buffer.zero_()
        if hasattr(symm_handle, "get_signal_pad"):
            signal_pad_size = int(getattr(symm_handle, "signal_pad_size", 0))
            if signal_pad_size > 0:
                signal_pad = symm_handle.get_signal_pad(
                    int(cp_rank),
                    (signal_pad_size // 4,),
                    torch.uint32,
                    0,
                )
                start = int(_SIGNAL_PAD_CHANNEL_BASE) * int(spec.cp_size)
                end = min(
                    int(signal_pad.numel()),
                    (int(_SIGNAL_PAD_CHANNEL_BASE) + int(_SIGNAL_PAD_CHANNEL_COUNT))
                    * int(spec.cp_size),
                )
                if start < end:
                    signal_pad[start:end].zero_()
                if os.environ.get("DSV4_CP_ATTENTION_DEBUG_PY", "0") == "1":
                    logging.warning(
                        "[DSV4 CP Attention Py] symm signal pad cp_rank=%s "
                        "size_bytes=%s u32=%s zero_range=[%s,%s) channel_base=%s "
                        "channel_count=%s",
                        int(cp_rank),
                        int(signal_pad_size),
                        int(signal_pad.numel()),
                        int(start),
                        int(end),
                        int(_SIGNAL_PAD_CHANNEL_BASE),
                        int(_SIGNAL_PAD_CHANNEL_COUNT),
                    )
        if hasattr(group, "barrier"):
            group.barrier()
        else:
            dist.barrier(group=group)
        torch.cuda.synchronize()
        buf = Dsv4CpAttentionBuffer(
            spec=spec,
            symm_buffer=symm_buffer,
            symm_handle=symm_handle,
        )
    else:
        local_rank_raw = os.environ.get("LOCAL_RANK")
        if local_rank_raw is not None and local_rank_raw != "":
            local_rank = int(local_rank_raw)
        elif torch.cuda.is_available() and torch.cuda.device_count() > 0:
            local_rank = int(cp_rank) % int(torch.cuda.device_count())
        else:
            local_rank = int(cp_rank)

        communicator = communicator_factory(
            group,
            int(local_rank),
            int(spec.cp_size),
            int(spec.total_bytes),
        )
        buf = Dsv4CpAttentionBuffer(spec=spec, communicator=communicator)
    _BUFFER_CACHE[key] = buf
    logging.info(
        "[DSV4 CP Attention] allocated %s symmetric buffer: cp_size=%d "
        "rank=%d max_tokens_per_rank=%d batch_cap=%d per_rank=%.3f MiB "
        "total=%.3f MiB",
        "torch" if communicator_factory is None else "userbuffer",
        spec.cp_size,
        int(cp_rank),
        spec.max_tokens_per_rank,
        spec.batch_cap,
        spec.per_rank_bytes / (1024**2),
        spec.total_bytes / (1024**2),
    )
    return buf


def clear_dsv4_cp_attention_buffer_cache() -> None:
    _BUFFER_CACHE.clear()
