"""Experimental K3 Decode KDA-TP / MLA request-DP helpers.

This module deliberately contains no scheduler policy.  Phase one accepts a
fixed, q_len=1 batch and assigns one contiguous, equally-sized request range to
each rank.  Keeping the layout helper separate makes the ownership contract
unit-testable before the production scheduler learns about MLA owners.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

from rtp_llm.ops import ParallelismConfig, RoleType
from rtp_llm.ops.compute_ops import PyAttentionInputs


_DECODE_KTP_ENV = "KIMI_K3_DECODE_KTP"


def decode_ktp_enabled(role_type: RoleType) -> bool:
    raw = os.environ.get(_DECODE_KTP_ENV, "0").strip()
    if raw not in ("0", "1"):
        raise ValueError(f"{_DECODE_KTP_ENV} must be 0 or 1, got {raw!r}")
    return raw == "1" and role_type == RoleType.DECODE


def logical_tp1_parallelism(config: ParallelismConfig) -> ParallelismConfig:
    """Clone a rank config for replicated MLA weights and local MLA compute."""

    result = copy.copy(config)
    result.tp_size = 1
    result.tp_rank = 0
    # MLA does not consume FFN placement, but setting these fields prevents a
    # future generic attention helper from accidentally observing TP8 there.
    result.ffn_tp_size = 1
    result.ffn_tp_rank = 0
    return result


@dataclass(frozen=True)
class DecodeOwnerLayout:
    global_batch: int
    local_batch: int
    start: int
    stop: int

    @classmethod
    def fixed(cls, global_batch: int, tp_size: int, tp_rank: int) -> "DecodeOwnerLayout":
        if tp_size <= 0 or not 0 <= tp_rank < tp_size:
            raise ValueError(f"invalid TP placement size={tp_size} rank={tp_rank}")
        if global_batch < tp_size or global_batch % tp_size:
            raise ValueError(
                "KIMI_K3_DECODE_KTP requires BS >= TP and BS divisible by TP; "
                f"got BS={global_batch}, TP={tp_size}"
            )
        local = global_batch // tp_size
        start = tp_rank * local
        return cls(global_batch, local, start, start + local)

    def narrow(self, value: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return value.narrow(dim, self.start, self.local_batch).contiguous()


def _maybe_owner_rows(
    value: Optional[torch.Tensor], layout: DecodeOwnerLayout, *, dim: int = 0
) -> Optional[torch.Tensor]:
    if value is None or not value.numel():
        return value
    if value.shape[dim] != layout.global_batch:
        raise ValueError(
            "K3 Decode owner metadata has unexpected batch dimension: "
            f"shape={tuple(value.shape)} dim={dim} BS={layout.global_batch}"
        )
    return layout.narrow(value, dim)


def _owner_group_rows(
    values: Sequence[torch.Tensor], layout: DecodeOwnerLayout
) -> list[torch.Tensor]:
    return [_maybe_owner_rows(value, layout) for value in values]


def build_owner_attention_inputs(
    attention_inputs: PyAttentionInputs,
    layout: DecodeOwnerLayout,
    *,
    device: torch.device,
    global_query_tokens: int,
) -> PyAttentionInputs:
    """Return the local request view consumed by logical-TP1 MLA Decode."""

    if attention_inputs.is_prefill:
        raise ValueError("KIMI_K3_DECODE_KTP supports Decode only")
    if bool(getattr(attention_inputs, "is_target_verify", False)):
        raise ValueError("KIMI_K3_DECODE_KTP does not support target verify")
    if bool(getattr(attention_inputs, "is_cuda_graph", False)):
        raise ValueError("KIMI_K3_DECODE_KTP phase one does not support CUDA Graph")
    if getattr(attention_inputs, "cache_store_inputs", None) is not None:
        raise ValueError("KIMI_K3_DECODE_KTP phase one does not support cache-store")

    # Decode PyAttentionInputs intentionally has no cu_seqlens_host: the C++
    # boundary only materializes that host mirror for Prefill.  The packed
    # query tensor is authoritative here.  q_len=1 therefore means exactly
    # one query token for every request in the global owner layout.
    if global_query_tokens != layout.global_batch:
        raise ValueError(
            "KIMI_K3_DECODE_KTP phase one requires q_len=1: "
            f"tokens={global_query_tokens} BS={layout.global_batch}"
        )

    local = copy.copy(attention_inputs)
    tensor_fields = (
        "input_lengths",
        "prefix_lengths",
        "sequence_lengths",
        "sequence_lengths_plus_1_d",
        "input_lengths_host",
        "prefix_lengths_host",
        "sequence_lengths_host",
    )
    for name in tensor_fields:
        value = getattr(attention_inputs, name, None)
        if value is not None and value.numel():
            setattr(local, name, _maybe_owner_rows(value, layout))

    local_lengths_host = getattr(local, "input_lengths_host", None)
    if local_lengths_host is None or local_lengths_host.numel() != layout.local_batch:
        raise ValueError("KIMI_K3_DECODE_KTP requires host input lengths")

    local.cu_seqlens = torch.arange(
        layout.local_batch + 1, dtype=torch.int32, device=device
    )
    local.cu_seqlens_host = torch.arange(layout.local_batch + 1, dtype=torch.int32)
    sequence_host = getattr(local, "sequence_lengths_host", None)
    if sequence_host is None or not sequence_host.numel():
        raise ValueError("KIMI_K3_DECODE_KTP requires host sequence lengths")
    cu_kv = [0]
    for length in sequence_host.tolist():
        cu_kv.append(cu_kv[-1] + int(length))
    local.cu_kv_seqlens = torch.tensor(cu_kv, dtype=torch.int32, device=device)
    local.padding_offset = torch.zeros(
        layout.local_batch, dtype=torch.int32, device=device
    )
    local.total_tokens = layout.local_batch
    local.context_total_kv_length = cu_kv[-1]

    for name in (
        "kv_cache_block_id_host",
        "kv_cache_kernel_block_id_host",
        "kv_cache_kernel_block_id_device",
    ):
        value = getattr(attention_inputs, name, None)
        if value is None or not value.numel():
            continue
        batch_dim = 1 if name == "kv_cache_block_id_host" and value.ndim == 3 else 0
        setattr(local, name, _maybe_owner_rows(value, layout, dim=batch_dim))

    for name in (
        "kv_cache_block_id_host_by_group",
        "kv_cache_kernel_block_id_host_by_group",
        "kv_cache_kernel_block_id_device_by_group",
    ):
        values = getattr(attention_inputs, name, None)
        if values is not None:
            setattr(local, name, _owner_group_rows(values, layout))
    return local


__all__ = [
    "DecodeOwnerLayout",
    "build_owner_attention_inputs",
    "decode_ktp_enabled",
    "logical_tp1_parallelism",
]
