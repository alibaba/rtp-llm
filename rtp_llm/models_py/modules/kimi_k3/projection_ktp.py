"""Projection-only KTP planning and tensor layout for Kimi K3 Decode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather,
    all_to_all_single,
)
from rtp_llm.models_py.modules.factory.linear.quantized_activation import (
    QuantizedActivation,
)


def resolve_projection_local_heads(
    *, total_heads: int, attention_tp_size: int, ktp_size: int
) -> int:
    """Resolve the checkpoint-local projection heads for this topology.

    Projection-KTP replaces attention-TP sharding only when KTP is enabled.
    Prefill and legacy KTP1 Decode retain their attention-TP-local weights.
    """

    if attention_tp_size <= 0 or ktp_size <= 0:
        raise ValueError("attention TP and KTP sizes must be positive")
    projection_parallel_size = ktp_size if ktp_size > 1 else attention_tp_size
    if total_heads % projection_parallel_size:
        raise ValueError(
            f"KDA heads {total_heads} must be divisible by projection parallel "
            f"size {projection_parallel_size}"
        )
    return total_heads // projection_parallel_size


def validate_projection_ktp_sp_type(sp_type: str) -> str:
    """Validate the speculative session type for the score-model KTP path.

    ``PyModelInitResources.is_speculative`` describes the whole serving
    session, so it is true for both target and draft models. Target versus
    draft is instead encoded by the model's parallelism: the target model keeps
    KTP while Eagle3 and K3 MTP proposal construction force the draft model to
    KTP1.
    """

    normalized = str(sp_type).strip().lower()
    if normalized not in ("", "eagle3", "mtp"):
        raise RuntimeError(
            "Projection KTP supports ordinary Decode, Eagle3 target "
            "verification, or K3 MTP target verification; the speculative "
            "model itself must use KTP1, "
            f"got SP_TYPE={normalized!r}"
        )
    return normalized


@dataclass(frozen=True)
class KtpProjectionResult:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    raw_gate: torch.Tensor
    raw_beta: torch.Tensor
    output_gate: torch.Tensor


def _apply_projection(
    inputs: torch.Tensor | QuantizedActivation, projection: Any
) -> torch.Tensor:
    """Apply either a dense weight tensor or a quantized Linear module."""

    if isinstance(projection, torch.Tensor):
        return torch.matmul(inputs, projection)
    return projection(inputs.contiguous())


def _all_gather_projection_input(
    local_input: torch.Tensor | QuantizedActivation,
    *,
    ktp_size: int,
) -> torch.Tensor | QuantizedActivation:
    """Gather either BF16 rows or the complete group128 FP8 wire format.

    ``QuantizedActivation`` is deliberately not a Tensor subclass.  Its FP8
    values and UE8M0 scales therefore have to be communicated independently,
    then repacked into the row-major global KTP order expected by the FP8
    projection.  Per-rank scale padding is removed before the global wire is
    assembled; otherwise graph buckets 1 and 2 would attach the wrong scale to
    rows from ranks after rank 0.
    """

    if not isinstance(local_input, QuantizedActivation):
        return all_gather(local_input.contiguous(), group=Group.KTP)

    local_rows, hidden_size = local_input.shape
    local_scale_columns = int(local_input.scale_wire.shape[1])
    scale_groups = int(local_input.scale_wire.shape[0])

    gathered_values_wire = all_gather(
        local_input.values.view(torch.uint8).contiguous(), group=Group.KTP
    )
    gathered_values = gathered_values_wire.view(torch.float8_e4m3fn)
    expected_values_shape = (ktp_size * local_rows, hidden_size)
    if tuple(gathered_values.shape) != expected_values_shape:
        raise RuntimeError(
            "KTP FP8 value AllGather returned shape "
            f"{tuple(gathered_values.shape)}, expected {expected_values_shape}"
        )

    gathered_scale_wire = all_gather(
        local_input.scale_wire.contiguous(), group=Group.KTP
    )
    expected_scale_shape = (ktp_size * scale_groups, local_scale_columns)
    if tuple(gathered_scale_wire.shape) != expected_scale_shape:
        raise RuntimeError(
            "KTP FP8 scale AllGather returned shape "
            f"{tuple(gathered_scale_wire.shape)}, expected {expected_scale_shape}"
        )

    # all_gather concatenates dim 0, yielding [rank, K-group, local-M-pad].
    # Drop each rank's local alignment columns before concatenating M globally.
    scale_by_rank = gathered_scale_wire.reshape(
        ktp_size, scale_groups, local_scale_columns
    )
    global_scale_wire = (
        scale_by_rank[:, :, :local_rows]
        .permute(1, 0, 2)
        .contiguous()
        .reshape(scale_groups, ktp_size * local_rows)
    )
    return QuantizedActivation(gathered_values, global_scale_wire)


def pack_ktp_projection_payload(
    gathered_hidden: torch.Tensor,
    fused_projection: Any,
    forget_up_projection: Any,
    *,
    total_heads: int,
    head_dim: int,
    forget_latent_size: int,
    ktp_size: int,
    ktp_rank: int,
) -> torch.Tensor:
    """Project all owner rows with one rank's head shard and pack A2A input."""

    local_heads = total_heads // ktp_size
    local_projection_size = local_heads * head_dim
    projected = _apply_projection(gathered_hidden, fused_projection)
    q, k, v, output_gate, forget_latent, full_raw_beta = torch.split(
        projected,
        [
            local_projection_size,
            local_projection_size,
            local_projection_size,
            local_projection_size,
            forget_latent_size,
            total_heads,
        ],
        dim=-1,
    )
    raw_gate = _apply_projection(forget_latent, forget_up_projection)
    raw_beta = full_raw_beta.narrow(1, ktp_rank * local_heads, local_heads)
    return torch.cat((q, k, v, output_gate, raw_gate, raw_beta), dim=-1)


def reassemble_ktp_projection_payload(
    received: torch.Tensor,
    *,
    ktp_size: int,
    physical_batch: int,
    local_projection_size: int,
    local_heads: int,
) -> KtpProjectionResult:
    """Convert source-major A2A payload into owner-major full-head tensors."""

    payload_width = 5 * local_projection_size + local_heads
    expected = (ktp_size * physical_batch, payload_width)
    if tuple(received.shape) != expected:
        raise ValueError(
            f"KTP A2A payload shape {tuple(received.shape)} != expected {expected}"
        )
    source_major = received.reshape(ktp_size, physical_batch, payload_width)
    sections = torch.split(
        source_major,
        [
            local_projection_size,
            local_projection_size,
            local_projection_size,
            local_projection_size,
            local_projection_size,
            local_heads,
        ],
        dim=-1,
    )

    def _heads_full(section: torch.Tensor) -> torch.Tensor:
        return section.permute(1, 0, 2).contiguous().reshape(physical_batch, -1)

    q, k, v, output_gate, raw_gate, raw_beta = (
        _heads_full(section) for section in sections
    )
    return KtpProjectionResult(q, k, v, raw_gate, raw_beta, output_gate)


def project_kda_inputs_ktp(
    hidden_states: torch.Tensor,
    fused_projection: Any,
    forget_up_projection: Any,
    *,
    total_heads: int,
    head_dim: int,
    forget_latent_size: int,
    ktp_size: int,
    ktp_rank: int,
) -> KtpProjectionResult:
    """Run KDA's projection-only KTP AllGather/GEMM/AllToAll pipeline."""

    physical_batch = int(hidden_states.shape[0])
    local_heads = total_heads // ktp_size
    local_projection_size = local_heads * head_dim
    gathered_hidden = _all_gather_projection_input(
        hidden_states,
        ktp_size=ktp_size,
    )
    send = pack_ktp_projection_payload(
        gathered_hidden,
        fused_projection,
        forget_up_projection,
        total_heads=total_heads,
        head_dim=head_dim,
        forget_latent_size=forget_latent_size,
        ktp_size=ktp_size,
        ktp_rank=ktp_rank,
    )
    received = all_to_all_single(send, group=Group.KTP)
    return reassemble_ktp_projection_payload(
        received,
        ktp_size=ktp_size,
        physical_batch=physical_batch,
        local_projection_size=local_projection_size,
        local_heads=local_heads,
    )


__all__ = [
    "KtpProjectionResult",
    "pack_ktp_projection_payload",
    "project_kda_inputs_ktp",
    "reassemble_ktp_projection_payload",
    "resolve_projection_local_heads",
    "validate_projection_ktp_sp_type",
]
