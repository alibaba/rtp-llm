"""Projection-only KTP planning and tensor layout for Kimi K3 Decode."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Sequence

import torch

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather,
    all_to_all_single,
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


class KtpForwardMode(IntEnum):
    DECODE = 0
    PREFILL = 1
    TARGET_VERIFY = 2
    MTP_DRAFT_UPDATE = 3


@dataclass(frozen=True)
class KtpStepPlan:
    valid_batch_sizes: tuple[int, ...]
    global_max_batch: int
    common_physical_batch: int
    common_graph_bucket: int
    use_cuda_graph: bool
    all_idle: bool


def normalize_capture_buckets(values: Iterable[int]) -> tuple[int, ...]:
    buckets = tuple(sorted(set(int(value) for value in values)))
    if any(value <= 0 for value in buckets):
        raise ValueError(f"Decode capture buckets must be positive, got {buckets}")
    return buckets


def parse_decode_capture_config(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    try:
        return normalize_capture_buckets(
            int(part.strip()) for part in raw.split(",") if part.strip()
        )
    except ValueError as error:
        raise ValueError(f"Invalid DECODE_CAPTURE_CONFIG={raw!r}: {error}") from error


def default_decode_capture_buckets(max_batch: int) -> tuple[int, ...]:
    """Mirror ``CudaGraphRunner::getDecodeBatchSizesToCapture`` defaults."""

    if max_batch <= 0:
        raise ValueError(f"max_batch must be positive, got {max_batch}")
    buckets = [value for value in (1, 8, 16, 24, 32) if value <= max_batch]
    buckets.extend(range(48, max_batch + 1, 16))
    if not buckets or buckets[-1] != max_batch:
        buckets.append(max_batch)
    return normalize_capture_buckets(buckets)


def _pad_dim(
    tensor: torch.Tensor, rows: int, *, dim: int, value: int = 0
) -> torch.Tensor:
    if tensor is None or not tensor.numel() or tensor.shape[dim] == rows:
        return tensor
    if tensor.shape[dim] > rows:
        raise ValueError(
            f"cannot shrink KTP tensor dim {dim} from {tensor.shape[dim]} "
            f"to {rows} rows"
        )
    preserve_pinned = tensor.device.type == "cpu" and tensor.is_pinned()
    padding_shape = list(tensor.shape)
    padding_shape[dim] = rows - tensor.shape[dim]
    padding = torch.full(
        padding_shape,
        value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    result = torch.cat((tensor, padding), dim=dim)
    return result.pin_memory() if preserve_pinned else result


def _pad_dim0(tensor: torch.Tensor, rows: int, value: int = 0) -> torch.Tensor:
    return _pad_dim(tensor, rows, dim=0, value=value)


def _pad_optional_tensor_attr(obj, name: str, rows: int, value: int = 0) -> None:
    """Pad a bound Tensor attribute without assigning ``None`` back to pybind.

    Undefined ``torch::Tensor`` members are exposed as ``None`` by pybind, while
    their setters only accept an actual Tensor.  Several Decode-only block-table
    aliases are intentionally undefined for HybridPool inputs, so writing the
    unchanged ``None`` value back would raise before the model forward starts.
    """

    tensor = getattr(obj, name)
    if tensor is not None:
        setattr(obj, name, _pad_dim0(tensor, rows, value))


def _pad_block_table_attr(obj, name: str, rows: int) -> None:
    """Pad either ``[batch, blocks]`` or ``[group, batch, blocks]`` tables."""

    tensor = getattr(obj, name)
    if tensor is None:
        return
    if tensor.dim() == 2:
        batch_dim = 0
    elif tensor.dim() == 3:
        batch_dim = 1
    else:
        raise ValueError(
            f"KTP block table {name} must be rank 2 or 3, got {tensor.dim()}"
        )
    setattr(obj, name, _pad_dim(tensor, rows, dim=batch_dim, value=0))


def pad_ktp_decode_inputs(inputs, plan: KtpStepPlan, *, ktp_rank: int) -> None:
    """Pad one rank's ordinary Decode inputs to the group physical batch."""

    attention = inputs.attention_inputs
    current = int(attention.input_lengths.shape[0])
    physical = int(plan.common_physical_batch)
    # Every idle rank already owns one framework fake row whose block-0 mapping
    # is the established scratch convention.  Keep that row for the no-op wave;
    # the executor may subsequently elide all-idle waves without changing this
    # tensor contract.
    if plan.all_idle:
        physical = max(current, 1)
    if physical < current:
        raise RuntimeError(
            f"KTP physical batch {physical} is smaller than local batch {current}"
        )

    inputs.input_ids = _pad_dim0(inputs.input_ids.reshape(-1), physical, 0)
    attention.input_lengths = _pad_dim0(attention.input_lengths, physical, 1)
    _pad_optional_tensor_attr(attention, "input_lengths_host", physical, 1)
    attention.sequence_lengths = _pad_dim0(attention.sequence_lengths, physical, 0)
    _pad_optional_tensor_attr(attention, "sequence_lengths_host", physical, 0)
    _pad_optional_tensor_attr(attention, "sequence_lengths_plus_1_d", physical, 1)
    _pad_block_table_attr(attention, "kv_cache_kernel_block_id_device", physical)
    _pad_block_table_attr(attention, "kv_cache_kernel_block_id_host", physical)
    _pad_block_table_attr(attention, "kv_cache_block_id_device", physical)
    _pad_block_table_attr(attention, "kv_cache_block_id_host", physical)
    attention.kv_cache_kernel_block_id_device_by_group = [
        _pad_dim0(tensor, physical, 0)
        for tensor in attention.kv_cache_kernel_block_id_device_by_group
    ]
    attention.kv_cache_kernel_block_id_host_by_group = [
        _pad_dim0(tensor, physical, 0)
        for tensor in attention.kv_cache_kernel_block_id_host_by_group
    ]
    attention.kv_cache_block_id_host_by_group = [
        _pad_dim0(tensor, physical, 0)
        for tensor in attention.kv_cache_block_id_host_by_group
    ]

    device = attention.input_lengths.device
    cu_seqlens = torch.arange(
        physical + 1, dtype=torch.int32, device=device
    )
    attention.cu_seqlens = cu_seqlens
    attention.decode_cu_seqlens_d = cu_seqlens
    host_cu = torch.arange(physical + 1, dtype=torch.int32, device="cpu")
    if (
        attention.cu_seqlens_host is not None
        and attention.cu_seqlens_host.numel()
        and attention.cu_seqlens_host.is_pinned()
    ):
        host_cu = host_cu.pin_memory()
    attention.cu_seqlens_host = host_cu
    # decode_cu_seqlens_host is a read-only pybind view and is normally
    # undefined; decode_cu_seqlens_d is the runtime metadata consumed by K3.
    if attention.cu_kv_seqlens is not None and attention.cu_kv_seqlens.numel():
        attention.cu_kv_seqlens = cu_seqlens
    attention.total_tokens = physical
    _pad_optional_tensor_attr(attention, "padding_offset", physical, 0)
    local_real_batch = plan.valid_batch_sizes[ktp_rank]
    attention.is_s_padded = physical != local_real_batch

    mask = torch.zeros(physical, dtype=torch.int32, device=device)
    mask[:local_real_batch] = 1
    inputs.ktp_valid_batch_sizes = torch.tensor(
        plan.valid_batch_sizes, dtype=torch.int32, device=device
    )
    inputs.ktp_valid_row_mask = mask
    inputs.ktp_local_real_batch = local_real_batch
    inputs.ktp_common_physical_batch = physical
    inputs.ktp_common_graph_bucket = plan.common_graph_bucket
    inputs.ktp_use_cuda_graph = plan.use_cuda_graph
    inputs.ktp_all_idle = plan.all_idle


def build_ktp_step_plan(
    metadata: Sequence[Sequence[int]],
    capture_buckets: Iterable[int],
) -> KtpStepPlan:
    """Build one deterministic plan from rank-ordered KTP metadata.

    Each row is ``[local_real_batch, graph_eligible, forward_mode]``.
    This pure helper is deliberately separated from the collective so topology,
    bucket and fallback behaviour can be exhaustively unit tested.
    """

    if not metadata:
        raise ValueError("KTP metadata must contain at least one rank")
    rows = tuple(tuple(int(value) for value in row) for row in metadata)
    if any(len(row) != 3 for row in rows):
        raise ValueError(f"KTP metadata rows must have width 3, got {rows}")
    batches = tuple(row[0] for row in rows)
    if any(batch < 0 for batch in batches):
        raise ValueError(f"KTP local batch sizes must be non-negative, got {batches}")
    modes = {row[2] for row in rows}
    if len(modes) != 1:
        raise RuntimeError(f"KTP ranks disagree on forward mode: {rows}")

    global_max = max(batches)
    all_idle = global_max == 0
    buckets = normalize_capture_buckets(capture_buckets)
    graph_bucket = next((value for value in buckets if value >= global_max), 0)
    use_graph = (
        not all_idle
        and graph_bucket > 0
        and all(bool(row[1]) for row in rows)
    )
    physical_batch = graph_bucket if use_graph else global_max
    return KtpStepPlan(
        valid_batch_sizes=batches,
        global_max_batch=global_max,
        common_physical_batch=physical_batch,
        common_graph_bucket=graph_bucket if use_graph else 0,
        use_cuda_graph=use_graph,
        all_idle=all_idle,
    )


def coordinate_ktp_step(
    *,
    local_real_batch: int,
    graph_eligible: bool,
    forward_mode: KtpForwardMode,
    capture_buckets: Iterable[int],
    device: torch.device,
    ktp_size: int,
) -> KtpStepPlan:
    """AllGather fixed-width metadata and perform exactly one D2H transfer."""

    local = [int(local_real_batch), int(graph_eligible), int(forward_mode)]
    if ktp_size <= 1:
        return build_ktp_step_plan([local], capture_buckets)
    metadata_d = torch.tensor([local], dtype=torch.int32, device=device)
    gathered_d = all_gather(metadata_d, group=Group.KTP).reshape(ktp_size, 3)
    # The coordinator is outside CUDA Graph.  One compact D2H is intentional:
    # all subsequent planning is host-only and identical on every rank.
    metadata_h = gathered_d.cpu()
    return build_ktp_step_plan(metadata_h.tolist(), capture_buckets)


@dataclass(frozen=True)
class KtpProjectionResult:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    raw_gate: torch.Tensor
    raw_beta: torch.Tensor
    output_gate: torch.Tensor


def pack_ktp_projection_payload(
    gathered_hidden: torch.Tensor,
    fused_projection: torch.Tensor,
    forget_up_projection: torch.Tensor,
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
    projected = torch.matmul(gathered_hidden, fused_projection)
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
    raw_gate = torch.matmul(forget_latent, forget_up_projection)
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
    fused_projection: torch.Tensor,
    forget_up_projection: torch.Tensor,
    *,
    total_heads: int,
    head_dim: int,
    forget_latent_size: int,
    ktp_size: int,
    ktp_rank: int,
) -> KtpProjectionResult:
    """Run KDA's projection-only KTP AllGather/GEMM/AllToAll pipeline."""

    if total_heads % ktp_size:
        raise ValueError(
            f"KDA heads {total_heads} must be divisible by KTP {ktp_size}"
        )
    physical_batch = int(hidden_states.shape[0])
    local_heads = total_heads // ktp_size
    local_projection_size = local_heads * head_dim
    gathered_hidden = all_gather(hidden_states.contiguous(), group=Group.KTP)
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
    "KtpForwardMode",
    "KtpProjectionResult",
    "KtpStepPlan",
    "build_ktp_step_plan",
    "coordinate_ktp_step",
    "default_decode_capture_buckets",
    "normalize_capture_buckets",
    "pack_ktp_projection_payload",
    "parse_decode_capture_config",
    "pad_ktp_decode_inputs",
    "project_kda_inputs_ktp",
    "reassemble_ktp_projection_payload",
]
