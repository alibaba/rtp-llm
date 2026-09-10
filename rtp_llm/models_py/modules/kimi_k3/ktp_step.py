"""Step coordination and input padding for Kimi K3 Projection-KTP Decode."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Sequence

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather


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
    forward_mode: KtpForwardMode
    tokens_per_batch: int


def normalize_capture_buckets(values: Iterable[int]) -> tuple[int, ...]:
    buckets = tuple(sorted(set(int(value) for value in values)))
    if any(value <= 0 for value in buckets):
        raise ValueError(f"Decode capture buckets must be positive, got {buckets}")
    return buckets


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


def _pad_token_values(
    tensor: torch.Tensor,
    logical_tokens: int,
    physical_tokens: int,
    value: int = 0,
) -> torch.Tensor:
    """Pad flattened per-token metadata using the validated MTP contract."""

    if tensor is None or not tensor.numel() or logical_tokens == physical_tokens:
        return tensor
    if logical_tokens <= 0 or tensor.numel() % logical_tokens:
        raise ValueError(
            "token metadata cannot be padded: "
            f"numel={tensor.numel()} logical_tokens={logical_tokens}"
        )
    values_per_token = tensor.numel() // logical_tokens
    return _pad_dim0(tensor.reshape(-1), physical_tokens * values_per_token, value)


def _pad_optional_token_attr(
    obj, name: str, logical_tokens: int, physical_tokens: int, value: int = 0
) -> None:
    if obj is None:
        return
    tensor = getattr(obj, name, None)
    if tensor is not None:
        setattr(
            obj,
            name,
            _pad_token_values(tensor, logical_tokens, physical_tokens, value),
        )


def _pad_optional_tensor_attr(obj, name: str, rows: int, value: int = 0) -> None:
    """Pad an optional bound Tensor without assigning ``None`` to pybind."""

    tensor = getattr(obj, name, None)
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


def _pad_cumulative_lengths(
    tensor: torch.Tensor,
    request_rows: int,
    physical_batch: int,
    token_width: int,
) -> torch.Tensor:
    """Preserve real cumulative lengths and append scratch-request entries."""

    if tensor is None or not tensor.numel() or physical_batch == request_rows:
        return tensor
    if tensor.numel() != request_rows + 1:
        raise ValueError(
            "KTP cumulative lengths must have requests + 1 entries, got "
            f"{tensor.numel()} for {request_rows} requests"
        )
    tail = tensor[-1] + torch.arange(
        1,
        physical_batch - request_rows + 1,
        dtype=tensor.dtype,
        device=tensor.device,
    ) * token_width
    return torch.cat((tensor, tail), dim=0)


def pad_ktp_decode_inputs(inputs, plan: KtpStepPlan, *, ktp_rank: int) -> None:
    """Pad one rank's Decode request slots and token rows to a common shape."""

    attention = inputs.attention_inputs
    current = int(attention.input_lengths.shape[0])
    physical = int(plan.common_physical_batch)
    # An idle rank already owns one fake row mapped to the reserved block 0.
    # Preserve it until the executor elides an all-idle wave.
    if plan.all_idle:
        physical = max(current, 1)
    if physical < current:
        raise RuntimeError(
            f"KTP physical batch {physical} is smaller than local batch {current}"
        )

    token_width = int(plan.tokens_per_batch)
    physical_tokens = physical * token_width
    current_tokens = current * token_width
    if inputs.input_ids.numel() != current_tokens:
        raise RuntimeError(
            "Projection KTP input token rows do not match request metadata: "
            f"tokens={inputs.input_ids.numel()} expected={current_tokens}"
        )
    inputs.input_ids = _pad_dim0(inputs.input_ids.reshape(-1), physical_tokens, 0)
    input_hiddens = getattr(inputs, "input_hiddens", None)
    if input_hiddens is not None and input_hiddens.numel():
        inputs.input_hiddens = _pad_dim0(input_hiddens, physical_tokens, 0)
    combo_position_ids = getattr(inputs, "combo_position_ids", None)
    if combo_position_ids is not None and combo_position_ids.numel():
        inputs.combo_position_ids = _pad_token_values(
            combo_position_ids, current_tokens, physical_tokens, 0
        )
        if getattr(attention, "combo_position_ids", None) is not None:
            attention.combo_position_ids = inputs.combo_position_ids
    embedding_inputs = getattr(inputs, "embedding_inputs", None)
    _pad_optional_token_attr(
        embedding_inputs, "combo_tokens_type_ids", current_tokens, physical_tokens, 0
    )
    _pad_optional_token_attr(
        embedding_inputs, "text_tokens_mask", current_tokens, physical_tokens, 1
    )
    bert_inputs = getattr(inputs, "bert_embedding_inputs", None)
    _pad_optional_token_attr(
        bert_inputs, "combo_position_ids", current_tokens, physical_tokens, 0
    )
    _pad_optional_token_attr(
        bert_inputs, "combo_tokens_type_ids", current_tokens, physical_tokens, 0
    )
    attention.input_lengths = _pad_dim0(
        attention.input_lengths, physical, token_width
    )
    _pad_optional_tensor_attr(attention, "input_lengths_host", physical, token_width)
    _pad_optional_tensor_attr(attention, "prefix_lengths", physical, 0)
    _pad_optional_tensor_attr(attention, "prefix_lengths_host", physical, 0)
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
    cu_seqlens = (
        torch.arange(physical + 1, dtype=torch.int32, device=device) * token_width
    )
    attention.cu_seqlens = cu_seqlens
    attention.decode_cu_seqlens_d = cu_seqlens
    host_cu = torch.arange(physical + 1, dtype=torch.int32, device="cpu") * token_width
    if (
        attention.cu_seqlens_host is not None
        and attention.cu_seqlens_host.numel()
        and attention.cu_seqlens_host.is_pinned()
    ):
        host_cu = host_cu.pin_memory()
    attention.cu_seqlens_host = host_cu
    if attention.cu_kv_seqlens is not None and attention.cu_kv_seqlens.numel():
        attention.cu_kv_seqlens = _pad_cumulative_lengths(
            attention.cu_kv_seqlens, current, physical, token_width
        )
    attention.total_tokens = physical_tokens
    _pad_optional_tensor_attr(attention, "padding_offset", physical_tokens, 0)
    local_real_batch = plan.valid_batch_sizes[ktp_rank]
    attention.is_s_padded = physical != local_real_batch

    local_real_tokens = local_real_batch * token_width
    mask = torch.zeros(physical_tokens, dtype=torch.int32, device=device)
    mask[:local_real_tokens] = 1
    inputs.ktp_valid_row_mask = mask
    inputs.ktp_local_real_batch = local_real_tokens
    inputs.ktp_common_physical_batch = physical
    inputs.ktp_use_cuda_graph = plan.use_cuda_graph
    inputs.ktp_all_idle = plan.all_idle


def build_ktp_step_plan(
    metadata: Sequence[Sequence[int]],
    capture_buckets: Iterable[int],
) -> KtpStepPlan:
    """Build a deterministic plan from rank-ordered fixed-width metadata."""

    if not metadata:
        raise ValueError("KTP metadata must contain at least one rank")
    rows = tuple(tuple(int(value) for value in row) for row in metadata)
    if any(len(row) != 4 for row in rows):
        raise ValueError(f"KTP metadata rows must have width 4, got {rows}")
    batches = tuple(row[0] for row in rows)
    if any(batch < 0 for batch in batches):
        raise ValueError(f"KTP local batch sizes must be non-negative, got {batches}")
    modes = {row[2] for row in rows}
    if len(modes) != 1:
        raise RuntimeError(f"KTP ranks disagree on forward mode: {rows}")
    token_widths = {row[3] for row in rows}
    if len(token_widths) != 1 or next(iter(token_widths)) <= 0:
        raise RuntimeError(f"KTP ranks disagree on positive token width: {rows}")

    global_max = max(batches)
    all_idle = global_max == 0
    buckets = normalize_capture_buckets(capture_buckets)
    graph_bucket = next((value for value in buckets if value >= global_max), 0)
    use_graph = (
        not all_idle and graph_bucket > 0 and all(bool(row[1]) for row in rows)
    )
    return KtpStepPlan(
        valid_batch_sizes=batches,
        global_max_batch=global_max,
        common_physical_batch=graph_bucket if use_graph else global_max,
        common_graph_bucket=graph_bucket if use_graph else 0,
        use_cuda_graph=use_graph,
        all_idle=all_idle,
        forward_mode=KtpForwardMode(next(iter(modes))),
        tokens_per_batch=next(iter(token_widths)),
    )


def coordinate_ktp_step(
    *,
    local_real_batch: int,
    graph_eligible: bool,
    forward_mode: KtpForwardMode,
    capture_buckets: Iterable[int],
    tokens_per_batch: int,
    device: torch.device,
    ktp_size: int,
) -> KtpStepPlan:
    """AllGather step metadata on the graph-external CPU control group."""

    local = [
        int(local_real_batch),
        int(graph_eligible),
        int(forward_mode),
        int(tokens_per_batch),
    ]
    if ktp_size <= 1:
        return build_ktp_step_plan([local], capture_buckets)
    # Bucket selection is control-plane work and stays outside CUDA Graph. Use
    # the dedicated Gloo group so it neither inserts an NCCL operation into the
    # model stream nor requires a device-to-host synchronization afterwards.
    del device  # Retained in the API so callers do not need a device special case.
    metadata_h = torch.tensor([local], dtype=torch.int32, device="cpu")
    gathered_h = all_gather(metadata_h, group=Group.KTP_CONTROL).reshape(
        ktp_size, 4
    )
    return build_ktp_step_plan(gathered_h.tolist(), capture_buckets)


__all__ = [
    "KtpForwardMode",
    "KtpStepPlan",
    "build_ktp_step_plan",
    "coordinate_ktp_step",
    "default_decode_capture_buckets",
    "normalize_capture_buckets",
    "pad_ktp_decode_inputs",
]
