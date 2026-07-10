import logging
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

import rtp_llm.ops.compute_ops as compute_ops
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, all_reduce
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FMHAImplBase,
    LinearFactory,
    RMSNorm,
    RMSResNorm,
)
from rtp_llm.models_py.modules.base.common.kvcache_store import WriteCacheStoreOp
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    CausalConv1dMetadata,
    causal_conv1d_fn,
    causal_conv1d_update,
    prepare_causal_conv1d_metadata,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated
from rtp_llm.models_py.triton_kernels.common.scatter_qkv import scatter_qkv
from rtp_llm.models_py.triton_kernels.fla.block import (
    load_initial_state_from_block_map,
    store_ssm_state_to_block_map,
)
from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule
from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating
from rtp_llm.models_py.utils.debug import cudagraph_debug_kernel
from rtp_llm.models_py.utils.typed_storage_view import LinearCacheConverter
from rtp_llm.ops import (
    AttentionConfigs,
    HybridAttentionType,
    LinearAttentionConfig,
    ParallelismConfig,
)
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W
from rtp_llm.utils.swizzle_utils import can_swizzle_kn
from rtp_llm.utils.util import to_torch_dtype


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


_Q3N_TRACE_ENABLED = _env_flag("RTPLLM_QWEN3_NEXT_TRACE_DEBUG", False)
_Q3N_TRACE_FILTERS = tuple(
    item.strip()
    for item in os.environ.get("RTPLLM_QWEN3_NEXT_TRACE_FILTER", "").split(",")
    if item.strip()
)
_Q3N_TRACE_LAYERS_RAW = os.environ.get("RTPLLM_QWEN3_NEXT_TRACE_LAYERS", "linear").strip()
_Q3N_TRACE_EVERY = max(1, _env_int("RTPLLM_QWEN3_NEXT_TRACE_EVERY", 1))
_Q3N_TRACE_MAX_RECORDS = _env_int("RTPLLM_QWEN3_NEXT_TRACE_MAX_RECORDS", 0)
_Q3N_TRACE_MAX_LANES = max(1, _env_int("RTPLLM_QWEN3_NEXT_TRACE_MAX_LANES", 4))
_Q3N_TRACE_MAX_ELEMS = max(1, _env_int("RTPLLM_QWEN3_NEXT_TRACE_MAX_ELEMS", 8192))
_Q3N_TRACE_SYNC = _env_flag("RTPLLM_QWEN3_NEXT_TRACE_SYNC_DEVICE", True)
_Q3N_TRACE_PREFILL = _env_flag("RTPLLM_QWEN3_NEXT_TRACE_PREFILL", False)
_Q3N_TRACE_TENSOR_MODE = os.environ.get(
    "RTPLLM_QWEN3_NEXT_TRACE_TENSOR_MODE", "summary"
).strip().lower()
_Q3N_TRACE_METADATA_ONLY = _Q3N_TRACE_TENSOR_MODE in ("metadata", "meta", "none")
_Q3N_TRACE_STATE = {"records": 0, "per_trace_step": {}}

_Q3N_GRAPH_PROBE_ENABLED = _env_flag("RTPLLM_QWEN3_NEXT_GRAPH_PROBE", False)
_Q3N_GRAPH_PROBE_LAYERS = tuple(
    dict.fromkeys(
        int(item.strip())
        for item in os.environ.get(
            "RTPLLM_QWEN3_NEXT_GRAPH_PROBE_LAYERS",
            "0,8,16,24,32,40,48,56,63",
        ).split(",")
        if item.strip().lstrip("-").isdigit()
    )
)


def _graph_probe_stats(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.reshape(tensor.shape[0], -1).to(torch.float32)
    finite = torch.isfinite(flat)
    safe = torch.where(finite, flat, torch.zeros_like(flat))
    has_finite = finite.any(dim=1)
    minimum = torch.where(
        has_finite,
        torch.where(finite, flat, torch.full_like(flat, float("inf"))).amin(dim=1),
        torch.zeros_like(has_finite, dtype=torch.float32),
    )
    maximum = torch.where(
        has_finite,
        torch.where(finite, flat, torch.full_like(flat, float("-inf"))).amax(dim=1),
        torch.zeros_like(has_finite, dtype=torch.float32),
    )
    return torch.stack(
        (
            safe.sum(dim=1),
            safe.abs().sum(dim=1),
            (safe * safe).sum(dim=1),
            minimum,
            maximum,
            (~finite).sum(dim=1).to(torch.float32),
        ),
        dim=1,
    )


class _CudaGraphLayerProbe:
    field_names = (
        "sum",
        "abs_sum",
        "square_sum",
        "min",
        "max",
        "nonfinite_count",
        "residual_sum",
        "residual_abs_sum",
        "residual_square_sum",
        "residual_min",
        "residual_max",
        "residual_nonfinite_count",
    )

    def __init__(
        self,
        enabled: bool,
        layers: tuple[int, ...],
        layer_num: int,
    ):
        self.enabled = enabled
        self.layers = tuple(
            layer
            for layer in dict.fromkeys(layers)
            if 0 <= layer < int(layer_num)
        )
        self._layer_slots = {layer: slot for slot, layer in enumerate(self.layers)}
        self._buffers: dict[int, torch.Tensor] = {}
        self._record_debug = {
            "attempts": 0,
            "recorded": 0,
            "skipped_not_cuda_graph": 0,
            "skipped_invalid_tensor": 0,
            "skipped_invalid_layout": 0,
            "last_layer_idx": -1,
            "last_graph_bs": -1,
            "last_token_rows": -1,
            "last_residual_rows": -1,
            "last_is_cuda_graph": -1,
        }

    def record(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        *,
        graph_bs: int,
        is_cuda_graph: bool,
    ) -> None:
        if not self.enabled or layer_idx not in self._layer_slots:
            return
        debug = self._record_debug
        debug["attempts"] += 1
        debug["last_layer_idx"] = int(layer_idx)
        debug["last_graph_bs"] = int(graph_bs)
        debug["last_token_rows"] = (
            int(hidden_states.shape[0]) if hidden_states.dim() > 0 else -1
        )
        debug["last_residual_rows"] = (
            int(residual.shape[0]) if residual.dim() > 0 else -1
        )
        debug["last_is_cuda_graph"] = int(bool(is_cuda_graph))
        if not is_cuda_graph:
            debug["skipped_not_cuda_graph"] += 1
            return
        if hidden_states.dim() == 0 or residual.dim() == 0:
            debug["skipped_invalid_tensor"] += 1
            return
        graph_bs = int(graph_bs)
        token_rows = int(hidden_states.shape[0])
        if (
            graph_bs <= 0
            or token_rows <= 0
            or residual.shape[0] != token_rows
            or token_rows % graph_bs != 0
        ):
            debug["skipped_invalid_layout"] += 1
            return
        buffer = self._buffers.get(graph_bs)
        if buffer is None:
            buffer = torch.zeros(
                (len(self.layers), graph_bs, len(self.field_names)),
                dtype=torch.float32,
                device=hidden_states.device,
            )
            self._buffers[graph_bs] = buffer
        stats = torch.cat(
            (
                _graph_probe_stats(hidden_states.reshape(graph_bs, -1)),
                _graph_probe_stats(residual.reshape(graph_bs, -1)),
            ),
            dim=1,
        )
        buffer[self._layer_slots[layer_idx]].copy_(stats)
        debug["recorded"] += 1

    def get_debug_status(self) -> dict[str, int]:
        return dict(self._record_debug)

    def get_buffer(self, graph_bs: int) -> Optional[torch.Tensor]:
        return self._buffers.get(int(graph_bs))

    def get_capture(
        self, graph_bs: int
    ) -> Optional[tuple[torch.Tensor, tuple[int, ...]]]:
        buffer = self.get_buffer(graph_bs)
        if buffer is None:
            return None
        return buffer, self.layers


def _trace_file() -> Optional[Path]:
    if not _Q3N_TRACE_ENABLED:
        return None
    explicit_file = os.environ.get("RTPLLM_QWEN3_NEXT_TRACE_FILE")
    if explicit_file:
        return Path(explicit_file)
    trace_dir = Path(
        os.environ.get("RTPLLM_QWEN3_NEXT_TRACE_DIR", "/tmp/rtpllm_qwen3_next_trace")
    )
    rank = (
        os.environ.get("WORLD_RANK")
        or os.environ.get("RANK")
        or os.environ.get("LOCAL_RANK")
        or "0"
    )
    return trace_dir / f"qwen3_next_trace_rank{rank}_pid{os.getpid()}.jsonl"


_Q3N_TRACE_FILE = _trace_file()
if _Q3N_TRACE_FILE is not None:
    _Q3N_TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)


def _trace_layer_set() -> Optional[set[int]]:
    if not _Q3N_TRACE_LAYERS_RAW or _Q3N_TRACE_LAYERS_RAW in ("*", "all"):
        return None
    if _Q3N_TRACE_LAYERS_RAW == "linear":
        return set()
    result: set[int] = set()
    for item in _Q3N_TRACE_LAYERS_RAW.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.add(int(item))
        except ValueError:
            continue
    return result


_Q3N_TRACE_LAYERS = _trace_layer_set()


def _trace_selected_lanes(trace_ids: list[str], batch_size: int) -> list[int]:
    lanes = []
    for lane in range(min(batch_size, _Q3N_TRACE_MAX_LANES)):
        trace_id = trace_ids[lane] if lane < len(trace_ids) else ""
        if _Q3N_TRACE_FILTERS and not any(
            needle in trace_id for needle in _Q3N_TRACE_FILTERS
        ):
            continue
        lanes.append(lane)
    return lanes


def _trace_layer_enabled(layer_idx: int, layer_type: HybridAttentionType) -> bool:
    if not _Q3N_TRACE_ENABLED:
        return False
    if _Q3N_TRACE_LAYERS is None:
        return True
    if _Q3N_TRACE_LAYERS:
        return layer_idx in _Q3N_TRACE_LAYERS
    return layer_type == HybridAttentionType.LINEAR


def _trace_tensor_lanes(
    attention_inputs: PyAttentionInputs, hidden_states: torch.Tensor, batch_size: int
) -> dict[int, int]:
    if not attention_inputs.is_prefill or attention_inputs.is_target_verify:
        return {lane: lane for lane in range(batch_size)}
    if (
        attention_inputs.input_lengths is None
        or not torch.is_tensor(attention_inputs.input_lengths)
        or attention_inputs.input_lengths.numel() <= 0
    ):
        return {lane: lane for lane in range(batch_size)}
    input_lengths = attention_inputs.input_lengths.detach()
    if input_lengths.is_cuda:
        if _Q3N_TRACE_SYNC:
            torch.cuda.synchronize(input_lengths.device)
        input_lengths = input_lengths.cpu()
    cumsum = torch.cumsum(input_lengths.to(torch.int64), dim=0) - 1
    max_row = int(hidden_states.shape[0]) - 1 if hidden_states.dim() > 0 else 0
    tensor_lanes: dict[int, int] = {}
    for lane in range(min(batch_size, cumsum.numel())):
        tensor_lanes[lane] = max(0, min(max_row, int(cumsum[lane].item())))
    return tensor_lanes


def _make_trace_ctx(
    inputs: PyModelInputs, attention_inputs: PyAttentionInputs, hidden_states: torch.Tensor
) -> Optional[dict[str, Any]]:
    if not _Q3N_TRACE_ENABLED or _Q3N_TRACE_FILE is None:
        return None
    if (
        attention_inputs.is_prefill
        and not attention_inputs.is_target_verify
        and not _Q3N_TRACE_PREFILL
    ):
        return None
    trace_ids = list(getattr(inputs, "trace_ids", []) or [])
    batch_size = len(trace_ids) if trace_ids else (hidden_states.shape[0] if hidden_states.dim() else 1)
    lanes = _trace_selected_lanes(trace_ids, int(batch_size))
    if not lanes:
        return None
    tensor_lanes = _trace_tensor_lanes(attention_inputs, hidden_states, int(batch_size))
    per_trace_step = _Q3N_TRACE_STATE["per_trace_step"]
    trace_steps = {}
    kept_lanes = []
    for lane in lanes:
        trace_id = trace_ids[lane] if lane < len(trace_ids) else ""
        step = int(per_trace_step.get(trace_id, 0)) + 1
        per_trace_step[trace_id] = step
        if step % _Q3N_TRACE_EVERY != 0:
            continue
        trace_steps[lane] = step
        kept_lanes.append(lane)
    if not kept_lanes:
        return None
    return {
        "trace_ids": trace_ids,
        "lanes": kept_lanes,
        "tensor_lanes": tensor_lanes,
        "trace_steps": trace_steps,
        "is_prefill": bool(attention_inputs.is_prefill),
        "is_target_verify": bool(attention_inputs.is_target_verify),
        "input_shape": list(hidden_states.shape),
        "created_at": time.time(),
    }


def _tensor_summary(tensor: Any, lane: Optional[int] = None) -> dict[str, Any]:
    if tensor is None or not torch.is_tensor(tensor):
        return {"defined": False}
    if not tensor.is_meta and tensor.numel() == 0:
        return {
            "defined": True,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "numel": 0,
        }
    selected = tensor
    if lane is not None and tensor.dim() > 0 and lane < tensor.shape[0]:
        selected = tensor.narrow(0, lane, 1)
    selected = selected.detach()
    if _Q3N_TRACE_METADATA_ONLY:
        return {
            "defined": True,
            "mode": "metadata",
            "shape": list(tensor.shape),
            "selected_shape": list(selected.shape),
            "stride": list(tensor.stride()),
            "selected_stride": list(selected.stride()),
            "storage_offset": int(selected.storage_offset()),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "numel": int(selected.numel()),
            "data_ptr": int(selected.data_ptr()) if selected.numel() else 0,
            "is_contiguous": bool(selected.is_contiguous()),
        }
    if selected.is_cuda and _Q3N_TRACE_SYNC:
        torch.cuda.synchronize(selected.device)
    flat = selected.reshape(-1)
    truncated = int(max(0, flat.numel() - _Q3N_TRACE_MAX_ELEMS))
    if flat.numel() > _Q3N_TRACE_MAX_ELEMS:
        flat = flat[:_Q3N_TRACE_MAX_ELEMS]
    if flat.is_floating_point():
        cpu = flat.to(torch.float32).cpu()
    else:
        cpu = flat.cpu()
    contiguous = cpu.contiguous()
    raw = contiguous.numpy().tobytes()
    summary = {
        "defined": True,
        "shape": list(tensor.shape),
        "selected_shape": list(selected.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(selected.numel()),
        "sample_numel": int(contiguous.numel()),
        "truncated": truncated,
        "sha256": hashlib.sha256(raw).hexdigest()[:16],
        "head": contiguous[: min(8, contiguous.numel())].tolist(),
    }
    if contiguous.numel() and contiguous.is_floating_point():
        finite = torch.isfinite(contiguous)
        summary["nan_count"] = int(torch.isnan(contiguous).sum().item())
        summary["inf_count"] = int(torch.isinf(contiguous).sum().item())
        if bool(finite.any().item()):
            finite_values = contiguous[finite]
            summary["min"] = float(finite_values.min().item())
            summary["max"] = float(finite_values.max().item())
            summary["mean"] = float(finite_values.mean().item())
    elif contiguous.numel():
        summary["min"] = int(contiguous.min().item())
        summary["max"] = int(contiguous.max().item())
    return summary


def _lane_int(tensor: torch.Tensor, lane: int) -> Optional[int]:
    if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
        return None
    if tensor.dim() == 0:
        selected = tensor.detach()
    elif lane < tensor.shape[0]:
        selected = tensor.narrow(0, lane, 1).reshape(-1)[0].detach()
    else:
        return None
    if _Q3N_TRACE_METADATA_ONLY and selected.is_cuda:
        return None
    if selected.is_cuda and _Q3N_TRACE_SYNC:
        torch.cuda.synchronize(selected.device)
    return int(selected.cpu().item())


def _block_id_for_lane(
    attention_inputs: PyAttentionInputs, lane: int, seq_size_per_block: int
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    block_map = attention_inputs.kv_cache_kernel_block_id_device
    seq_len_plus_1 = _lane_int(attention_inputs.sequence_lengths_plus_1_d, lane)
    if (
        block_map is None
        or not torch.is_tensor(block_map)
        or block_map.numel() == 0
        or seq_len_plus_1 is None
        or seq_size_per_block <= 0
        or block_map.dim() < 2
        or lane >= block_map.shape[0]
    ):
        return seq_len_plus_1, None, None
    logical_block = max(0, (seq_len_plus_1 - 1) // seq_size_per_block)
    if logical_block >= block_map.shape[1]:
        return seq_len_plus_1, logical_block, None
    row = block_map.narrow(0, lane, 1).reshape(-1)
    if row.is_cuda and _Q3N_TRACE_SYNC:
        torch.cuda.synchronize(row.device)
    physical_block = int(row.narrow(0, logical_block, 1).cpu().item())
    return seq_len_plus_1, logical_block, physical_block


def _prefix_block_id_for_lane(
    attention_inputs: PyAttentionInputs, lane: int, seq_size_per_block: int
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    block_map = attention_inputs.kv_cache_kernel_block_id_device
    prefix_len = _lane_int(attention_inputs.prefix_lengths, lane)
    if (
        block_map is None
        or not torch.is_tensor(block_map)
        or block_map.numel() == 0
        or prefix_len is None
        or prefix_len <= 0
        or seq_size_per_block <= 0
        or block_map.dim() < 2
        or lane >= block_map.shape[0]
    ):
        return prefix_len, None, None
    logical_block = max(0, (prefix_len - 1) // seq_size_per_block)
    if logical_block >= block_map.shape[1]:
        return prefix_len, logical_block, None
    row = block_map.narrow(0, lane, 1).reshape(-1)
    if row.is_cuda and _Q3N_TRACE_SYNC:
        torch.cuda.synchronize(row.device)
    physical_block = int(row.narrow(0, logical_block, 1).cpu().item())
    return prefix_len, logical_block, physical_block


def _trace_event(
    ctx: Optional[dict[str, Any]],
    event: str,
    *,
    layer_idx: Optional[int] = None,
    layer_type: Optional[HybridAttentionType] = None,
    group_id: Optional[int] = None,
    attention_inputs: Optional[PyAttentionInputs] = None,
    seq_size_per_block: Optional[int] = None,
    tensors: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    if ctx is None or _Q3N_TRACE_FILE is None:
        return
    if _Q3N_TRACE_MAX_RECORDS > 0 and _Q3N_TRACE_STATE["records"] >= _Q3N_TRACE_MAX_RECORDS:
        return
    rows = []
    for lane in ctx["lanes"]:
        trace_ids = ctx["trace_ids"]
        trace_id = trace_ids[lane] if lane < len(trace_ids) else ""
        tensor_lane = int(ctx.get("tensor_lanes", {}).get(lane, lane))
        row = {
            "ts": time.time(),
            "pid": os.getpid(),
            "rank": os.environ.get("WORLD_RANK") or os.environ.get("RANK"),
            "event": event,
            "trace_id": trace_id,
            "trace_step": ctx["trace_steps"].get(lane),
            "lane": lane,
            "tensor_lane": tensor_lane,
            "layer_idx": layer_idx,
            "layer_type": str(layer_type) if layer_type is not None else None,
            "group_id": group_id,
            "is_prefill": ctx["is_prefill"],
            "is_target_verify": ctx["is_target_verify"],
            "input_shape": ctx["input_shape"],
        }
        if attention_inputs is not None:
            row["attention"] = {
                "input_length": _lane_int(attention_inputs.input_lengths, lane),
                "prefix_length": _lane_int(attention_inputs.prefix_lengths, lane),
                "sequence_length": _lane_int(attention_inputs.sequence_lengths, lane),
                "sequence_length_plus_1": _lane_int(
                    attention_inputs.sequence_lengths_plus_1_d, lane
                ),
                "decode_cu_seqlens": _tensor_summary(
                    attention_inputs.decode_cu_seqlens_d, lane
                ),
                "block_map": _tensor_summary(
                    attention_inputs.kv_cache_kernel_block_id_device, lane
                ),
            }
            if seq_size_per_block is not None:
                seq_len, logical_block, physical_block = _block_id_for_lane(
                    attention_inputs, lane, seq_size_per_block
                )
                row["attention"]["seq_size_per_block"] = seq_size_per_block
                row["attention"]["current_seq_len_plus_1"] = seq_len
                row["attention"]["current_logical_block"] = logical_block
                row["attention"]["current_physical_block"] = physical_block
        if tensors:
            row["tensors"] = {
                name: _tensor_summary(value, tensor_lane)
                for name, value in tensors.items()
            }
        if extra:
            row["extra"] = extra
        rows.append(row)
        _Q3N_TRACE_STATE["records"] += 1
        if (
            _Q3N_TRACE_MAX_RECORDS > 0
            and _Q3N_TRACE_STATE["records"] >= _Q3N_TRACE_MAX_RECORDS
        ):
            break
    if not rows:
        return
    with _Q3N_TRACE_FILE.open("a", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _trace_cache_state(
    ctx: Optional[dict[str, Any]],
    event: str,
    *,
    layer_idx: Optional[int],
    group_id: Optional[int],
    attention_inputs: PyAttentionInputs,
    seq_size_per_block: int,
    conv_states: Optional[torch.Tensor],
    ssm_states: Optional[torch.Tensor],
    tensors: Optional[dict[str, Any]] = None,
) -> None:
    if ctx is None:
        return
    for lane in ctx["lanes"]:
        _, _, physical_block = _block_id_for_lane(
            attention_inputs, lane, seq_size_per_block
        )
        cache_tensors = dict(tensors or {})
        if physical_block is not None:
            if (
                conv_states is not None
                and torch.is_tensor(conv_states)
                and 0 <= physical_block < conv_states.shape[0]
            ):
                cache_tensors["conv_state_current_block"] = conv_states.narrow(
                    0, physical_block, 1
                )
            if (
                ssm_states is not None
                and torch.is_tensor(ssm_states)
                and 0 <= physical_block < ssm_states.shape[0]
            ):
                cache_tensors["ssm_state_current_block"] = ssm_states.narrow(
                    0, physical_block, 1
                )
        lane_ctx = dict(ctx)
        lane_ctx["lanes"] = [lane]
        _trace_event(
            lane_ctx,
            event,
            layer_idx=layer_idx,
            layer_type=HybridAttentionType.LINEAR,
            group_id=group_id,
            attention_inputs=attention_inputs,
            seq_size_per_block=seq_size_per_block,
            tensors=cache_tensors,
        )


def _trace_prefill_cache_state(
    ctx: Optional[dict[str, Any]],
    event: str,
    *,
    layer_idx: Optional[int],
    group_id: Optional[int],
    attention_inputs: PyAttentionInputs,
    seq_size_per_block: int,
    conv_states: Optional[torch.Tensor] = None,
    ssm_states: Optional[torch.Tensor] = None,
    tensors: Optional[dict[str, Any]] = None,
    batch_tensors: Optional[dict[str, Any]] = None,
) -> None:
    if ctx is None:
        return
    for lane in ctx["lanes"]:
        prefix_len, logical_block, physical_block = _prefix_block_id_for_lane(
            attention_inputs, lane, seq_size_per_block
        )
        cache_tensors = dict(tensors or {})
        for name, value in (batch_tensors or {}).items():
            if torch.is_tensor(value) and value.dim() > 0 and lane < value.shape[0]:
                cache_tensors[name] = value.narrow(0, lane, 1)
            else:
                cache_tensors[name] = value
        if physical_block is not None:
            if (
                conv_states is not None
                and torch.is_tensor(conv_states)
                and 0 <= physical_block < conv_states.shape[0]
            ):
                cache_tensors["conv_state_prefix_block"] = conv_states.narrow(
                    0, physical_block, 1
                )
            if (
                ssm_states is not None
                and torch.is_tensor(ssm_states)
                and 0 <= physical_block < ssm_states.shape[0]
            ):
                cache_tensors["ssm_state_prefix_block"] = ssm_states.narrow(
                    0, physical_block, 1
                )
        lane_ctx = dict(ctx)
        lane_ctx["lanes"] = [lane]
        _trace_event(
            lane_ctx,
            event,
            layer_idx=layer_idx,
            layer_type=HybridAttentionType.LINEAR,
            group_id=group_id,
            attention_inputs=attention_inputs,
            seq_size_per_block=seq_size_per_block,
            tensors=cache_tensors,
            extra={
                "prefix_len_for_state": prefix_len,
                "prefix_logical_block": logical_block,
                "prefix_physical_block": physical_block,
            },
        )


class Qwen3NextMetadata(object):
    def __init__(
        self,
        prefill_conv1d_meta: Optional[CausalConv1dMetadata] = None,
        is_target_verify: bool = False,
        full_prefill_conv1d_meta: Optional[CausalConv1dMetadata] = None,
        full_prefill_cu_seqlens: Optional[torch.Tensor] = None,
        cp_restore_indices: Optional[torch.Tensor] = None,
        cp_local_extract_indices: Optional[torch.Tensor] = None,
        cp_local_valid_mask: Optional[torch.Tensor] = None,
        cp_write_cache_store_impl: Optional[WriteCacheStoreOp] = None,
    ):
        self.prefill_conv1d_meta = prefill_conv1d_meta
        self.is_target_verify = is_target_verify
        self.full_prefill_conv1d_meta = full_prefill_conv1d_meta
        self.full_prefill_cu_seqlens = full_prefill_cu_seqlens
        self.cp_restore_indices = cp_restore_indices
        self.cp_local_extract_indices = cp_local_extract_indices
        self.cp_local_valid_mask = cp_local_valid_mask
        self.cp_write_cache_store_impl = cp_write_cache_store_impl
        self.trace_ctx: Optional[dict[str, Any]] = None

    def get_prefill_conv1d_meta(self) -> Optional[CausalConv1dMetadata]:
        return self.prefill_conv1d_meta

    @property
    def is_cp_linear_attn(self) -> bool:
        return self.cp_restore_indices is not None


class Qwen3NextGatedDeltaNetBase(torch.nn.Module):
    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__()
        self.linear_attn_config = linear_attn_config
        self.parallelism_config = parallelism_config
        self.weights = weights
        # params
        self.head_k_dim: int = linear_attn_config.linear_key_head_dim
        self.head_v_dim: int = linear_attn_config.linear_value_head_dim
        assert (
            self.head_k_dim == self.head_v_dim
        ), "head_k_dim and head_v_dim must be the same now"
        attn_tp_size = parallelism_config.get_attn_tp_size()
        self.local_num_k_heads: int = (
            linear_attn_config.linear_num_key_heads // attn_tp_size
        )
        self.local_num_v_heads: int = (
            linear_attn_config.linear_num_value_heads // attn_tp_size
        )
        self.num_key_value_heads: int = self.local_num_v_heads // self.local_num_k_heads
        self.linear_conv_kernel_dim: int = (
            self.linear_attn_config.linear_conv_kernel_dim
        )
        self.ssm_state_size: int = (
            self.local_num_v_heads * self.head_k_dim * self.head_v_dim
        )
        self.qkv_size: int = (
            self.head_k_dim * self.local_num_k_heads * 2
            + self.head_v_dim * self.local_num_v_heads
        )
        self.conv_state_size: int = (self.linear_conv_kernel_dim - 1) * self.qkv_size
        self.ssm_state_dtype: torch.dtype = to_torch_dtype(
            linear_attn_config.ssm_state_dtype
        )
        self.conv_state_dtype: torch.dtype = to_torch_dtype(
            linear_attn_config.conv_state_dtype
        )
        self.linear_cache_converter = LinearCacheConverter(
            local_num_v_heads=self.local_num_v_heads,
            head_v_dim=self.head_v_dim,
            head_k_dim=self.head_k_dim,
            ssm_state_dtype=self.ssm_state_dtype,
            linear_conv_kernel_dim=self.linear_conv_kernel_dim,
            qkv_size=self.qkv_size,
            conv_state_dtype=self.conv_state_dtype,
        )
        # weights
        self.conv_weights = weights[W.linear_attn_conv1d_w].squeeze(1)
        self.dt_bias = weights[W.linear_attn_dt_b]
        self.alog = weights[W.linear_attn_alog]

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: Qwen3NextMetadata,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _get_conv_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        conv_states = self.linear_cache_converter.get_conv_state_tensor(kv_cache_tensor)
        return conv_states

    def _get_ssm_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        ssm_states = self.linear_cache_converter.get_ssm_state_tensor(kv_cache_tensor)
        return ssm_states


class Qwen3NextGatedDeltaNetPrefill(Qwen3NextGatedDeltaNetBase):
    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(linear_attn_config, parallelism_config, weights)

    def _conv1d(
        self,
        mixed_qkv: torch.Tensor,
        kv_cache_tensor: Optional[torch.Tensor],
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        metadata: Optional[CausalConv1dMetadata] = None,
    ) -> torch.Tensor:
        # cu_seqlen_without_padding = attn_inputs.cu_seqlens[
        #     : attn_inputs.input_lengths.size(0) + 1
        # ]
        cu_seqlen_without_padding = attn_inputs.cu_seqlens
        conv_states = (
            self._get_conv_states(kv_cache_tensor).transpose(1, 2)
            if kv_cache_tensor is not None
            else None
        )
        out = causal_conv1d_fn(
            x=mixed_qkv.transpose(0, 1),
            weight=self.conv_weights,
            bias=None,
            conv_states=conv_states,
            query_start_loc=cu_seqlen_without_padding,
            block_map=attn_inputs.kv_cache_kernel_block_id_device,
            seq_size_per_block=seq_size_per_block,
            prefix_lengths=attn_inputs.prefix_lengths_d,
            metadata=metadata,
        ).transpose(0, 1)
        return out

    def _fla(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        kv_cache_tensor: Optional[torch.Tensor],
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        attn_meta: Optional[Qwen3NextMetadata] = None,
    ) -> torch.Tensor:
        g, beta = fused_gdn_gating(self.alog, a, b, self.dt_bias)
        ssm_states = (
            self._get_ssm_states(kv_cache_tensor)
            if kv_cache_tensor is not None
            else None
        )
        trace_ctx = attn_meta.trace_ctx if attn_meta is not None else None
        layer_idx = getattr(attn_meta, "trace_layer_idx", None)
        group_id = getattr(attn_meta, "trace_group_id", None)
        context_batch_size = attn_inputs.input_lengths.shape[0]
        # cu_seqlens_without_padding = attn_inputs.cu_seqlens[: context_batch_size + 1]
        cu_seqlens_without_padding = attn_inputs.cu_seqlens
        initial_states: Optional[torch.Tensor] = None
        if ssm_states is not None:
            initial_states = torch.empty(
                context_batch_size,
                self.local_num_v_heads,
                self.head_v_dim,
                self.head_k_dim,
                device=mixed_qkv.device,
                dtype=self.ssm_state_dtype,
            )

            load_initial_state_from_block_map(
                attn_inputs.prefix_lengths_d,
                attn_inputs.kv_cache_kernel_block_id_device,
                ssm_states,
                initial_states,
                seq_size_per_block,
            )
        g_for_trace = g.squeeze(0) if g.dim() == 3 and g.shape[0] == 1 else g
        beta_for_trace = (
            beta.squeeze(0) if beta.dim() == 3 and beta.shape[0] == 1 else beta
        )
        _trace_prefill_cache_state(
            trace_ctx,
            "linear_prefill_after_load_state",
            layer_idx=layer_idx,
            group_id=group_id,
            attention_inputs=attn_inputs,
            seq_size_per_block=seq_size_per_block,
            ssm_states=ssm_states,
            tensors={"mixed_qkv": mixed_qkv, "g": g_for_trace, "beta": beta_for_trace},
            batch_tensors={"initial_state": initial_states},
        )
        # M >= 2048: scatter_qkv (Triton, SGLang port) avoids the .view() ->
        # .contiguous() copies that torch.split + view triggers. Below 2048,
        # kernel launch overhead beats the savings (microbench measured).
        if mixed_qkv.shape[0] >= 2048 and self.head_k_dim == self.head_v_dim:
            query, key, value = scatter_qkv(
                mixed_qkv,
                self.local_num_k_heads,
                self.local_num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
            )
        else:
            query, key, value = torch.split(
                mixed_qkv,
                [
                    self.local_num_k_heads * self.head_k_dim,
                    self.local_num_k_heads * self.head_k_dim,
                    self.local_num_v_heads * self.head_v_dim,
                ],
                dim=-1,
            )
            query = query.view(
                1, query.shape[0], self.local_num_k_heads, self.head_k_dim
            )
            key = key.view(1, key.shape[0], self.local_num_k_heads, self.head_k_dim)
            value = value.view(
                1, value.shape[0], self.local_num_v_heads, self.head_v_dim
            )
        _trace_prefill_cache_state(
            trace_ctx,
            "linear_prefill_before_chunk",
            layer_idx=layer_idx,
            group_id=group_id,
            attention_inputs=attn_inputs,
            seq_size_per_block=seq_size_per_block,
            ssm_states=ssm_states,
            tensors={
                "query": query.squeeze(0),
                "key": key.squeeze(0),
                "value": value.squeeze(0),
            },
        )
        attn_out, h, final_state = chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_states,
            output_final_state=True,
            cu_seqlens=cu_seqlens_without_padding,
            use_qk_l2norm_in_kernel=True,
        )
        _trace_prefill_cache_state(
            trace_ctx,
            "linear_prefill_after_chunk",
            layer_idx=layer_idx,
            group_id=group_id,
            attention_inputs=attn_inputs,
            seq_size_per_block=seq_size_per_block,
            ssm_states=ssm_states,
            tensors={"attn_out": attn_out.squeeze(0)},
            batch_tensors={"final_state": final_state},
        )
        if ssm_states is not None:
            store_ssm_state_to_block_map(
                h,
                final_state,
                attn_inputs.prefix_lengths_d,
                cu_seqlens_without_padding,
                attn_inputs.kv_cache_kernel_block_id_device,
                ssm_states,
                seq_size_per_block,
                chunk_size=64,
            )
        _trace_prefill_cache_state(
            trace_ctx,
            "linear_prefill_after_store_state",
            layer_idx=layer_idx,
            group_id=group_id,
            attention_inputs=attn_inputs,
            seq_size_per_block=seq_size_per_block,
            ssm_states=ssm_states,
            batch_tensors={"final_state": final_state},
        )
        return attn_out.squeeze_(0)

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: Qwen3NextMetadata,
    ) -> torch.Tensor:
        kv_cache_tensor: Optional[torch.Tensor] = None
        seq_size_per_block = 1
        if kv_cache is not None:
            kv_cache_tensor = kv_cache.kv_cache_base.reshape(
                kv_cache.kv_cache_base.shape[0], -1
            )
            seq_size_per_block = kv_cache.seq_size_per_block
        trace_ctx = attn_meta.trace_ctx
        layer_idx = getattr(attn_meta, "trace_layer_idx", None)
        group_id = getattr(attn_meta, "trace_group_id", None)
        conv_states = (
            self._get_conv_states(kv_cache_tensor)
            if kv_cache_tensor is not None and trace_ctx is not None
            else None
        )
        _trace_prefill_cache_state(
            trace_ctx,
            "linear_prefill_before_conv",
            layer_idx=layer_idx,
            group_id=group_id,
            attention_inputs=attn_inputs,
            seq_size_per_block=seq_size_per_block,
            conv_states=conv_states,
            tensors={"mixed_qkv": mixed_qkv, "b": b, "a": a},
        )
        mixed_qkv = self._conv1d(
            mixed_qkv,
            kv_cache_tensor,
            seq_size_per_block,
            attn_inputs,
            metadata=attn_meta.get_prefill_conv1d_meta(),
        )
        _trace_prefill_cache_state(
            trace_ctx,
            "linear_prefill_after_conv",
            layer_idx=layer_idx,
            group_id=group_id,
            attention_inputs=attn_inputs,
            seq_size_per_block=seq_size_per_block,
            conv_states=conv_states,
            tensors={"mixed_qkv": mixed_qkv},
        )
        attn_out = self._fla(
            mixed_qkv, b, a, kv_cache_tensor, seq_size_per_block, attn_inputs, attn_meta
        )
        if kv_cache is not None:
            # write kvcache to cache store
            compute_ops.write_cache_store(
                attn_inputs.input_lengths,
                attn_inputs.prefix_lengths,
                attn_inputs.kv_cache_block_id_host,
                attn_inputs.cache_store_inputs,
                kv_cache,
            )
        return attn_out


class Qwen3NextGatedDeltaNetDecode(Qwen3NextGatedDeltaNetBase):
    def _conv1d(
        self,
        mixed_qkv: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        is_target_verify: bool,
    ) -> torch.Tensor:
        conv_states = self._get_conv_states(kv_cache_tensor)
        # (batch, dim) -> # (batch, dim, 1)
        batch, seq = self._get_bs_from_attenion_input(
            mixed_qkv, attn_inputs, is_target_verify
        )
        origin_shape = mixed_qkv.shape
        mixed_qkv = mixed_qkv.reshape(batch, seq, -1).transpose(1, 2)
        out = causal_conv1d_update(
            mixed_qkv,
            conv_states.transpose(1, 2),
            self.conv_weights,
            bias=None,
            activation="silu",
            cache_seqlens=None,
            block_map=attn_inputs.kv_cache_kernel_block_id_device,
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=attn_inputs.sequence_lengths_plus_1_d,
        )
        out = out.transpose(1, 2).reshape(origin_shape)
        return out

    def _fla(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        is_target_verify: bool,
    ) -> torch.Tensor:
        batch, seq = self._get_bs_from_attenion_input(
            mixed_qkv, attn_inputs, is_target_verify
        )
        # asserr head_k_dim == head_v_dim
        mixed_qkv = mixed_qkv.reshape(
            batch,
            seq,
            self.local_num_k_heads * 2 + self.local_num_v_heads,
            self.head_k_dim,
        )
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.local_num_k_heads,
                self.local_num_k_heads,
                self.local_num_v_heads,
            ],
            dim=2,
        )

        g, beta = fused_gdn_gating(self.alog, a, b, self.dt_bias)

        # contiguous will be applyed when call fused_recurrent_gated_delta_rule
        g = g.view(batch, seq, self.local_num_v_heads)
        beta = beta.view(batch, seq, self.local_num_v_heads)
        ssm_states = self._get_ssm_states(kv_cache_tensor)
        core_attn_out, _ = fused_recurrent_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            scale=None,
            initial_state=ssm_states,
            inplace_final_state=True,
            block_map=attn_inputs.kv_cache_kernel_block_id_device,
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=attn_inputs.sequence_lengths_plus_1_d,
            use_qk_l2norm_in_kernel=True,
        )
        res = core_attn_out.reshape(
            [-1, core_attn_out.shape[2], core_attn_out.shape[3]]
        )
        return res

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: Qwen3NextMetadata,
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required for decode"
        assert (
            kv_cache.kv_cache_base is not None
        ), "kv_cache_tensor is required for decode"
        kv_cache_tensor: torch.Tensor = kv_cache.kv_cache_base.reshape(
            kv_cache.kv_cache_base.shape[0], -1
        )
        is_target_verify = attn_meta.is_target_verify
        trace_ctx = attn_meta.trace_ctx
        layer_idx = getattr(attn_meta, "trace_layer_idx", None)
        group_id = getattr(attn_meta, "trace_group_id", None)
        conv_states = self._get_conv_states(kv_cache_tensor)
        ssm_states = self._get_ssm_states(kv_cache_tensor)
        _trace_cache_state(
            trace_ctx,
            "linear_decode_before_conv",
            layer_idx=layer_idx,
            group_id=group_id,
            attention_inputs=attn_inputs,
            seq_size_per_block=kv_cache.seq_size_per_block,
            conv_states=conv_states,
            ssm_states=ssm_states,
            tensors={"mixed_qkv": mixed_qkv, "b": b, "a": a},
        )
        mixed_qkv = self._conv1d(
            mixed_qkv,
            kv_cache_tensor,
            kv_cache.seq_size_per_block,
            attn_inputs,
            is_target_verify,
        )
        _trace_cache_state(
            trace_ctx,
            "linear_decode_after_conv",
            layer_idx=layer_idx,
            group_id=group_id,
            attention_inputs=attn_inputs,
            seq_size_per_block=kv_cache.seq_size_per_block,
            conv_states=conv_states,
            ssm_states=ssm_states,
            tensors={"mixed_qkv": mixed_qkv},
        )
        attn_out = self._fla(
            mixed_qkv,
            b,
            a,
            kv_cache_tensor,
            kv_cache.seq_size_per_block,
            attn_inputs,
            is_target_verify,
        )
        _trace_cache_state(
            trace_ctx,
            "linear_decode_after_fla",
            layer_idx=layer_idx,
            group_id=group_id,
            attention_inputs=attn_inputs,
            seq_size_per_block=kv_cache.seq_size_per_block,
            conv_states=conv_states,
            ssm_states=ssm_states,
            tensors={"attn_out": attn_out},
        )

        return attn_out

    def _get_bs_from_attenion_input(
        self,
        mixed_qkv: torch.Tensor,
        attention_inputs: PyAttentionInputs,
        is_target_verify: bool,
    ) -> tuple[int, int]:
        token, _ = mixed_qkv.shape
        if not is_target_verify:
            return token, 1
        assert (
            attention_inputs.prefix_lengths.size(0) > 0
        ), f"prefill_lengths size: {attention_inputs.prefix_lengths.size(0)} <=0 when target verify"
        assert (
            token % attention_inputs.prefix_lengths.size(0) == 0
        ), f"token: {token} is not divisible by prefill_lengths size: {attention_inputs.prefix_lengths.size(0)} when target verify"
        b, s = attention_inputs.prefix_lengths.size(
            0
        ), token // attention_inputs.prefix_lengths.size(0)
        return b, s


class Qwen3NextAttention(CausalAttention):
    def __init__(
        self,
        attn_config: AttentionConfigs,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layernorm_eps: float,
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__(
            attn_config,
            parallelism_config,
            weights,
            layernorm_eps,
            quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        # maybe fuse gate in qkv_proj later
        self.gate = LinearFactory.create_linear_from_weights(
            weights,
            W.attn_gate_w,
            W.attn_gate_s,
            None,
            quant_config,
            hw_kernel_config=hw_kernel_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        attention_inputs: Optional[PyAttentionInputs],
        attn_meta: Qwen3NextMetadata = Qwen3NextMetadata(),
    ) -> torch.Tensor:
        gate = self.gate(hidden_states)
        attn_out = super().forward(hidden_states, fmha_impl, kv_cache, gate)
        return attn_out


class Qwen3NextGatedDeltaNet(nn.Module):
    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layernorm_eps: float,
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.linear_attn_config = linear_attn_config
        self.parallelism_config = parallelism_config
        self.weights = weights
        self.quant_config = quant_config
        self.head_k_dim = linear_attn_config.linear_key_head_dim
        self.head_v_dim = linear_attn_config.linear_value_head_dim
        attn_tp_size = parallelism_config.get_attn_tp_size()
        self.local_num_k_heads = linear_attn_config.linear_num_key_heads // attn_tp_size
        self.local_num_v_heads = (
            linear_attn_config.linear_num_value_heads // attn_tp_size
        )
        self.num_key_value_heads = self.local_num_v_heads // self.local_num_k_heads

        # qkvz+ba fusion (BF16 only): combine two in-projection GEMMs into one.
        # Saves a small kernel launch on each forward; on decode (M=1) HBM-access
        # merging shaves a few us per layer (trace measurement: -0.094 ms/step
        # on Qwen3.5-9B TP=2 in the original session).
        # FP8/quantized: qkvz has scales but ba doesn't, dtypes mismatch -> fall
        # back to the original 2-GEMM path.
        self._qkvz_ba_fused = weights.get(W.linear_attn_qkvz_s) is None
        if self._qkvz_ba_fused:
            qkvz_w = weights[W.linear_attn_qkvz_w]
            ba_w = weights[W.linear_attn_ba_w]
            self._qkvz_size = qkvz_w.shape[1]
            self._ba_size = ba_w.shape[1]
            # Allocate the fused buffer ONCE; copy qkvz/ba into it; then
            # replace the original dict entries with VIEWS into the fused
            # buffer. This achieves two goals at once:
            #
            #  (a) Memory: avoids the ~48MB-per-layer redundant copy from
            #      torch.cat. The originals get released when this method
            #      returns (the local vars go out of scope and the dict
            #      entries no longer reference them).
            #
            #  (b) Online weight update: WeightManager.update_layer_weight
            #      runs `ori_tensor.copy_(data)` against the dict entries.
            #      With the entries now being views into the fused buffer,
            #      an update writes directly into the right slice of the
            #      fused buffer, so in_proj_fused.weight (which references
            #      the same buffer) sees the update on the next forward.
            #      copy_ accepts non-contig destinations, so the view's
            #      stride mismatch is fine.
            K = qkvz_w.shape[0]
            fused_w = torch.empty(
                K,
                self._qkvz_size + self._ba_size,
                dtype=qkvz_w.dtype,
                device=qkvz_w.device,
            )
            fused_w[:, : self._qkvz_size].copy_(qkvz_w)
            fused_w[:, self._qkvz_size :].copy_(ba_w)
            weights[W.linear_attn_qkvz_w] = fused_w[:, : self._qkvz_size]
            weights[W.linear_attn_ba_w] = fused_w[:, self._qkvz_size :]
            del qkvz_w, ba_w
            self.in_proj_fused = LinearFactory.create_linear(
                fused_w, None, None, quant_config, hw_kernel_config=hw_kernel_config
            )
            self.in_proj_qkvz = None
            self.in_proj_ba = None
        else:
            self.in_proj_qkvz = LinearFactory.create_linear_from_weights(
                weights,
                W.linear_attn_qkvz_w,
                W.linear_attn_qkvz_s,
                None,
                quant_config,
                hw_kernel_config=hw_kernel_config,
            )
            # BA out-dim may be non-16-aligned after TP split (e.g. TP=4 -> 24).
            # Match device_impl: keep swizzle (WithSwizzle dispatch) only when
            # aligned, otherwise pass None so dispatch picks NoSwizzle, staying
            # consistent with the swizzle skipped on the data side.
            ba_weight = weights[W.linear_attn_ba_w]
            ba_hw_kernel_config = (
                hw_kernel_config if can_swizzle_kn(ba_weight) else None
            )
            self.in_proj_ba = LinearFactory.create_linear_from_weights(
                weights,
                W.linear_attn_ba_w,
                None,
                None,
                quant_config,
                hw_kernel_config=ba_hw_kernel_config,
            )
            self.in_proj_fused = None

        self.prefill_gdn = Qwen3NextGatedDeltaNetPrefill(
            linear_attn_config, parallelism_config, weights
        )
        self.decode_gdn = Qwen3NextGatedDeltaNetDecode(
            linear_attn_config, parallelism_config, weights
        )
        self.norm = RmsNormGated(
            weights[W.linear_attn_norm_w],
            eps=layernorm_eps,
            group_size=linear_attn_config.linear_value_head_dim,
        )
        self.out_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.linear_attn_out_w,
            W.linear_attn_out_s,
            None,
            quant_config,
            hw_kernel_config=hw_kernel_config,
        )

    def _input_project(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the input projection and return (projected_qkvz, projected_ba).

        Hides the fusion vs 2-GEMM dispatch from callers (forward + tests).
        Both branches produce tensors with identical shape/semantics; the
        fused branch slices a single GEMM output, the fallback runs two.
        """
        if self._qkvz_ba_fused:
            fused = self.in_proj_fused(hidden_states)
            return fused[..., : self._qkvz_size], fused[..., self._qkvz_size :]
        return self.in_proj_qkvz(hidden_states), self.in_proj_ba(hidden_states)

    # mixed_qkvz, mixed_ba -> q, k, v, z, b, a
    def fix_query_key_value_ordering(
        self, mixed_qkvz: torch.Tensor, mixed_ba: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        split_arg_list_qkvz = [
            self.head_k_dim * self.local_num_k_heads
            + self.head_k_dim * self.local_num_k_heads
            + self.head_v_dim * self.local_num_v_heads,
            self.head_v_dim * self.local_num_v_heads,
        ]

        mixed_qkv, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=1)
        b, a = torch.split(
            mixed_ba, [self.local_num_v_heads, self.local_num_v_heads], dim=1
        )
        # reshape to [token, v_head_num, v_head_dim]
        # b,a should be contiguous for fused_gdn_gating
        return mixed_qkv, z, b, a

    # TODO: extract shared conv1d/FLA/ssm-state logic with Qwen3NextGatedDeltaNetPrefill
    # to eliminate duplication
    def _forward_cp_prefill(
        self,
        mixed_qkv: torch.Tensor,
        z: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        attention_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: Qwen3NextMetadata,
    ) -> torch.Tensor:
        """CP prefill path: all-gather projected states, compute on full sequence,
        extract local zigzag tokens."""
        cp_info = attention_inputs.context_parallel_info

        packed = torch.cat([mixed_qkv, b, a], dim=-1)
        full_packed = all_gather(packed, group=Group.TP)

        padding_mask = cp_info.prefill_qkv_padding_mask
        restore_indices = cp_info.prefill_qkv_restore_indice
        unpad_restore = restore_indices[padding_mask == 1]
        full_packed = full_packed[unpad_restore]

        qkv_dim = mixed_qkv.shape[-1]
        b_dim = b.shape[-1]
        full_mixed_qkv = full_packed[:, :qkv_dim].contiguous()
        full_b = full_packed[:, qkv_dim : qkv_dim + b_dim].contiguous()
        full_a = full_packed[:, qkv_dim + b_dim :].contiguous()

        gdn = self.prefill_gdn
        full_cu = attn_meta.full_prefill_cu_seqlens
        full_conv_meta = attn_meta.full_prefill_conv1d_meta

        kv_cache_tensor: Optional[torch.Tensor] = None
        seq_size_per_block = 1
        if kv_cache is not None:
            kv_cache_tensor = kv_cache.kv_cache_base.reshape(
                kv_cache.kv_cache_base.shape[0], -1
            )
            seq_size_per_block = kv_cache.seq_size_per_block

        conv_states = (
            gdn._get_conv_states(kv_cache_tensor).transpose(1, 2)
            if kv_cache_tensor is not None
            else None
        )
        full_mixed_qkv = causal_conv1d_fn(
            x=full_mixed_qkv.transpose(0, 1),
            weight=gdn.conv_weights,
            bias=None,
            conv_states=conv_states,
            query_start_loc=full_cu,
            block_map=attention_inputs.kv_cache_kernel_block_id_device,
            seq_size_per_block=seq_size_per_block,
            prefix_lengths=attention_inputs.prefix_lengths_d,
            metadata=full_conv_meta,
        ).transpose(0, 1)

        g, beta = fused_gdn_gating(gdn.alog, full_a, full_b, gdn.dt_bias)
        ssm_states = (
            gdn._get_ssm_states(kv_cache_tensor)
            if kv_cache_tensor is not None
            else None
        )
        context_batch_size = attention_inputs.input_lengths.shape[0]
        initial_states: Optional[torch.Tensor] = None
        if ssm_states is not None:
            initial_states = torch.empty(
                context_batch_size,
                gdn.local_num_v_heads,
                gdn.head_v_dim,
                gdn.head_k_dim,
                device=full_mixed_qkv.device,
                dtype=gdn.ssm_state_dtype,
            )
            load_initial_state_from_block_map(
                attention_inputs.prefix_lengths_d,
                attention_inputs.kv_cache_kernel_block_id_device,
                ssm_states,
                initial_states,
                seq_size_per_block,
            )

        if full_mixed_qkv.shape[0] >= 2048 and gdn.head_k_dim == gdn.head_v_dim:
            query, key, value = scatter_qkv(
                full_mixed_qkv,
                gdn.local_num_k_heads,
                gdn.local_num_v_heads,
                gdn.head_k_dim,
                gdn.head_v_dim,
            )
        else:
            query, key, value = torch.split(
                full_mixed_qkv,
                [
                    gdn.local_num_k_heads * gdn.head_k_dim,
                    gdn.local_num_k_heads * gdn.head_k_dim,
                    gdn.local_num_v_heads * gdn.head_v_dim,
                ],
                dim=-1,
            )
            query = query.view(1, -1, gdn.local_num_k_heads, gdn.head_k_dim)
            key = key.view(1, -1, gdn.local_num_k_heads, gdn.head_k_dim)
            value = value.view(1, -1, gdn.local_num_v_heads, gdn.head_v_dim)

        attn_out, h, final_state = chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_states,
            output_final_state=True,
            cu_seqlens=full_cu,
            use_qk_l2norm_in_kernel=True,
        )

        if ssm_states is not None:
            store_ssm_state_to_block_map(
                h,
                final_state,
                attention_inputs.prefix_lengths_d,
                full_cu,
                attention_inputs.kv_cache_kernel_block_id_device,
                ssm_states,
                seq_size_per_block,
                chunk_size=64,
            )

        if kv_cache is not None and attn_meta.cp_write_cache_store_impl is not None:
            attn_meta.cp_write_cache_store_impl(kv_cache)

        full_attn_out = attn_out.squeeze_(0)

        n_local = z.shape[0]
        local_attn_out = torch.zeros(
            n_local,
            *full_attn_out.shape[1:],
            device=full_attn_out.device,
            dtype=full_attn_out.dtype,
        )
        valid_mask = attn_meta.cp_local_valid_mask
        local_attn_out[valid_mask] = full_attn_out[attn_meta.cp_local_extract_indices]

        local_attn_out = self.norm(
            local_attn_out.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim),
        )
        local_attn_out = local_attn_out.reshape(
            -1, self.local_num_v_heads * self.head_v_dim
        )
        local_attn_out = self.out_proj(local_attn_out)
        return local_attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        attention_inputs: Optional[PyAttentionInputs],
        attn_meta: Qwen3NextMetadata,
    ) -> torch.Tensor:
        assert attention_inputs is not None, "attention_inputs is required"
        assert (
            attention_inputs.is_target_verify
            or not attention_inputs.is_prefill
            or attn_meta.get_prefill_conv1d_meta() is not None
            or attn_meta.is_cp_linear_attn
        ), "prefill_conv1d_meta is required for prefill"
        projected_states_qkvz, projected_states_ba = self._input_project(hidden_states)
        mixed_qkv, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        trace_ctx = attn_meta.trace_ctx
        layer_idx = getattr(attn_meta, "trace_layer_idx", None)
        group_id = getattr(attn_meta, "trace_group_id", None)
        _trace_event(
            trace_ctx,
            "linear_project",
            layer_idx=layer_idx,
            layer_type=HybridAttentionType.LINEAR,
            group_id=group_id,
            attention_inputs=attention_inputs,
            tensors={
                "hidden_states": hidden_states,
                "mixed_qkv": mixed_qkv,
                "z": z,
                "b": b,
                "a": a,
            },
        )
        if attention_inputs.is_prefill and not attn_meta.is_target_verify:
            if attn_meta.is_cp_linear_attn:
                return self._forward_cp_prefill(
                    mixed_qkv, z, b, a, attention_inputs, kv_cache, attn_meta
                )
            attn_output = self.prefill_gdn(
                mixed_qkv, b, a, attention_inputs, kv_cache, attn_meta
            )
        else:
            attn_output = self.decode_gdn(
                mixed_qkv, b, a, attention_inputs, kv_cache, attn_meta
            )
        _trace_event(
            trace_ctx,
            "linear_raw_output",
            layer_idx=layer_idx,
            layer_type=HybridAttentionType.LINEAR,
            group_id=group_id,
            attention_inputs=attention_inputs,
            tensors={"attn_output": attn_output, "z": z},
        )
        attn_output = self.norm(
            attn_output.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
        )
        # from [token * head, dim] -> [token, head * dim]
        attn_output = attn_output.reshape(-1, self.local_num_v_heads * self.head_v_dim)
        _trace_event(
            trace_ctx,
            "linear_norm_output",
            layer_idx=layer_idx,
            layer_type=HybridAttentionType.LINEAR,
            group_id=group_id,
            attention_inputs=attention_inputs,
            tensors={"attn_output": attn_output},
        )
        attn_output = self.out_proj(attn_output)
        _trace_event(
            trace_ctx,
            "linear_out_proj_output",
            layer_idx=layer_idx,
            layer_type=HybridAttentionType.LINEAR,
            group_id=group_id,
            attention_inputs=attention_inputs,
            tensors={"attn_output": attn_output},
        )
        if self.parallelism_config.get_attn_tp_size() > 1:
            attn_output = all_reduce(attn_output, group=Group.TP)
        _trace_event(
            trace_ctx,
            "linear_output",
            layer_idx=layer_idx,
            layer_type=HybridAttentionType.LINEAR,
            group_id=group_id,
            attention_inputs=attention_inputs,
            tensors={"attn_output": attn_output},
        )
        return attn_output


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        moe_config,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.hybrid_attention_config.hybrid_attention_types[
            layer_idx
        ]
        if self.layer_type == HybridAttentionType.LINEAR:
            self.self_attn = Qwen3NextGatedDeltaNet(
                config.linear_attention_config,
                parallelism_config,
                weights,
                config.layernorm_eps,
                config.quant_config,
                hw_kernel_config=hw_kernel_config,
            )
        else:
            attn_configs = config.getAttentionConfigs(
                parallelism_config.get_attn_tp_size()
            )
            self.self_attn = Qwen3NextAttention(
                attn_configs,
                parallelism_config,
                weights,
                config.layernorm_eps,
                config.quant_config,
                hw_kernel_config=hw_kernel_config,
            )

        if config.moe_style == 2:
            self.mlp = GenericMoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph,
                hw_kernel_config=hw_kernel_config,
            )
        elif config.moe_style == 0:
            self.mlp = DenseMLP(
                config.activation_type,
                parallelism_config,
                weights,
                config.quant_config,
                hw_kernel_config=hw_kernel_config,
            )

        self.input_layernorm = RMSResNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSResNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        attn_meta: Qwen3NextMetadata = Qwen3NextMetadata(),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
            attention_inputs=attention_inputs,
            attn_meta=attn_meta,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Qwen3NextModel(GptModelBase):
    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        # Get enable_cuda_graph from py_hw_kernel_config
        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                Qwen3NextDecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSResNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )
        self._cuda_graph_layer_probe = (
            _CudaGraphLayerProbe(True, _Q3N_GRAPH_PROBE_LAYERS, self.layer_num)
            if _Q3N_GRAPH_PROBE_ENABLED
            else None
        )

    def get_cuda_graph_probe_buffer(
        self, graph_bs: int
    ) -> Optional[tuple[torch.Tensor, tuple[int, ...]]]:
        if self._cuda_graph_layer_probe is None:
            return None
        return self._cuda_graph_layer_probe.get_capture(graph_bs)

    def get_cuda_graph_probe_debug_status(self, graph_bs: int) -> dict[str, Any]:
        probe = self._cuda_graph_layer_probe
        if probe is None:
            return {
                "module_env_enabled": _Q3N_GRAPH_PROBE_ENABLED,
                "probe_created": False,
                "buffer_available": False,
                "layers": (),
                "buffer_bucket_bs": (),
                "record_debug": {
                    "attempts": 0,
                    "recorded": 0,
                    "skipped_not_cuda_graph": 0,
                    "skipped_invalid_tensor": 0,
                    "skipped_invalid_layout": 0,
                    "last_layer_idx": -1,
                    "last_graph_bs": -1,
                    "last_token_rows": -1,
                    "last_residual_rows": -1,
                    "last_is_cuda_graph": -1,
                },
            }
        return {
            "module_env_enabled": _Q3N_GRAPH_PROBE_ENABLED,
            "probe_created": True,
            "buffer_available": probe.get_buffer(graph_bs) is not None,
            "layers": probe.layers,
            "buffer_bucket_bs": tuple(sorted(probe._buffers)),
            "record_debug": probe.get_debug_status(),
        }

    def _build_cp_linear_attn_metadata(
        self,
        attention_inputs: PyAttentionInputs,
        device: torch.device,
    ) -> tuple[
        Optional[CausalConv1dMetadata],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Precompute metadata for CP linear attention (per-layer all-gather path).

        Returns (full_conv1d_meta, full_cu_seqlens, restore_indices,
                 local_extract_indices, local_valid_mask).
        """
        cp_info = attention_inputs.context_parallel_info
        if cp_info is None:
            return None, None, None, None, None

        # In CP mode the TP group is repurposed as the CP group, so tp_size == cp_size.
        cp_size = self.parallelism_config.tp_size
        cp_rank = self.parallelism_config.tp_rank

        full_new_lengths = cp_info.prefill_actual_input_lengths_cpu
        full_cu = torch.zeros(
            full_new_lengths.shape[0] + 1, dtype=torch.int32, device=device
        )
        full_cu[1:] = full_new_lengths.cumsum(0).to(device)
        full_conv1d_meta = prepare_causal_conv1d_metadata(
            query_start_loc=full_cu, device=device
        )

        restore_indices = cp_info.prefill_qkv_restore_indice
        padding_mask = cp_info.prefill_qkv_padding_mask
        unpad_restore = restore_indices[padding_mask == 1]

        total_ag = padding_mask.shape[0]
        local_chunk_total = total_ag // cp_size
        local_start = cp_rank * local_chunk_total

        inv_restore = torch.full((total_ag,), -1, dtype=torch.long, device=device)
        inv_restore[unpad_restore.long()] = torch.arange(
            unpad_restore.shape[0], device=device
        )

        local_inv = inv_restore[local_start : local_start + local_chunk_total]
        cp_local_valid_mask = local_inv >= 0
        cp_local_extract_indices = local_inv[cp_local_valid_mask]

        return (
            full_conv1d_meta,
            full_cu,
            restore_indices,
            cp_local_extract_indices,
            cp_local_valid_mask,
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        attention_inputs: PyAttentionInputs = inputs.attention_inputs
        graph_probe_bs = (
            int(attention_inputs.input_lengths.size(0))
            if self._cuda_graph_layer_probe is not None
            else 0
        )
        prefill_conv1d_meta = None
        is_target_verify = attention_inputs.is_target_verify
        is_cp = self.parallelism_config.prefill_cp_config.is_enabled()

        full_prefill_conv1d_meta = None
        full_prefill_cu_seqlens = None
        cp_restore_indices = None
        cp_local_extract_indices = None
        cp_local_valid_mask = None
        cp_write_cache_store_impl = None

        if attention_inputs.is_prefill and not is_target_verify:
            if is_cp:
                (
                    full_prefill_conv1d_meta,
                    full_prefill_cu_seqlens,
                    cp_restore_indices,
                    cp_local_extract_indices,
                    cp_local_valid_mask,
                ) = self._build_cp_linear_attn_metadata(
                    attention_inputs, hidden_states.device
                )
                if attention_inputs.cache_store_inputs:
                    cp_info = attention_inputs.context_parallel_info
                    cp_write_cache_store_impl = WriteCacheStoreOp(
                        cp_info.prefill_actual_input_lengths_cpu,
                        attention_inputs.prefix_lengths,
                        attention_inputs.kv_cache_block_id_host,
                        attention_inputs.cache_store_inputs,
                    )
            else:
                cu_seqlen_without_padding = attention_inputs.cu_seqlens
                prefill_conv1d_meta = prepare_causal_conv1d_metadata(
                    query_start_loc=cu_seqlen_without_padding,
                    device=hidden_states.device,
                )

        attn_meta = Qwen3NextMetadata(
            prefill_conv1d_meta=prefill_conv1d_meta,
            is_target_verify=is_target_verify,
            full_prefill_conv1d_meta=full_prefill_conv1d_meta,
            full_prefill_cu_seqlens=full_prefill_cu_seqlens,
            cp_restore_indices=cp_restore_indices,
            cp_local_extract_indices=cp_local_extract_indices,
            cp_local_valid_mask=cp_local_valid_mask,
            cp_write_cache_store_impl=cp_write_cache_store_impl,
        )
        trace_ctx = _make_trace_ctx(inputs, attention_inputs, hidden_states)
        attn_meta.trace_ctx = trace_ctx

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        residual = torch.zeros_like(hidden_states)
        _trace_event(
            trace_ctx,
            "model_decode_start",
            attention_inputs=attention_inputs,
            tensors={"input_ids": input_ids, "hidden_states": hidden_states},
        )

        for i, decoder_layer in enumerate(self.layers):
            group_id = select_block_map_for_layer(attention_inputs, i)
            layer_type = decoder_layer.layer_type
            attn_meta.trace_layer_idx = i
            attn_meta.trace_group_id = group_id
            if _trace_layer_enabled(i, layer_type):
                _trace_event(
                    trace_ctx,
                    "layer_start",
                    layer_idx=i,
                    layer_type=layer_type,
                    group_id=group_id,
                    attention_inputs=attention_inputs,
                    tensors={"hidden_states": hidden_states, "residual": residual},
                )
            hidden_states, residual = decoder_layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                attention_inputs=attention_inputs,
                attn_meta=attn_meta,
            )
            if self._cuda_graph_layer_probe is not None:
                self._cuda_graph_layer_probe.record(
                    i,
                    hidden_states,
                    residual,
                    graph_bs=graph_probe_bs,
                    is_cuda_graph=bool(
                        getattr(attention_inputs, "is_cuda_graph", False)
                    ),
                )
            if _trace_layer_enabled(i, layer_type):
                _trace_event(
                    trace_ctx,
                    "layer_end",
                    layer_idx=i,
                    layer_type=layer_type,
                    group_id=group_id,
                    attention_inputs=attention_inputs,
                    tensors={"hidden_states": hidden_states, "residual": residual},
                )

        hidden_states, residual = self.norm(hidden_states, residual)
        _trace_event(
            trace_ctx,
            "model_decode_end",
            attention_inputs=attention_inputs,
            tensors={"hidden_states": hidden_states, "residual": residual},
        )
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
