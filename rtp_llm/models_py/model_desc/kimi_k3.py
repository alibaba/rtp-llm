"""RTP-LLM serving model for the text-only Kimi K3 decoder.

The hybrid decoder interleaves KDA (linear-attention) and MLA (full-attention)
layers. MLA reuses the framework cache and attention kernels while K3 owns its
NoPE convention, sigmoid output gate, packed input projection and
Sequence-Parallel projection boundary. KDA runs K3's own optimized path:
packed prefill dispatches to cuLA and token decode dispatches to the recurrent
Triton kernel. KDA canonical states are
mapped onto RTP's paged linear-cache ABI; MLA uses RTP's compressed latent
cache layout, so the same layer caches can flow through PD transfer.
"""

from __future__ import annotations

import inspect
import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

import rtp_llm.ops.compute_ops as compute_ops
from rtp_llm.model_loader.linear_attn_weight import split_kda_qkvg_fa_beta_sections
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3WeightNames as K3W
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather,
    all_gather_into,
    all_gather_trim,
    all_reduce,
    all_to_all_single,
    barrier,
    get_process_group,
    reduce_scatter,
    reduce_scatter_padded,
)
from rtp_llm.models_py.distributed.fused_all_gather_matmul import (
    fused_all_gather_matmul,
    reserve_fused_all_gather_matmul_workspace,
)
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.kimi_k3_cuda_graph_cache import (
    load_cuda_graph_decode_tensors,
    store_cuda_graph_decode_state,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.base import GroupTopK
from rtp_llm.models_py.modules.base.common.kvcache_store import (
    create_write_cache_store_impl,
)
from rtp_llm.models_py.modules.hybrid.mla_attention import MlaAttention
from rtp_llm.models_py.modules.kimi_k3.diagnostics.accuracy_trace import (
    accuracy_trace_mode,
    kimi_k3_accuracy_trace,
    mark_accuracy_fake_stream,
    prepare_kimi_kda_trace_inputs,
    record_accuracy_tensor,
    tensor_dump_enabled,
    tensor_dump_full_router,
)
from rtp_llm.models_py.modules.kimi_k3.kda_state import KDAExecutionMode, KimiKDAState
from rtp_llm.models_py.modules.kimi_k3.mxfp4 import dequantize_mxfp4
from rtp_llm.models_py.triton_kernels.causal_conv1d import causal_conv1d_update
from rtp_llm.models_py.triton_kernels.kimi_kda import (
    chunk_kda,
    fused_recurrent_kda,
    is_kimi_k3_attn_res_supported,
    is_kimi_kda_short_conv_paged_decode_supported,
    kimi_k3_a2a_unpack_rms_norm_sigmoid_gate,
    kimi_k3_attn_res,
    kimi_k3_interleave_tp_hidden,
    kimi_k3_pack_a2a_projection,
    kimi_k3_rms_norm_strided,
    kimi_k3_situ,
    kimi_k3_store_linear_cache_state,
    kimi_k3_two_way_attn_res,
    kimi_kda_rms_norm_sigmoid_gate,
    kimi_kda_short_conv_decode,
    kimi_kda_short_conv_paged_decode,
    kimi_kda_short_conv_prefill,
)
from rtp_llm.models_py.utils.typed_storage_view import LinearCacheConverter
from rtp_llm.ops import HybridAttentionType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W
from rtp_llm.utils.util import to_torch_dtype

if TYPE_CHECKING:
    from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3ModelConfig


KIMI_K3_MLA_LATENT_NORM_EPS = 1e-6
# 上游 f59c38cb0 新增的融合 AG GEMM 门限,保留。
_FUSED_AG_GEMM_MIN_GLOBAL_TOKENS = 32 * 1024
# Headroom the KDA A2A preflight leaves free after replicating the per-layer
# weights.  It used to be tunable through KIMI_K3_KDA_A2A_SAFETY_GIB; nobody
# ever moved it off 8 GiB, and A2A is capped at three layers anyway, so the
# knob was pure surface area.
_KDA_A2A_SAFETY_BYTES = 8 * (1 << 30)
_CULA_LOGGED_DEVICES: set[int] = set()
_DEEPGEMM_MEGA_LOGGED_DEVICES: set[int] = set()


@dataclass(frozen=True)
class _TokenShardLayout:
    """Equal contiguous TP shards with padding only after the logical tail."""

    logical_tokens: int
    local_tokens: int
    local_start: int
    local_valid_tokens: int


def _token_shard_layout(
    logical_tokens: int,
    tp_size: int,
    tp_rank: int,
) -> _TokenShardLayout:
    local_tokens = (logical_tokens + tp_size - 1) // tp_size
    local_start = tp_rank * local_tokens
    return _TokenShardLayout(
        logical_tokens=logical_tokens,
        local_tokens=local_tokens,
        local_start=local_start,
        local_valid_tokens=max(0, min(local_tokens, logical_tokens - local_start)),
    )


def _env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").strip() == "1"


def _is_prefill_role(parallelism_config) -> bool:
    """Prefill vs Decode,取引擎已经镜像进 parallelism_config 的 role。

    engine_config.py 把 role_config.role_type 镜像到 parallelism_config,正是为了
    让模型构造期不必去读 os.environ["ROLE_TYPE"];DSv4 也是这么取的。K3 只以 PD
    分离部署,所以 PDFUSION 这类取值是配置错误,直接报出来而不是默默当 Decode。
    """

    role = str(parallelism_config.role_type).rsplit(".", 1)[-1].upper()
    if role == "PREFILL":
        return True
    if role == "DECODE":
        return False
    raise RuntimeError(f"Kimi K3 只支持 PD 分离部署,role_type={role} 不受支持")


def _perf_fusions_enabled() -> bool:
    """Select the explicitly staged performance-fusion implementations.

    恒定开启:两个角色的生产配置都是 1,精度已按这个组合封版。原先的
    KIMI_K3_PERF_FUSIONS 只有 Prefill 侧 KDA=kernel 时才有第二个取值,而那正是
    实测会破坏精度的坏组合(融合 kernel 是围绕 cuLA 设计与验证的)—— 现在 KDA
    按角色固定,这个开关就没有合法的第二个值了。
    """

    return True


def _fused_ag_gemm_mode() -> str:
    mode = os.environ.get("KIMI_K3_FUSED_AG_GEMM", "auto").strip().lower()
    if mode not in ("auto", "off", "force"):
        raise ValueError(
            f"KIMI_K3_FUSED_AG_GEMM must be auto, off, or force; got {mode!r}"
        )
    return mode


def _batched_kda_decode_enabled() -> bool:
    """Enable the experimental indexed KDA decode path."""

    return _env_flag("KIMI_K3_BATCHED_KDA_DECODE")


def _perf_profile(name: str, tensor: Optional[torch.Tensor] = None):
    # PERF_MODE 已删:profiler 标注恒不产出。保留这个壳是因为调用点遍布前向,
    # 一并摘掉会把改动摊到几十处无关的地方。
    if True:
        return nullcontext()
    suffix = ""
    if tensor is not None:
        shape = "x".join(str(dim) for dim in tensor.shape)
        suffix = f"[shape={shape},dtype={tensor.dtype}]"
    return torch.autograd.profiler.record_function(f"{name}{suffix}")


def _kda_comm_backend() -> str:
    # KDA 通信只有 rs_ag 一种生产实现。这里原先读 KIMI_K3_KDA_COMM_BACKEND,
    # 但代码对任何其它取值都直接抛错 —— 这个变量事实上没有第二个合法值。
    return "rs_ag"


def _debug_enabled() -> bool:
    """One switch for every K3 diagnostic log stream (Decode SP, PD transfer)."""

    return _env_flag("KIMI_K3_DEBUG")


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply an RTP internal-layout ``[in_features, out_features]`` weight."""

    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"linear input width {x.shape[-1]} does not match weight "
            f"shape {tuple(weight.shape)}"
        )
    return torch.matmul(x, weight)


def _accuracy_trace_enabled() -> bool:
    """Return whether this forward is actively recording accuracy tensors."""

    return accuracy_trace_mode() is not None


def _accuracy_trace_requested() -> bool:
    """Return the process-wide trace selector shared by every TP rank."""

    return bool(os.environ.get("KIMI_K3_TENSOR_DUMP"))


def _host_metadata_enabled() -> bool:
    """Use gather-time pinned host metadata instead of synchronous D2H reads."""

    # 恒定开启:档 A→B 实测精度无差异,生产两端都是 1。
    return True


def _prepare_mla_fmha_for_group(
    fmha_impl: Any,
    attention_inputs: PyAttentionInputs,
    selected_group_id: int,
    prepared_group_id: Optional[int],
) -> int:
    """Refresh cached MLA params when HybridCache switches FULL groups.

    FlashInfer MLA derives ``slot_mapping`` and its page table from the
    singular block-map fields during ``prepare``.  K3 owns several FULL cache
    groups, so changing only ``attention_inputs`` leaves the wrapper writing
    every later MLA layer through group 0's slot mapping.
    """

    if selected_group_id == prepared_group_id:
        return selected_group_id
    sequence_lengths = getattr(attention_inputs, "sequence_lengths", None)
    is_capturing = bool(
        sequence_lengths is not None
        and sequence_lengths.is_cuda
        and torch.cuda.is_current_stream_capturing()
    )
    if is_capturing:
        prepare_group = getattr(fmha_impl, "prepare_cuda_graph_group", None)
        if not callable(prepare_group):
            raise RuntimeError(
                "Kimi K3 HybridCache MLA requires graph-safe group refresh "
                "during CUDA Graph capture"
            )
        prepare_group(attention_inputs)
        return selected_group_id

    prepare = getattr(fmha_impl, "prepare", None)
    if not callable(prepare):
        raise RuntimeError(
            "Kimi K3 HybridCache MLA requires an FMHA implementation with prepare()"
        )
    prepare(attention_inputs)
    return selected_group_id


def _select_mla_attention_inputs(
    explicit_inputs: Optional[PyAttentionInputs],
    fmha_impl: Any,
) -> Optional[PyAttentionInputs]:
    """Select the group-current attention-input view for K3 MLA."""

    if explicit_inputs is not None:
        return explicit_inputs
    return getattr(fmha_impl, "attn_inputs", None)


def _replicated_column_weight(
    local_weight: torch.Tensor,
    tp_size: int,
    cache: dict[str, torch.Tensor],
    cache_key: str,
) -> torch.Tensor:
    """Gather a column-sharded ``[in, out/tp]`` weight on every TP rank."""

    if tp_size <= 1:
        return local_weight
    full_weight = cache.get(cache_key)
    if full_weight is None:
        full_weight = (
            all_gather(local_weight.transpose(0, 1).contiguous(), group=Group.TP)
            .transpose(0, 1)
            .contiguous()
        )
        cache[cache_key] = full_weight
    return full_weight


def _replicated_row_weight(
    local_weight: torch.Tensor,
    tp_size: int,
    cache: dict[str, torch.Tensor],
    cache_key: str,
) -> torch.Tensor:
    """Gather a row-sharded ``[in/tp, out]`` weight on every TP rank."""

    if tp_size <= 1:
        return local_weight
    full_weight = cache.get(cache_key)
    if full_weight is None:
        full_weight = all_gather(local_weight.contiguous(), group=Group.TP)
        cache[cache_key] = full_weight
    return full_weight


def _sequence_parallel_column_weight(
    weights: Dict[str, torch.Tensor],
    weight_name: str,
    tp_size: int,
    tp_rank: int,
    cache: dict[str, torch.Tensor],
    cache_key: str,
    *,
    sequence_parallel: bool,
) -> torch.Tensor:
    """Materialize and cache the full column weight for token-sharded SP."""

    if tp_size <= 1:
        return weights[weight_name]
    full_weight = cache.get(cache_key)
    if sequence_parallel:
        if full_weight is None:
            full_weight = _replicated_column_weight(
                weights[weight_name],
                tp_size,
                cache,
                cache_key,
            )
            # Drop the independently allocated TP shard. The cache and weight
            # dictionary now share the same full-weight storage.
            weights[weight_name] = full_weight
        return full_weight
    if full_weight is None:
        return weights[weight_name]
    local_width = full_weight.shape[1] // tp_size
    begin = tp_rank * local_width
    return full_weight[:, begin : begin + local_width]


def _sequence_parallel_row_weight(
    weights: Dict[str, torch.Tensor],
    weight_name: str,
    tp_size: int,
    tp_rank: int,
    cache: dict[str, torch.Tensor],
    cache_key: str,
    *,
    sequence_parallel: bool,
) -> torch.Tensor:
    """Materialize an SP row replica once and replace the original TP shard."""

    if tp_size <= 1:
        return weights[weight_name]
    full_weight = cache.get(cache_key)
    if sequence_parallel:
        if full_weight is None:
            full_weight = _replicated_row_weight(
                weights[weight_name],
                tp_size,
                cache,
                cache_key,
            )
            weights[weight_name] = full_weight
        return full_weight
    if full_weight is None:
        return weights[weight_name]
    local_height = full_weight.shape[0] // tp_size
    begin = tp_rank * local_height
    return full_weight[begin : begin + local_height]


def _use_fused_prefill_ag_gemm(global_token_count: int) -> bool:
    """Whether K3 should fuse this Prefill AllGather and projection."""

    mode = _fused_ag_gemm_mode()
    if mode == "off" or global_token_count < _FUSED_AG_GEMM_MIN_GLOBAL_TOKENS:
        return False
    if mode == "force":
        return True
    return (
        _perf_fusions_enabled()
        # canonical / local-eager 对照模式已删除,这里只剩 trace:tracing 需要
        # 源实现的投影边界,所以不能走融合 AG GEMM。
        and not _accuracy_trace_requested()
    )


def _prefill_all_gather_input(
    local_input: torch.Tensor,
    tp_size: int,
    logical_tokens: Optional[int] = None,
) -> torch.Tensor:
    gathered = all_gather_into(
        local_input,
        local_input.new_empty((local_input.shape[0] * tp_size, *local_input.shape[1:])),
        Group.TP,
    )
    if logical_tokens is None or gathered.shape[0] == logical_tokens:
        return gathered
    return gathered.narrow(0, 0, logical_tokens)


def _prefill_all_gather_matmul(
    local_input: torch.Tensor,
    weight: torch.Tensor,
    *,
    tp_size: int,
    logical_tokens: int,
) -> torch.Tensor:
    """Gather equal shards, project them, and return logical Prefill rows.

    The fused operator consumes all physical rows. The serial fallback trims
    the gathered tail before GEMM because fake rows have no consumers.
    """

    physical_tokens = local_input.shape[0] * tp_size
    if _use_fused_prefill_ag_gemm(physical_tokens):
        process_group = get_process_group(Group.TP)
        _, outputs = fused_all_gather_matmul(
            local_input,
            [weight],
            process_group,
            return_gathered=False,
        )
        output = outputs[0]
    else:
        with _perf_profile("k3_separate_all_gather_then_gemm", local_input):
            gathered = _prefill_all_gather_input(
                local_input,
                tp_size,
                logical_tokens,
            )
            output = _linear(gathered, weight)

    return (
        output
        if output.shape[0] == logical_tokens
        else output.narrow(0, 0, logical_tokens)
    )


class _KimiK3MLALatentRMSNorm(nn.Module):
    """Match the source MLA latent RMSNorm's FP32 reduction and BF16 rounding."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = weight
        self.variance_epsilon = KIMI_K3_MLA_LATENT_NORM_EPS

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _rms_norm(
            hidden_states,
            self.weight,
            self.variance_epsilon,
        )


def _row_parallel_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    tp_size: int,
    *,
    reduce_scatter_tokens: bool = False,
    pad_reduce_scatter_tokens: bool = False,
    use_input_dtype_reduce_scatter: bool = False,
) -> torch.Tensor:
    """Apply and reduce a K3 row-parallel projection.

    Decode keeps the CUDA GEMM output and collective in FP32, then rounds once
    after the reduction. Optimized Prefill uses BF16 partials to match its
    established production path. The CPU branch retains the ordinary path for
    unit tests and environments where CUDA's ``mm(out_dtype=...)`` is
    unavailable.
    """

    if pad_reduce_scatter_tokens and not reduce_scatter_tokens:
        raise ValueError(
            "pad_reduce_scatter_tokens requires reduce_scatter_tokens=True"
        )
    if tp_size <= 1:
        return _linear(x, weight)
    if (
        x.is_cuda
        and x.ndim == 2
        and x.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == x.dtype
    ):
        if (
            _perf_fusions_enabled()
            and reduce_scatter_tokens
            and use_input_dtype_reduce_scatter
        ):
            # The optimized SP path keeps the projection and token
            # ReduceScatter in BF16 for long Prefill rows. Decode uses the
            # FP32 path below: its tiny recurrent batches are sensitive to
            # a per-layer BF16 partial/collective rounding point, while the
            # FP32 collective cost is negligible at those token counts.
            partial = (
                _matmul_with_padded_rows(x, weight, tp_size, x.dtype)
                if pad_reduce_scatter_tokens
                else torch.mm(x, weight)
            )
            return reduce_scatter(partial, group=Group.TP)
        output = torch.mm(x, weight, out_dtype=torch.float32)
        if reduce_scatter_tokens:
            output = (
                reduce_scatter_padded(output, group=Group.TP)
                if pad_reduce_scatter_tokens
                else reduce_scatter(output, group=Group.TP)
            )
        else:
            output = all_reduce(output, group=Group.TP)
        return output.to(dtype=x.dtype)
    output = _linear(x, weight)
    if reduce_scatter_tokens:
        return (
            reduce_scatter_padded(output, group=Group.TP)
            if pad_reduce_scatter_tokens
            else reduce_scatter(output, group=Group.TP)
        )
    return all_reduce(output, group=Group.TP)


def _matmul_with_padded_rows(
    x: torch.Tensor,
    weight: torch.Tensor,
    tp_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    padded_rows = ((int(x.shape[0]) + tp_size - 1) // tp_size) * tp_size
    output = torch.empty(
        (padded_rows, weight.shape[1]),
        dtype=output_dtype,
        device=x.device,
    )
    valid_output = output.narrow(0, 0, x.shape[0])
    if output_dtype == x.dtype:
        torch.mm(x, weight, out=valid_output)
    else:
        torch.mm(x, weight, out_dtype=output_dtype, out=valid_output)
    if padded_rows != x.shape[0]:
        output.narrow(0, x.shape[0], padded_rows - x.shape[0]).zero_()
    return output


def _prefill_token_shard(
    tensor: torch.Tensor,
    layout: _TokenShardLayout,
) -> torch.Tensor:
    """Build one equal TP shard without materializing a global padded tensor."""

    if tensor.ndim == 0 or tensor.shape[0] != layout.logical_tokens:
        raise ValueError(
            "padded token shard expects dim0 to equal logical tokens: "
            f"shape={tuple(tensor.shape)}, logical={layout.logical_tokens}"
        )
    if layout.local_valid_tokens == layout.local_tokens:
        return tensor.narrow(
            0,
            layout.local_start,
            layout.local_tokens,
        ).contiguous()
    local = tensor.new_zeros((layout.local_tokens, *tensor.shape[1:]))
    if layout.local_valid_tokens:
        local.narrow(0, 0, layout.local_valid_tokens).copy_(
            tensor.narrow(
                0,
                layout.local_start,
                layout.local_valid_tokens,
            )
        )
    return local


def _padded_token_shard(
    tensor: torch.Tensor,
    logical_tokens: int,
    tp_size: int,
    tp_rank: int,
) -> tuple[torch.Tensor, int]:
    """Return this TP rank's equal dim-0 shard and its valid row count."""

    if tensor.ndim == 0 or tensor.shape[0] != logical_tokens:
        raise ValueError(
            "padded token shard expects dim0 to equal logical tokens: "
            f"shape={tuple(tensor.shape)}, logical={logical_tokens}"
        )
    if tp_size <= 1:
        return tensor, logical_tokens
    tokens_per_rank = (logical_tokens + tp_size - 1) // tp_size
    padded_tokens = tokens_per_rank * tp_size
    if padded_tokens != logical_tokens:
        tensor = torch.cat(
            (
                tensor,
                tensor.new_zeros(
                    [padded_tokens - logical_tokens] + list(tensor.shape[1:])
                ),
            ),
            dim=0,
        )
    begin = tp_rank * tokens_per_rank
    valid_tokens = max(0, min(tokens_per_rank, logical_tokens - begin))
    return tensor.narrow(0, begin, tokens_per_rank).contiguous(), valid_tokens


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if _perf_fusions_enabled() and x.is_cuda:
        if x.ndim == 2 and x.stride(-1) == 1:
            return kimi_k3_rms_norm_strided(x, weight, eps)
        output = torch.empty_like(x)
        compute_ops.rtp_llm_ops.rmsnorm(
            output,
            x,
            weight,
            float(eps),
            torch.cuda.current_stream().cuda_stream,
        )
        return output
    x_float = x.float()
    normalized = x_float * torch.rsqrt(
        x_float.square().mean(dim=-1, keepdim=True) + eps
    )
    # Match KimiRMSNorm's rounding point exactly: the normalized activation is
    # cast back to the input dtype before the affine multiply.  Keeping the
    # multiply in FP32 changes BF16 values by one or two ULPs, which is enough
    # to change a routed expert when the top-k boundary margin is small.
    return weight * normalized.to(dtype=x.dtype)


def _situ(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float,
    linear_beta: Optional[float],
) -> torch.Tensor:
    if _perf_fusions_enabled() and gate.is_cuda:
        return kimi_k3_situ(gate, up, beta, linear_beta)
    gate_float = gate.float()
    up_float = up.float()
    activated_gate = beta * torch.tanh(gate_float / beta) * torch.sigmoid(gate_float)
    if linear_beta is not None:
        up_float = linear_beta * torch.tanh(up_float / linear_beta)
    return (activated_gate * up_float).to(dtype=gate.dtype)


def _attention_residual(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    norm_weight: torch.Tensor,
    projection_weight: torch.Tensor,
    eps: float,
    output_norm_weight: Optional[torch.Tensor] = None,
    output_norm_eps: Optional[float] = None,
    delta: Optional[torch.Tensor] = None,
    num_blocks: Optional[int] = None,
    block_write_idx: int = -1,
) -> torch.Tensor:
    """Exact K3 AttnRes soft selection over block anchors and prefix sum."""

    if prefix_sum.ndim != 2:
        raise ValueError("AttnRes prefix_sum must have shape [tokens, hidden]")
    if (
        block_residual.ndim != 3
        or block_residual.shape[0] != prefix_sum.shape[0]
        or block_residual.shape[2] != prefix_sum.shape[1]
    ):
        raise ValueError(
            "AttnRes block_residual must have shape [tokens, blocks, hidden]: "
            f"prefix_sum={tuple(prefix_sum.shape)}, "
            f"block_residual={tuple(block_residual.shape)}"
        )
    active_blocks = block_residual.shape[1] if num_blocks is None else int(num_blocks)
    if active_blocks < 0 or active_blocks > block_residual.shape[1]:
        raise ValueError("AttnRes valid block count is outside the residual bank")
    if block_write_idx < -1 or block_write_idx >= block_residual.shape[1]:
        raise ValueError("AttnRes block write index is outside the residual bank")
    if _perf_fusions_enabled() and is_kimi_k3_attn_res_supported(
        prefix_sum,
        block_residual,
        norm_weight,
        projection_weight,
        output_norm_weight,
        delta,
        active_blocks,
        block_write_idx,
    ):
        return kimi_k3_attn_res(
            prefix_sum,
            block_residual,
            norm_weight,
            projection_weight,
            eps,
            output_norm_weight,
            output_norm_eps,
            delta,
            active_blocks,
            block_write_idx,
        )
    if delta is not None:
        prefix_sum.add_(delta)
    if block_write_idx >= 0:
        block_residual[:, block_write_idx].copy_(prefix_sum)
    if active_blocks == 0:
        output = prefix_sum
    else:
        candidates = torch.cat(
            (block_residual[:, :active_blocks], prefix_sum.unsqueeze(1)), dim=1
        )
        candidates_float = candidates.float()
        normalized = candidates_float * torch.rsqrt(
            candidates_float.square().mean(dim=-1, keepdim=True) + eps
        )
        score_weight = norm_weight.float() * projection_weight.reshape(-1).float()
        probabilities = torch.softmax(
            (normalized * score_weight).sum(dim=-1), dim=-1
        )
        output = torch.einsum("tb,tbd->td", probabilities, candidates_float).to(
            dtype=prefix_sum.dtype
        )
    if output_norm_weight is not None:
        if output_norm_eps is None:
            raise ValueError(
                "output_norm_eps is required when output RMSNorm is requested"
            )
        return _rms_norm(output, output_norm_weight, output_norm_eps)
    return output


def _sequence_offsets(
    cu_seqlens: torch.Tensor,
    token_count: int,
    *,
    cu_seqlens_host: Optional[torch.Tensor] = None,
) -> list[tuple[int, int]]:
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a one-dimensional [batch + 1] tensor")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must use an integer dtype")
    source = (
        cu_seqlens_host
        if cu_seqlens_host is not None and cu_seqlens_host.numel()
        else cu_seqlens
    )
    offsets = [int(value) for value in source.detach().cpu().tolist()]
    if offsets[0] != 0 or offsets[-1] != token_count:
        raise ValueError(
            f"cu_seqlens must start at 0 and end at {token_count}, got {offsets}"
        )
    if any(left > right for left, right in zip(offsets, offsets[1:])):
        raise ValueError("cu_seqlens must be non-decreasing")
    return list(zip(offsets, offsets[1:]))


@torch.compiler.disable
def _packed_causal_depthwise_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    *,
    mode: KDAExecutionMode = "prefill",
    use_initial_state: Optional[bool] = None,
    sequence_ranges: Optional[list[tuple[int, int]]] = None,
    output_target: Optional[torch.Tensor] = None,
    final_state_outputs: Optional[list[torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed KDA short convolution matching Dummy/FLA arithmetic.

    The returned cache is ``[batch, channels, kernel_size - 1]`` in
    chronological order. Prefill and decode deliberately use separate Triton
    paths matching FLA's forward and cache-update kernels.
    """

    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[0]:
        raise ValueError(
            "packed causal conv expects x=[tokens,channels] and "
            "weight=[channels,kernel]"
        )
    if not x.is_cuda:
        raise RuntimeError("Kimi K3 short convolution requires CUDA")
    if sequence_ranges is not None:
        ranges = sequence_ranges
    elif mode == "decode":
        # K3 does not support target-verify yet, so recurrent decode always
        # contains exactly one token per packed sequence.  Building these
        # static ranges from tensor shapes avoids reading CUDA cu_seqlens back
        # to the host while a graph is being captured.
        if cu_seqlens.numel() != x.shape[0] + 1:
            raise ValueError(
                "KDA recurrent decode requires one cu_seqlens entry per token"
            )
        ranges = [(index, index + 1) for index in range(x.shape[0])]
    else:
        ranges = _sequence_offsets(cu_seqlens, x.shape[0])
    channels, kernel_size = weight.shape
    history_size = kernel_size - 1
    expected_state = (len(ranges), channels, history_size)
    had_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = x.new_zeros(expected_state)
    elif tuple(initial_state.shape) != expected_state:
        raise ValueError(
            f"conv state must have shape {expected_state}, got "
            f"{tuple(initial_state.shape)}"
        )

    if final_state_outputs is not None and len(final_state_outputs) != len(ranges):
        raise ValueError(
            "KDA short conv final-state output count must match packed sequences: "
            f"outputs={len(final_state_outputs)} sequences={len(ranges)}"
        )
    if output_target is not None:
        if (
            tuple(output_target.shape) != tuple(x.shape)
            or output_target.dtype != x.dtype
            or output_target.device != x.device
            or not output_target.is_contiguous()
        ):
            raise ValueError(
                "KDA short conv output target must be contiguous and match "
                f"the input: input={tuple(x.shape)}/{x.dtype}/{x.device}, "
                f"output={tuple(output_target.shape)}/{output_target.dtype}/"
                f"{output_target.device}"
            )
        if not x.is_cuda or mode != "prefill":
            raise ValueError(
                "KDA short conv output target is supported only for CUDA prefill"
            )
    outputs: list[torch.Tensor] = []
    final_states: list[torch.Tensor] = []
    for sequence_idx, (start, end) in enumerate(ranges):
        sequence = x[start:end]
        history = initial_state[sequence_idx].to(dtype=x.dtype)
        if end == start:
            output = x.new_empty((0, channels))
            combined = torch.cat((history, sequence.transpose(0, 1)), dim=-1)
            final_state = (
                combined[:, -history_size:] if history_size else combined[:, :0]
            )
        elif mode == "decode":
            if end - start != 1:
                raise ValueError(
                    "KDA recurrent decode short convolution requires exactly "
                    f"one token per sequence, got {end - start}"
                )
            output = kimi_kda_short_conv_decode(
                sequence[0],
                weight,
                history,
            ).unsqueeze(0)
            combined = torch.cat((history, sequence.transpose(0, 1)), dim=-1)
            final_state = (
                combined[:, -history_size:] if history_size else combined[:, :0]
            )
        elif mode == "prefill":
            output, kernel_final_state = kimi_kda_short_conv_prefill(
                sequence,
                weight,
                history,
                use_history=(
                    had_initial_state
                    if use_initial_state is None
                    else use_initial_state
                ),
                output=(None if output_target is None else output_target[start:end]),
                final_state=(
                    None
                    if final_state_outputs is None
                    else final_state_outputs[sequence_idx]
                ),
            )
            if _perf_fusions_enabled():
                final_state = kernel_final_state
            else:
                combined = torch.cat((history, sequence.transpose(0, 1)), dim=-1)
                final_state = (
                    combined[:, -history_size:] if history_size else combined[:, :0]
                )
        else:
            raise ValueError(f"unsupported KDA convolution mode {mode!r}")
        outputs.append(output)
        final_states.append(final_state)
    if len(outputs) == 1:
        return outputs[0], final_states[0].unsqueeze(0)
    return torch.cat(outputs, dim=0), torch.stack(final_states, dim=0)


class KimiK3LinearCacheAdapter:
    """Map backend-native KDA state to RTP's paged linear-cache byte layout.

    Each linear block stores one square recurrent-state tensor followed by a
    packed ``[history,QKV]`` convolution state.  The performance path keeps the
    selected backend's native layout in that square tensor: FlashKDA uses
    ``[H,V,K]`` while cuLA uses ``[H,K,V]``. Accuracy tracing retains the
    canonical conversion. Cached prefill writes a state at every physical
    page boundary (and at the partial tail), which makes both PD tail transfer
    and prefix reuse from an earlier page well-defined.
    """

    def __init__(
        self,
        config: KimiK3ModelConfig,
        parallelism_config: ParallelismConfig,
        local_heads: int,
        head_dim: int,
    ) -> None:
        self.local_heads = int(local_heads)
        self.head_dim = int(head_dim)
        self.projection_size = self.local_heads * self.head_dim
        self.history_size = (
            int(config.linear_attention_config.linear_conv_kernel_dim) - 1
        )
        self.converter = LinearCacheConverter(
            local_num_v_heads=self.local_heads,
            head_v_dim=self.head_dim,
            head_k_dim=self.head_dim,
            ssm_state_dtype=to_torch_dtype(
                config.linear_attention_config.ssm_state_dtype
            ),
            linear_conv_kernel_dim=int(
                config.linear_attention_config.linear_conv_kernel_dim
            ),
            qkv_size=3 * self.projection_size,
            conv_state_dtype=to_torch_dtype(
                config.linear_attention_config.conv_state_dtype
            ),
        )
        conv_section_bytes = self.projection_size * self.converter.conv_state_item_size
        self.cache_store_segment_sizes = (
            self.converter.ssm_state_size_bytes,
            *(
                (conv_section_bytes, conv_section_bytes, conv_section_bytes)
                * self.history_size
            ),
        )
        if sum(self.cache_store_segment_sizes) != self.converter.block_size_bytes:
            raise ValueError(
                "KDA cache-store segments do not cover the physical linear block: "
                f"segments={sum(self.cache_store_segment_sizes)} "
                f"block={self.converter.block_size_bytes}"
            )

    def _views(self, kv_cache: LayerKVCache) -> tuple[torch.Tensor, torch.Tensor]:
        base = kv_cache.kv_cache_base
        if base is None or base.numel() == 0:
            raise ValueError("KDA LayerKVCache has no backing tensor")
        base = base.reshape(base.shape[0], -1)
        return (
            self.converter.get_ssm_state_tensor(base),
            self.converter.get_conv_state_tensor(base),
        )

    @staticmethod
    def _lengths(
        attention_inputs: PyAttentionInputs,
        cu_seqlens: torch.Tensor,
        *,
        mode: KDAExecutionMode,
    ) -> tuple[list[int], list[int]]:
        cu_host = (
            getattr(attention_inputs, "cu_seqlens_host", None)
            if _host_metadata_enabled()
            else None
        )
        offsets = [
            int(value)
            for value in (
                cu_host
                if cu_host is not None and cu_host.numel()
                else cu_seqlens.detach().cpu()
            ).tolist()
        ]
        new_lengths = [right - left for left, right in zip(offsets, offsets[1:])]
        source_host = (
            (
                getattr(attention_inputs, "prefix_lengths_host", None)
                if mode == "prefill"
                else getattr(attention_inputs, "sequence_lengths_host", None)
            )
            if _host_metadata_enabled()
            else None
        )
        source = (
            source_host
            if source_host is not None and source_host.numel()
            else (
                attention_inputs.prefix_lengths
                if mode == "prefill"
                else attention_inputs.sequence_lengths
            )
        )
        if source is None or source.numel() == 0:
            past_lengths = [0] * len(new_lengths)
        else:
            past_lengths = [int(value) for value in source.detach().cpu().tolist()]
        if len(past_lengths) != len(new_lengths):
            raise ValueError(
                "KDA cache batch does not match packed sequence count: "
                f"past={len(past_lengths)} new={len(new_lengths)}"
            )
        return past_lengths, new_lengths

    @staticmethod
    def _block_map(attention_inputs: PyAttentionInputs) -> list[list[int]]:
        if _host_metadata_enabled():
            # Linear KDA uses one state per physical block. The selected
            # physical host table is therefore the exact map it needs.
            block_map = attention_inputs.kv_cache_block_id_host
            if block_map is None or block_map.numel() == 0:
                block_map = attention_inputs.kv_cache_kernel_block_id_host
        else:
            block_map = attention_inputs.kv_cache_kernel_block_id_device
        if block_map is None or block_map.numel() == 0 or block_map.ndim != 2:
            raise ValueError("KDA cache requires a two-dimensional kernel block map")
        return [
            [int(value) for value in row]
            for row in (
                block_map.tolist()
                if block_map.device.type == "cpu"
                else block_map.detach().cpu().tolist()
            )
        ]

    @staticmethod
    def _block_id(
        block_map: list[list[int]],
        sequence_idx: int,
        token_position: int,
        page_size: int,
    ) -> int:
        block_id = KimiK3LinearCacheAdapter._block_id_or_none(
            block_map, sequence_idx, token_position, page_size
        )
        if block_id is None:
            block_position = token_position // page_size
            raise ValueError(
                "linear cache has no materialized block at position "
                f"{block_position}"
            )
        return block_id

    @staticmethod
    def _block_id_or_none(
        block_map: list[list[int]],
        sequence_idx: int,
        token_position: int,
        page_size: int,
    ) -> Optional[int]:
        block_position = token_position // page_size
        if block_position >= len(block_map[sequence_idx]):
            raise ValueError(
                f"linear cache block map is too short for token {token_position}"
            )
        block_id = block_map[sequence_idx][block_position]
        if block_id <= 0:
            return None
        return block_id

    @staticmethod
    def _is_fake_block_row(block_row: list[int]) -> bool:
        """Return whether RTP created this row for a DP synchronization stream.

        ``NormalEngine::createMinFakeStream`` deliberately initializes every
        block id to zero.  Real cache rows use positive physical block ids and
        may contain ``-1`` for intentionally unmaterialized old linear pages.
        Keeping zero and ``-1`` distinct lets K3 execute the collective-only
        fake stream without hiding a missing page on a real request.
        """

        return bool(block_row) and all(block_id == 0 for block_id in block_row)

    @staticmethod
    def _is_fake_stream(attention_inputs: PyAttentionInputs) -> bool:
        """Use RTP's explicit DP synchronization-stream marker when available.

        The block-row check remains as a compatibility fallback for an older
        ``librtp_compute_ops`` that predates ``PyAttentionInputs.is_fake_stream``.
        Hybrid cache groups can represent a fake row differently, so block IDs
        are not a reliable primary signal.
        """

        return bool(getattr(attention_inputs, "is_fake_stream", False))

    @staticmethod
    def _is_cuda_graph_decode(
        attention_inputs: PyAttentionInputs,
        mode: KDAExecutionMode,
    ) -> bool:
        if mode != "decode":
            return False
        if bool(getattr(attention_inputs, "is_cuda_graph", False)):
            return True

        # CudaGraphRunner prepares the FMHA object with ``is_cuda_graph=True``,
        # but that Python call currently receives a pybind value-copy of
        # PyModelInputs.  The flag therefore does not always propagate to the
        # PyAttentionInputs instance passed into the captured model forward.
        # PyTorch's stream state is authoritative while recording the graph
        # and keeps this path independent of an experimental environment flag.
        sequence_lengths_plus_one = getattr(
            attention_inputs, "sequence_lengths_plus_1_d", None
        )
        return bool(
            sequence_lengths_plus_one is not None
            and sequence_lengths_plus_one.is_cuda
            and torch.cuda.is_current_stream_capturing()
        )

    def load(
        self,
        kv_cache: LayerKVCache,
        attention_inputs: PyAttentionInputs,
        cu_seqlens: torch.Tensor,
        *,
        mode: KDAExecutionMode,
    ) -> KimiKDAState:
        ssm_cache, conv_cache = self._views(kv_cache)
        page_size = int(kv_cache.seq_size_per_block)
        if page_size <= 0:
            raise ValueError("linear cache seq_size_per_block must be positive")
        if self._is_cuda_graph_decode(attention_inputs, mode):
            q_state, k_state, v_state, recurrent = load_cuda_graph_decode_tensors(
                ssm_cache,
                conv_cache,
                getattr(attention_inputs, "sequence_lengths_plus_1_d", None),
                attention_inputs.kv_cache_kernel_block_id_device,
                page_size,
                self.projection_size,
            )
            return KimiKDAState(
                q_conv_state=q_state,
                k_conv_state=k_state,
                v_conv_state=v_state,
                recurrent_state=recurrent,
            )

        past_lengths, _ = self._lengths(attention_inputs, cu_seqlens, mode=mode)
        block_map = self._block_map(attention_inputs)

        if _perf_fusions_enabled() and len(past_lengths) == 1:
            past_length = past_lengths[0]
            if (
                self._is_fake_stream(attention_inputs)
                or past_length == 0
                or self._is_fake_block_row(block_map[0])
            ):
                conv_states = conv_cache.new_zeros(
                    3, self.projection_size, self.history_size
                )
                recurrent = ssm_cache.new_zeros(
                    1, self.local_heads, self.head_dim, self.head_dim
                )
                return KimiKDAState(
                    q_conv_state=conv_states[0:1],
                    k_conv_state=conv_states[1:2],
                    v_conv_state=conv_states[2:3],
                    recurrent_state=recurrent,
                )
            block_id = self._block_id(block_map, 0, past_length - 1, page_size)
            packed_conv = conv_cache[block_id].transpose(0, 1)
            q_state, k_state, v_state = torch.split(
                packed_conv, self.projection_size, dim=0
            )
            # The physical K3 KDA cache has one backend-independent ABI:
            # [H,K,V].  cuLA Prefill and recurrent Decode use it directly;
            # a backend with a [H,V,K] native state converts while storing.
            recurrent = ssm_cache[block_id].unsqueeze(0)
            return KimiKDAState(
                q_conv_state=q_state.unsqueeze(0),
                k_conv_state=k_state.unsqueeze(0),
                v_conv_state=v_state.unsqueeze(0),
                recurrent_state=recurrent,
            )

        recurrent_states: list[torch.Tensor] = []
        q_states: list[torch.Tensor] = []
        k_states: list[torch.Tensor] = []
        v_states: list[torch.Tensor] = []
        is_fake_stream = self._is_fake_stream(attention_inputs)
        for sequence_idx, past_length in enumerate(past_lengths):
            if (
                is_fake_stream
                or past_length == 0
                or self._is_fake_block_row(block_map[sequence_idx])
            ):
                recurrent_states.append(
                    ssm_cache.new_zeros(self.local_heads, self.head_dim, self.head_dim)
                )
                empty_conv = conv_cache.new_zeros(
                    self.projection_size, self.history_size
                )
                q_states.append(empty_conv)
                k_states.append(empty_conv.clone())
                v_states.append(empty_conv.clone())
                continue
            block_id = self._block_id(
                block_map, sequence_idx, past_length - 1, page_size
            )
            recurrent_states.append(ssm_cache[block_id])
            packed_conv = conv_cache[block_id].transpose(0, 1)
            q_state, k_state, v_state = torch.split(
                packed_conv, self.projection_size, dim=0
            )
            q_states.append(q_state)
            k_states.append(k_state)
            v_states.append(v_state)
        return KimiKDAState(
            q_conv_state=torch.stack(q_states),
            k_conv_state=torch.stack(k_states),
            v_conv_state=torch.stack(v_states),
            recurrent_state=torch.stack(recurrent_states),
        )

    def store(
        self,
        state: KimiKDAState,
        kv_cache: LayerKVCache,
        attention_inputs: PyAttentionInputs,
        cu_seqlens: torch.Tensor,
        *,
        mode: KDAExecutionMode,
    ) -> None:
        ssm_cache, conv_cache = self._views(kv_cache)
        page_size = int(kv_cache.seq_size_per_block)
        if page_size <= 0:
            raise ValueError("linear cache seq_size_per_block must be positive")
        if self._is_cuda_graph_decode(attention_inputs, mode):
            store_cuda_graph_decode_state(
                state,
                ssm_cache,
                conv_cache,
                getattr(attention_inputs, "sequence_lengths_plus_1_d", None),
                attention_inputs.kv_cache_kernel_block_id_device,
                page_size,
            )
            return

        past_lengths, new_lengths = self._lengths(
            attention_inputs, cu_seqlens, mode=mode
        )
        block_map = self._block_map(attention_inputs)
        is_fake_stream = self._is_fake_stream(attention_inputs)
        for sequence_idx, (past_length, new_length) in enumerate(
            zip(past_lengths, new_lengths)
        ):
            if (
                is_fake_stream
                or new_length == 0
                or self._is_fake_block_row(block_map[sequence_idx])
            ):
                continue
            final_position = past_length + new_length - 1
            block_id = self._block_id(
                block_map, sequence_idx, final_position, page_size
            )
            self._copy_state_to_block(
                state,
                sequence_idx,
                block_id,
                ssm_cache,
                conv_cache,
                recurrent_v_first=False,
            )

    def store_position(
        self,
        state: KimiKDAState,
        state_index: int,
        kv_cache: LayerKVCache,
        attention_inputs: PyAttentionInputs,
        sequence_idx: int,
        absolute_position: int,
        *,
        block_map: Optional[list[list[int]]] = None,
        recurrent_v_first: bool = False,
    ) -> bool:
        """Store one sequence state when its linear page is materialized.

        With prefix reuse disabled RTP deliberately materializes only the last
        two linear-cache pages and leaves older page-table entries at ``-1``.
        Chunk prefill still visits those boundaries, but there is no physical
        destination to update.  When reuse is enabled (K3 uses step 1), every
        page is present and this persists every reusable boundary.
        """

        if absolute_position < 0:
            raise ValueError("KDA cache position must be non-negative")
        if self._is_fake_stream(attention_inputs):
            return False
        ssm_cache, conv_cache = self._views(kv_cache)
        block_map = (
            self._block_map(attention_inputs) if block_map is None else block_map
        )
        page_size = int(kv_cache.seq_size_per_block)
        if page_size <= 0:
            raise ValueError("linear cache seq_size_per_block must be positive")
        block_id = self._block_id_or_none(
            block_map, sequence_idx, absolute_position, page_size
        )
        if block_id is None:
            return False
        self._copy_state_to_block(
            state,
            state_index,
            block_id,
            ssm_cache,
            conv_cache,
            recurrent_v_first=recurrent_v_first,
        )
        return True

    def store_block_position(
        self,
        state: KimiKDAState,
        state_index: int,
        kv_cache: LayerKVCache,
        attention_inputs: PyAttentionInputs,
        sequence_idx: int,
        block_position: int,
        *,
        block_map: Optional[list[list[int]]] = None,
        recurrent_v_first: bool = False,
    ) -> bool:
        """Store a state checkpoint into an explicit linear-cache table slot.

        Target verification cannot use the normal ``token // page_size``
        mapping: every speculative token needs its own rollback-free state
        checkpoint even when all of them fall inside one ordinary token page.
        This mirrors qwen3-next's linear kernels, which publish token ``i`` to
        ``cal_block_idx(sequence_length) + i`` and let ``specUpdate`` commit an
        accepted checkpoint by swapping block-table entries.
        """

        if block_position < 0:
            raise ValueError("KDA cache block position must be non-negative")
        if self._is_fake_stream(attention_inputs):
            return False
        ssm_cache, conv_cache = self._views(kv_cache)
        block_map = (
            self._block_map(attention_inputs) if block_map is None else block_map
        )
        if block_position >= len(block_map[sequence_idx]):
            raise ValueError(
                "linear cache block map is too short for target-verify slot "
                f"{block_position}"
            )
        block_id = block_map[sequence_idx][block_position]
        if block_id <= 0:
            raise ValueError(
                "linear cache target-verify slot is not materialized: "
                f"sequence={sequence_idx} slot={block_position} block={block_id}"
            )
        self._copy_state_to_block(
            state,
            state_index,
            block_id,
            ssm_cache,
            conv_cache,
            recurrent_v_first=recurrent_v_first,
        )
        return True

    @staticmethod
    def _copy_state_to_block(
        state: KimiKDAState,
        state_index: int,
        block_id: int,
        ssm_cache: torch.Tensor,
        conv_cache: torch.Tensor,
        *,
        recurrent_v_first: bool,
    ) -> None:
        recurrent = state.recurrent_state[state_index]
        if recurrent_v_first:
            recurrent = recurrent.transpose(-1, -2)
        if _perf_fusions_enabled():
            kimi_k3_store_linear_cache_state(
                recurrent,
                state.q_conv_state[state_index],
                state.k_conv_state[state_index],
                state.v_conv_state[state_index],
                ssm_cache[block_id],
                conv_cache[block_id],
            )
            return
        ssm_cache[block_id].copy_(recurrent.to(dtype=ssm_cache.dtype))
        packed_conv = torch.cat(
            (
                state.q_conv_state[state_index],
                state.k_conv_state[state_index],
                state.v_conv_state[state_index],
            ),
            dim=0,
        ).transpose(0, 1)
        conv_cache[block_id].copy_(packed_conv.to(dtype=conv_cache.dtype))


class KimiK3KDA(nn.Module):
    """Weight-bound KDA layer with chunk-prefill/recurrent-decode dispatch."""

    def __init__(
        self,
        config: KimiK3ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int = -1,
    ) -> None:
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config
        self.weights = weights
        self.trace_prefix = f"layer.{layer_idx}.kda" if layer_idx >= 0 else "kda"
        runtime = config.k3_runtime_config
        self.head_dim = int(config.linear_attention_config.linear_key_head_dim)
        tp_size = int(parallelism_config.get_attn_tp_size())
        total_heads = int(config.linear_attention_config.linear_num_key_heads)
        if total_heads % tp_size:
            raise ValueError(
                f"KDA heads {total_heads} must be divisible by attention TP {tp_size}"
            )
        self.attn_tp_size = tp_size
        self.attn_tp_rank = int(parallelism_config.get_attn_tp_rank())
        self.total_heads = total_heads
        self.local_heads = total_heads // tp_size
        self._segment_cu_seqlens: dict[tuple[int, int], torch.Tensor] = {}
        self.projection_size = self.local_heads * self.head_dim
        self.full_projection_size = self.total_heads * self.head_dim
        self._segment_cu_seqlens_cpu: dict[int, torch.Tensor] = {}
        self._kda_comm_backend = _kda_comm_backend()
        logging.info(
            "[K3_KDA_COMM] tp_rank=%d backend=%s",
            self.attn_tp_rank,
            self._kda_comm_backend,
        )
        self._a2a_weights_ready = False
        self._a2a_qkvb_weight: Optional[torch.Tensor] = None
        self._a2a_g_weight: Optional[torch.Tensor] = None
        self._a2a_o_weight: Optional[torch.Tensor] = None
        self._a2a_buffers: dict[str, torch.Tensor] = {}
        self.eps = float(config.layernorm_eps)
        self.gate_lower_bound = runtime.kda_gate_lower_bound
        # KDA delta-net core backend.  Only the two production implementations
        # remain: cuLA chunked prefill and the ported Triton recurrent decode.
        # 'reference' (pure Torch), 'flash_kda' and 'fla37_precompiled' were
        # bring-up comparators and are gone.
        # Prefill 走 cula,Decode 走 kernel:随 KIMI_K3_KDA_BACKEND 一起固定下来。
        # 打包的 FLA 3.7 cubin 专用于 TP1 的 96 头状态,TP8 每 rank 只有 12 头,
        # 所以 Decode 用形状通用的 recurrent Triton kernel。
        self._kda_backend = "cula" if _is_prefill_role(parallelism_config) else "kernel"
        if not runtime.kda_use_full_rank_gate:
            raise NotImplementedError(
                "K3 checkpoint manifest currently requires full-rank KDA output gate"
            )
        self.cache_adapter = KimiK3LinearCacheAdapter(
            config,
            parallelism_config,
            self.local_heads,
            self.head_dim,
        )
        # Production rs_ag uses one rank-local physical projection. Q/K/V/G
        # are TP-head shards while F_A and beta96 are replicated. Keep section
        # views only for canonical diagnostics and downstream kernels;
        # production forward launches the complete projection as one GEMM.
        fused_projection = weights[W.linear_attn_qkvg_fa_beta_w]
        self.forget_latent_size = int(weights[W.linear_attn_f_b_w].shape[0])
        expected_fused_width = (
            4 * self.projection_size + self.forget_latent_size + self.total_heads
        )
        if fused_projection.shape[1] != expected_fused_width:
            raise ValueError(
                "fused KDA QKVG/F_A/beta width "
                f"{fused_projection.shape[1]} != {expected_fused_width}"
            )
        self.kda_fused_w = fused_projection
        fused_sections = split_kda_qkvg_fa_beta_sections(
            fused_projection,
            self.projection_size,
            self.projection_size,
            self.projection_size,
            self.projection_size,
            self.forget_latent_size,
            self.total_heads,
            dim=1,
        )
        (
            self.kda_q_w,
            self.kda_k_w,
            self.kda_v_w,
            self.kda_g_w,
            self.kda_f_a_w,
            self.kda_beta_w,
        ) = fused_sections
        self.kda_qkv_w = fused_projection[:, : 3 * self.projection_size]
        fused_conv = weights[W.linear_attn_conv1d_w].squeeze(1)
        if fused_conv.shape[0] != 3 * self.projection_size:
            raise ValueError(
                "fused KDA conv channels "
                f"{fused_conv.shape[0]} != 3*{self.projection_size}"
            )
        self.kda_conv = fused_conv
        self.kda_q_conv, self.kda_k_conv, self.kda_v_conv = torch.split(
            fused_conv, self.projection_size, dim=0
        )

    @property
    def uses_a2a_comm(self) -> bool:
        return self._kda_comm_backend == "a2a"

    def a2a_extra_weight_bytes(self) -> int:
        """Return the persistent replica increment before A2A materialization."""

        if not self.uses_a2a_comm or self._a2a_weights_ready:
            return 0
        local_weights = (
            self.weights[W.linear_attn_qkv_w],
            self.weights[W.linear_attn_b_w],
            self.weights[W.linear_attn_g_w],
            self.weights[W.linear_attn_out_w],
        )
        return (self.attn_tp_size - 1) * sum(
            tensor.numel() * tensor.element_size() for tensor in local_weights
        )

    def _a2a_buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        buffer = self._a2a_buffers.get(name)
        if (
            buffer is None
            or tuple(buffer.shape) != shape
            or buffer.dtype != reference.dtype
            or buffer.device != reference.device
        ):
            buffer = torch.empty(
                shape,
                dtype=reference.dtype,
                device=reference.device,
            )
            self._a2a_buffers[name] = buffer
        return buffer

    def materialize_a2a_weights(self) -> None:
        """Build the rank-major replicated weights once, outside measured work."""

        if not self.uses_a2a_comm or self._a2a_weights_ready:
            return
        local_fused_qkv = self.weights[W.linear_attn_qkv_w]
        local_beta = self.weights[W.linear_attn_b_w]
        local_gate = self.weights[W.linear_attn_g_w]
        local_output = self.weights[W.linear_attn_out_w]
        hidden_size = local_fused_qkv.shape[0]
        local_qkv_width = 3 * self.projection_size
        if local_fused_qkv.shape != (hidden_size, local_qkv_width):
            raise ValueError(
                "unexpected local KDA QKV weight for A2A materialization: "
                f"{tuple(local_fused_qkv.shape)}"
            )

        full_rank_major_qkv = (
            all_gather(local_fused_qkv.transpose(0, 1).contiguous(), Group.TP)
            .transpose(0, 1)
            .contiguous()
            .reshape(
                hidden_size,
                self.attn_tp_size,
                local_qkv_width,
            )
        )
        full_rank_major_beta = (
            all_gather(local_beta.transpose(0, 1).contiguous(), Group.TP)
            .transpose(0, 1)
            .contiguous()
            .reshape(
                hidden_size,
                self.attn_tp_size,
                self.local_heads,
            )
        )
        packed_qkvb = torch.cat(
            (full_rank_major_qkv, full_rank_major_beta),
            dim=2,
        ).reshape(hidden_size, -1)
        full_gate = (
            all_gather(local_gate.transpose(0, 1).contiguous(), Group.TP)
            .transpose(0, 1)
            .contiguous()
        )
        full_output = all_gather(local_output.contiguous(), Group.TP)

        expected_qkvb_width = self.attn_tp_size * (local_qkv_width + self.local_heads)
        if packed_qkvb.shape != (hidden_size, expected_qkvb_width):
            raise ValueError(
                "unexpected packed KDA QKVB weight shape: "
                f"{tuple(packed_qkvb.shape)}"
            )
        if full_gate.shape != (hidden_size, self.full_projection_size):
            raise ValueError(
                f"unexpected full KDA output-gate shape {tuple(full_gate.shape)}"
            )
        if full_output.shape != (self.full_projection_size, hidden_size):
            raise ValueError(
                f"unexpected full KDA output weight shape {tuple(full_output.shape)}"
            )

        self._a2a_qkvb_weight = packed_qkvb.contiguous()
        self._a2a_g_weight = full_gate
        self._a2a_o_weight = full_output
        # Replace the independently allocated TP shards so only the final
        # A2A layouts remain resident.  Q/K/V views would otherwise retain the
        # original fused storage.
        self.weights[W.linear_attn_qkv_w] = self._a2a_qkvb_weight
        self.weights[W.linear_attn_g_w] = self._a2a_g_weight
        self.weights[W.linear_attn_out_w] = self._a2a_o_weight
        del self.weights[W.linear_attn_b_w]
        self.kda_qkv_w = None
        self.kda_q_w = None
        self.kda_k_w = None
        self.kda_v_w = None
        self._a2a_weights_ready = True
        logging.info(
            "[K3_KDA_A2A] materialized %s qkvb=%s gate=%s output=%s",
            self.trace_prefix,
            tuple(self._a2a_qkvb_weight.shape),
            tuple(self._a2a_g_weight.shape),
            tuple(self._a2a_o_weight.shape),
        )

    def _project_fused_kda_inputs(
        self,
        hidden_states: torch.Tensor,
        *,
        prefill_sp_layout: Optional[_TokenShardLayout],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run and unpack the loader-provided Q/K/V/G/F_A/beta projection."""

        if prefill_sp_layout is not None:
            projected_fused = _prefill_all_gather_matmul(
                hidden_states,
                self.kda_fused_w,
                tp_size=self.attn_tp_size,
                logical_tokens=prefill_sp_layout.logical_tokens,
            )
        else:
            with _perf_profile(
                f"{self.trace_prefix}.qkvg_fa_beta_fused_projection",
                hidden_states,
            ):
                projected_fused = _linear(hidden_states, self.kda_fused_w)
        (
            q_projected,
            k_projected,
            v_projected,
            output_gate_projected,
            forget_latent,
            full_raw_beta,
        ) = split_kda_qkvg_fa_beta_sections(
            projected_fused,
            self.projection_size,
            self.projection_size,
            self.projection_size,
            self.projection_size,
            self.forget_latent_size,
            self.total_heads,
            dim=1,
        )
        with _perf_profile(
            f"{self.trace_prefix}.forget_gate_up_projection",
            forget_latent,
        ):
            raw_gate = _linear(forget_latent, self.weights[W.linear_attn_f_b_w])
        beta_begin = self.attn_tp_rank * self.local_heads
        raw_beta = full_raw_beta.narrow(1, beta_begin, self.local_heads)
        return (
            q_projected,
            k_projected,
            v_projected,
            raw_gate,
            raw_beta,
            output_gate_projected,
        )

    def _prepared_trace_values(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if accuracy_trace_mode() is None:
            return None
        return prepare_kimi_kda_trace_inputs(
            q,
            k,
            raw_gate,
            raw_beta,
            self.weights[W.linear_attn_alog],
            self.weights[W.linear_attn_dt_b_kda],
            lower_bound=self.gate_lower_bound,
        )

    def _paged_decode_cache(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        attention_inputs: Optional[PyAttentionInputs],
        *,
        mode: KDAExecutionMode,
    ) -> Optional[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
    ]:
        """Return device-resident paged KDA state for indexed recurrence.

        Regular decode uses one token per sequence.  Target verification uses
        multiple speculative tokens per sequence and writes every intermediate
        state directly to its reserve page.
        """

        is_target_verify = bool(
            attention_inputs is not None
            and getattr(attention_inputs, "is_target_verify", False)
        )
        if (
            (not _batched_kda_decode_enabled() and not is_target_verify)
            or (mode != "decode" and not is_target_verify)
            or kv_cache is None
            or attention_inputs is None
            or not hidden_states.is_cuda
            or _accuracy_trace_enabled()
            or self._kda_backend != "kernel"
        ):
            return None

        sequence_lengths_plus_one = getattr(
            attention_inputs, "sequence_lengths_plus_1_d", None
        )
        block_map = getattr(attention_inputs, "kv_cache_kernel_block_id_device", None)
        if (
            sequence_lengths_plus_one is None
            or block_map is None
            or not sequence_lengths_plus_one.is_cuda
            or not block_map.is_cuda
            or sequence_lengths_plus_one.ndim != 1
            or block_map.ndim != 2
            or sequence_lengths_plus_one.numel() != block_map.shape[0]
            or hidden_states.shape[0] % block_map.shape[0] != 0
            or block_map.shape[1] == 0
        ):
            return None

        ssm_cache, conv_cache = self.cache_adapter._views(kv_cache)
        page_size = int(kv_cache.seq_size_per_block)
        if (
            page_size <= 0
            or ssm_cache.dtype != torch.float32
            or ssm_cache.ndim != 4
            or tuple(ssm_cache.shape[1:])
            != (self.local_heads, self.head_dim, self.head_dim)
            or conv_cache.ndim != 3
            or tuple(conv_cache.shape[1:])
            != (self.cache_adapter.history_size, 3 * self.projection_size)
            or ssm_cache.device != hidden_states.device
            or conv_cache.device != hidden_states.device
        ):
            return None
        return (
            ssm_cache,
            conv_cache,
            block_map,
            sequence_lengths_plus_one,
            page_size,
        )

    def _paged_decode_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
        ssm_cache: torch.Tensor,
        block_map: torch.Tensor,
        sequence_lengths_plus_one: torch.Tensor,
        page_size: int,
    ) -> torch.Tensor:
        """Run one indexed recurrent launch and update physical SSM pages."""

        token_count = q.shape[0]
        batch = block_map.shape[0]
        if token_count % batch != 0:
            raise ValueError(
                f"KDA indexed token count {token_count} is not divisible by batch {batch}"
            )
        sequence_length = token_count // batch
        indexed_shape = (
            batch,
            sequence_length,
            self.local_heads,
            self.head_dim,
        )
        output, _ = fused_recurrent_kda(
            q.reshape(indexed_shape),
            k.reshape(indexed_shape),
            v.reshape(indexed_shape),
            raw_gate.reshape(indexed_shape),
            raw_beta.float().reshape(batch, sequence_length, self.local_heads),
            initial_state=ssm_cache,
            A_log=self.weights[W.linear_attn_alog],
            dt_bias=self.weights[W.linear_attn_dt_b_kda],
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=self.gate_lower_bound,
            # The physical RTP cache ABI is [H,K,V].  The kernel keeps its
            # register tile V-first internally; false makes its addresses
            # translate that tile to the cache's K-first storage.
            state_v_first=False,
            cu_seqlens=cu_seqlens,
            block_map=block_map,
            seq_size_per_block=page_size,
            sequence_lengths=sequence_lengths_plus_one,
        )
        return output.reshape(
            1, token_count, self.local_heads, self.head_dim
        ).to(dtype=q.dtype)

    def _kda_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        recurrent_state: Optional[torch.Tensor],
        *,
        mode: KDAExecutionMode,
        cu_seqlens: torch.Tensor,
        cu_seqlens_cpu: Optional[torch.Tensor] = None,
        output_target: Optional[torch.Tensor] = None,
        checkpoint_interval: Optional[int] = None,
        checkpoint_states: Optional[torch.Tensor] = None,
        backend_override: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the KDA delta-net core (l2norm + decay gate + scan).

        Uses the same ``[1,T,H,*]`` I/O and returns a ``[N,H,K,V]`` fp32 final
        state. Prefill dispatches to cuLA and decode to the ported recurrent
        Triton kernel.
        """

        kda_backend = self._kda_backend if backend_override is None else backend_override
        if kda_backend not in ("kernel", "cula"):
            raise ValueError(f"unsupported KDA backend override {kda_backend!r}")
        if not q.is_cuda:
            raise RuntimeError("Kimi K3 KDA requires CUDA")
        if checkpoint_interval is None and checkpoint_states is not None:
            raise ValueError("checkpoint_states requires checkpoint_interval")
        if checkpoint_interval is not None and (
            mode != "prefill" or kda_backend != "cula"
        ):
            raise ValueError(
                "FP32 checkpoint states are supported only by cuLA prefill"
            )

        copy_free_backend_prefill = (
            _perf_fusions_enabled()
            and kda_backend == "cula"
            and mode == "prefill"
        )
        if copy_free_backend_prefill:
            state_in = recurrent_state
            if state_in is not None and (
                state_in.dtype != torch.float32 or not state_in.is_contiguous()
            ):
                raise ValueError(
                    "K3 fused state must be contiguous FP32 in the "
                    f"{kda_backend} native layout"
                )
        else:
            # Accuracy/recurrent kernels may mutate initial_state. Feed a
            # private canonical [H,K,V] clone and convert at the kernel edge.
            state_in = (
                None
                if recurrent_state is None
                else recurrent_state.float().contiguous().clone()
            )
        if mode == "prefill":
            if kda_backend == "cula":
                try:
                    import cula
                    from cula.kda import chunk_kda as cula_chunk_kda
                except Exception as error:
                    raise RuntimeError(
                        "Prefill requires cuLA but the "
                        "cuda-linear-attention package could not be imported: "
                        f"{type(error).__name__}: {error}"
                    ) from error
                if self.gate_lower_bound is None:
                    raise RuntimeError("cuLA requires K3's finite gate lower bound")

                device_index = q.device.index if q.device.index is not None else 0
                if device_index not in _CULA_LOGGED_DEVICES:
                    logging.info(
                        "[KimiK3 cuLA] enabled device=%s package=%s version=%s",
                        q.device,
                        getattr(cula, "__file__", "<unknown>"),
                        getattr(cula, "__version__", "<unknown>"),
                    )
                    _CULA_LOGGED_DEVICES.add(device_index)
                single_sequence = int(cu_seqlens.numel()) == 2
                cula_cu_seqlens = (
                    None
                    if _perf_fusions_enabled() and single_sequence
                    else cu_seqlens.contiguous()
                )
                with (
                    torch.inference_mode(),
                    torch.autograd.profiler.record_function("k3.kda.cula"),
                ):
                    cula_result = cula_chunk_kda(
                        q.contiguous(),
                        k.contiguous(),
                        v.contiguous(),
                        raw_gate.to(dtype=q.dtype).contiguous(),
                        raw_beta.to(dtype=q.dtype),
                        scale=self.head_dim**-0.5,
                        initial_state=state_in,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                        use_gate_in_kernel=True,
                        use_beta_sigmoid_in_kernel=True,
                        cu_seqlens=cula_cu_seqlens,
                        cu_seqlens_cpu=(
                            None if cula_cu_seqlens is None else cu_seqlens_cpu
                        ),
                        safe_gate=True,
                        lower_bound=float(self.gate_lower_bound),
                        disable_recompute=False,
                        use_intracard_cp=(
                            False if checkpoint_interval is not None else "auto"
                        ),
                        A_log=a_log.float().contiguous(),
                        dt_bias=dt_bias.float().contiguous(),
                        checkpoint_interval=checkpoint_interval,
                        checkpoint_states=checkpoint_states,
                    )
                    if checkpoint_interval is None:
                        output, final_state = cula_result
                    else:
                        output, final_state, published_checkpoints = cula_result
                        if (
                            checkpoint_states is None
                            or published_checkpoints.data_ptr()
                            != checkpoint_states.data_ptr()
                        ):
                            raise RuntimeError(
                                "cuLA did not publish into the requested FP32 "
                                "checkpoint buffer"
                            )
                    if (
                        final_state is None
                        or final_state.dtype != torch.float32
                        or not final_state.is_contiguous()
                    ):
                        raise RuntimeError(
                            "cuLA must return a contiguous FP32 final state"
                        )
                    if output_target is not None:
                        output_target.copy_(output)
                        output = output_target
                return output.to(dtype=q.dtype), final_state

            # Dummy/FLA executes KDA with ``transpose_state_layout=True``.
            # Besides changing storage from [K,V] to [V,K], that selector
            # changes the TensorCore reduction order used for the chunk state
            # and output.  Keeping RTP's external/cache ABI [K,V] while using
            # the source V-first kernel path avoids a layer-dependent state
            # drift that can later flip near-tied routed experts and EOS.
            state_v_first = (
                None if state_in is None else state_in.transpose(-1, -2).contiguous()
            )
            output, final_state_v_first = chunk_kda(
                q,
                k,
                v,
                raw_gate,
                # The in-tree FLA beta preprocessor still uses flat contiguous
                # addressing. cuLA consumes the fused-prefix view directly;
                # keep this diagnostic fallback correct with an explicit copy.
                raw_beta.contiguous(),
                initial_state=state_v_first,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                A_log=a_log,
                dt_bias=dt_bias,
                safe_gate=self.gate_lower_bound is not None,
                lower_bound=self.gate_lower_bound,
                state_v_first=True,
            )
            final_state = final_state_v_first.transpose(-1, -2).contiguous()
        elif mode == "decode":
            # Match Dummy's recurrent call boundary: gate activation, beta
            # sigmoid, V-first register layout and state update all stay in
            # one Triton program.  Precomputing gate/beta or transposing the
            # register tile changes enough FP32 state ULPs to accumulate into
            # a BF16 hidden-state difference after several decode steps.
            state_v_first = (
                None if state_in is None else state_in.transpose(-1, -2).contiguous()
            )
            output, final_state_v_first = fused_recurrent_kda(
                q,
                k,
                v,
                raw_gate,
                raw_beta.float(),
                initial_state=state_v_first,
                A_log=a_log,
                dt_bias=dt_bias,
                inplace_final_state=False,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                lower_bound=self.gate_lower_bound,
                state_v_first=True,
                cu_seqlens=cu_seqlens,
            )
            final_state = final_state_v_first.transpose(-1, -2).contiguous()
        else:
            raise ValueError(f"unsupported KDA execution mode {mode!r}")
        # Normalize both backends to the [1,T,H,V] / q.dtype KDA contract.
        output = output.reshape(q.shape[0], q.shape[1], v.shape[2], v.shape[3]).to(
            dtype=q.dtype
        )
        return output, final_state

    def _cached_cula_checkpoint_prefill(
        self,
        q_projected: torch.Tensor,
        k_projected: torch.Tensor,
        v_projected: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
        initial_state: KimiKDAState,
        kv_cache: LayerKVCache,
        attention_inputs: PyAttentionInputs,
        *,
        past_length: int,
        page_size: int,
        block_map: list[list[int]],
    ) -> tuple[torch.Tensor, KimiKDAState]:
        """Run one long cuLA invocation and publish exact page checkpoints."""

        token_count = int(q_projected.shape[0])
        if token_count <= 0:
            raise ValueError("cuLA checkpoint prefill requires at least one token")
        if page_size % 64:
            raise ValueError(
                "cuLA checkpoint page size must be a multiple of 64 tokens, "
                f"got {page_size}"
            )
        if past_length % page_size:
            raise ValueError(
                "cuLA checkpoint prefill requires a page-aligned prefix, "
                f"got past_length={past_length}, page_size={page_size}"
            )
        checkpoint_count = (token_count + page_size - 1) // page_size
        q_conv = torch.empty_like(q_projected)
        k_conv = torch.empty_like(k_projected)
        v_conv = torch.empty_like(v_projected)
        sequence_range = [(0, token_count)]
        with _perf_profile(
            f"{self.trace_prefix}.all_pages.qkv_short_conv_and_final_state_export"
        ):
            q_conv_result, q_final = _packed_causal_depthwise_conv1d(
                q_projected,
                self.kda_q_conv,
                cu_seqlens,
                initial_state.q_conv_state,
                mode="prefill",
                use_initial_state=past_length > 0,
                sequence_ranges=sequence_range,
                output_target=q_conv,
            )
            k_conv_result, k_final = _packed_causal_depthwise_conv1d(
                k_projected,
                self.kda_k_conv,
                cu_seqlens,
                initial_state.k_conv_state,
                mode="prefill",
                use_initial_state=past_length > 0,
                sequence_ranges=sequence_range,
                output_target=k_conv,
            )
            v_conv_result, v_final = _packed_causal_depthwise_conv1d(
                v_projected,
                self.kda_v_conv,
                cu_seqlens,
                initial_state.v_conv_state,
                mode="prefill",
                use_initial_state=past_length > 0,
                sequence_ranges=sequence_range,
                output_target=v_conv,
            )
            if (
                q_conv_result.data_ptr() != q_conv.data_ptr()
                or k_conv_result.data_ptr() != k_conv.data_ptr()
                or v_conv_result.data_ptr() != v_conv.data_ptr()
            ):
                raise RuntimeError("K3 short convolution did not use its output target")

        recurrent_checkpoints = torch.empty(
            (
                1,
                checkpoint_count,
                self.local_heads,
                self.head_dim,
                self.head_dim,
            ),
            dtype=torch.float32,
            device=q_projected.device,
        )
        cu_seqlens_cpu = self._segment_cu_seqlens_cpu.get(token_count)
        if cu_seqlens_cpu is None:
            cu_seqlens_cpu = torch.tensor([0, token_count], dtype=torch.int32)
            self._segment_cu_seqlens_cpu[token_count] = cu_seqlens_cpu
        head_shape = (1, token_count, self.local_heads, self.head_dim)
        with _perf_profile(f"{self.trace_prefix}.all_pages.cula_recurrence_and_output"):
            output, recurrent_final = self._kda_core(
                q_conv.reshape(head_shape),
                k_conv.reshape(head_shape),
                v_conv.reshape(head_shape),
                raw_gate.reshape(head_shape),
                raw_beta.reshape(1, token_count, self.local_heads),
                self.weights[W.linear_attn_alog],
                self.weights[W.linear_attn_dt_b_kda],
                (None if past_length == 0 else initial_state.recurrent_state),
                mode="prefill",
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                checkpoint_interval=page_size,
                checkpoint_states=recurrent_checkpoints,
            )

        history_size = self.cache_adapter.history_size
        for checkpoint_index in range(checkpoint_count):
            end = min((checkpoint_index + 1) * page_size, token_count)
            absolute_end = past_length + end
            page_prefix = (
                f"{self.trace_prefix}.page.{(absolute_end - 1) // page_size}"
                f"[tokens={end - checkpoint_index * page_size},"
                f"physical_block={page_size}]"
            )
            if end >= history_size:
                q_checkpoint = (
                    q_projected.narrow(0, end - history_size, history_size)
                    .transpose(0, 1)
                    .unsqueeze(0)
                )
                k_checkpoint = (
                    k_projected.narrow(0, end - history_size, history_size)
                    .transpose(0, 1)
                    .unsqueeze(0)
                )
                v_checkpoint = (
                    v_projected.narrow(0, end - history_size, history_size)
                    .transpose(0, 1)
                    .unsqueeze(0)
                )
            else:
                q_checkpoint = q_final
                k_checkpoint = k_final
                v_checkpoint = v_final
            checkpoint_state = KimiKDAState(
                q_conv_state=q_checkpoint,
                k_conv_state=k_checkpoint,
                v_conv_state=v_checkpoint,
                recurrent_state=recurrent_checkpoints[:, checkpoint_index],
            )
            with _perf_profile(
                f"{page_prefix}.linear_cache_update_ssm_plus_3xqkv_history"
            ):
                self.cache_adapter.store_position(
                    checkpoint_state,
                    0,
                    kv_cache,
                    attention_inputs,
                    0,
                    absolute_end - 1,
                    block_map=block_map,
                )

        return output, KimiKDAState(
            q_conv_state=q_final,
            k_conv_state=k_final,
            v_conv_state=v_final,
            recurrent_state=recurrent_final,
        )

    def _cached_chunk_prefill(
        self,
        q_projected: torch.Tensor,
        k_projected: torch.Tensor,
        v_projected: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
        initial_state: KimiKDAState,
        kv_cache: LayerKVCache,
        attention_inputs: PyAttentionInputs,
    ) -> tuple[torch.Tensor, KimiKDAState]:
        """Run chunk form page-by-page and persist every reusable boundary."""

        trace_enabled = _accuracy_trace_enabled()
        is_target_verify = bool(
            getattr(attention_inputs, "is_target_verify", False)
        )
        target_verify_backend = (
            os.environ.get("KIMI_K3_TARGET_VERIFY_KDA_BACKEND")
            if is_target_verify
            else None
        )
        effective_backend = target_verify_backend or self._kda_backend
        ranges = _sequence_offsets(
            cu_seqlens,
            q_projected.shape[0],
            cu_seqlens_host=(
                getattr(attention_inputs, "cu_seqlens_host", None)
                if _host_metadata_enabled()
                else None
            ),
        )
        past_lengths, _ = self.cache_adapter._lengths(
            attention_inputs, cu_seqlens, mode="prefill"
        )
        block_map = self.cache_adapter._block_map(attention_inputs)
        page_size = int(kv_cache.seq_size_per_block)
        if page_size <= 0:
            raise ValueError("linear cache seq_size_per_block must be positive")
        if (
            _perf_fusions_enabled()
            and not is_target_verify
            and target_verify_backend is None
            and effective_backend == "cula"
            and len(ranges) == 1
            and past_lengths[0] % page_size == 0
        ):
            return self._cached_cula_checkpoint_prefill(
                q_projected,
                k_projected,
                v_projected,
                raw_gate,
                raw_beta,
                cu_seqlens,
                initial_state,
                kv_cache,
                attention_inputs,
                past_length=past_lengths[0],
                page_size=page_size,
                block_map=block_map,
            )

        packed_outputs: list[torch.Tensor] = []
        fused_output = (
            q_projected.new_empty(
                1, q_projected.shape[0], self.local_heads, self.head_dim
            )
            if _perf_fusions_enabled()
            else None
        )
        packed_q: list[torch.Tensor] = []
        packed_k: list[torch.Tensor] = []
        packed_v: list[torch.Tensor] = []
        prepared_q: list[torch.Tensor] = []
        prepared_k: list[torch.Tensor] = []
        prepared_alpha: list[torch.Tensor] = []
        prepared_beta: list[torch.Tensor] = []
        q_finals: list[torch.Tensor] = []
        k_finals: list[torch.Tensor] = []
        v_finals: list[torch.Tensor] = []
        recurrent_finals: list[torch.Tensor] = []
        for sequence_idx, ((start, end), past_length) in enumerate(
            zip(ranges, past_lengths)
        ):
            q_state = initial_state.q_conv_state[sequence_idx : sequence_idx + 1]
            k_state = initial_state.k_conv_state[sequence_idx : sequence_idx + 1]
            v_state = initial_state.v_conv_state[sequence_idx : sequence_idx + 1]
            recurrent_state = initial_state.recurrent_state[
                sequence_idx : sequence_idx + 1
            ]
            cursor = start
            absolute_position = past_length
            target_verify_write_base = past_length // page_size
            while cursor < end:
                # qwen3-next persists every target-verify intermediate state
                # into a separate reserve block.  Keep KDA's convolution and
                # recurrent scans cumulative, but cut the verify chunk into
                # one-token segments so both pieces of state form the same
                # checkpoint.
                if is_target_verify:
                    segment_end = cursor + 1
                else:
                    tokens_to_page_end = page_size - (
                        absolute_position % page_size
                    )
                    segment_end = min(end, cursor + tokens_to_page_end)
                segment_length = segment_end - cursor
                device_index = (
                    cu_seqlens.device.index
                    if cu_seqlens.device.index is not None
                    else 0
                )
                segment_key = (device_index, segment_length)
                segment_cu_seqlens = self._segment_cu_seqlens.get(segment_key)
                if segment_cu_seqlens is None:
                    segment_cu_seqlens = torch.tensor(
                        [0, segment_length],
                        dtype=torch.int32,
                        device=cu_seqlens.device,
                    )
                    self._segment_cu_seqlens[segment_key] = segment_cu_seqlens
                segment_ranges = [(0, segment_length)]
                page_index = absolute_position // page_size
                page_prefix = (
                    f"{self.trace_prefix}.page.{page_index}"
                    f"[tokens={segment_length},physical_block={page_size}]"
                )
                with _perf_profile(
                    f"{page_prefix}.qkv_short_conv_and_conv_state_export"
                ):
                    q, q_state = _packed_causal_depthwise_conv1d(
                        q_projected[cursor:segment_end],
                        self.kda_q_conv,
                        segment_cu_seqlens,
                        q_state,
                        # Target verify is deliberately segmented one token at
                        # a time above.  Use the exact recurrent decode path so
                        # row 0 is numerically identical to target-only decode;
                        # chunk-prefill arithmetic drifts from the first KDA
                        # layer and can flip rejection fallback tokens.
                        mode="decode" if is_target_verify else "prefill",
                        use_initial_state=absolute_position > 0,
                        sequence_ranges=segment_ranges,
                    )
                    k, k_state = _packed_causal_depthwise_conv1d(
                        k_projected[cursor:segment_end],
                        self.kda_k_conv,
                        segment_cu_seqlens,
                        k_state,
                        mode="decode" if is_target_verify else "prefill",
                        use_initial_state=absolute_position > 0,
                        sequence_ranges=segment_ranges,
                    )
                    v, v_state = _packed_causal_depthwise_conv1d(
                        v_projected[cursor:segment_end],
                        self.kda_v_conv,
                        segment_cu_seqlens,
                        v_state,
                        mode="decode" if is_target_verify else "prefill",
                        use_initial_state=absolute_position > 0,
                        sequence_ranges=segment_ranges,
                    )
                if trace_enabled:
                    packed_q.append(q)
                    packed_k.append(k)
                    packed_v.append(v)
                head_shape = (
                    1,
                    segment_length,
                    self.local_heads,
                    self.head_dim,
                )
                if trace_enabled:
                    prepared = self._prepared_trace_values(
                        q.reshape(head_shape),
                        k.reshape(head_shape),
                        raw_gate[cursor:segment_end].reshape(head_shape),
                        raw_beta[cursor:segment_end]
                        .float()
                        .reshape(1, segment_length, self.local_heads),
                    )
                    if prepared is not None:
                        q_work, k_work, alpha, beta = prepared
                        prepared_q.append(q_work)
                        prepared_k.append(k_work)
                        prepared_alpha.append(alpha)
                        prepared_beta.append(beta)
                beta_for_core = raw_beta[cursor:segment_end].reshape(
                    1, segment_length, self.local_heads
                )
                if not (
                    _perf_fusions_enabled()
                    and effective_backend == "cula"
                ):
                    beta_for_core = beta_for_core.float()
                with _perf_profile(
                    f"{page_prefix}.{effective_backend}_recurrence_and_output"
                ):
                    segment_cu_seqlens_cpu = None
                    if effective_backend == "cula":
                        segment_cu_seqlens_cpu = self._segment_cu_seqlens_cpu.get(
                            segment_length
                        )
                        if segment_cu_seqlens_cpu is None:
                            segment_cu_seqlens_cpu = torch.tensor(
                                [0, segment_length], dtype=torch.int32
                            )
                            self._segment_cu_seqlens_cpu[segment_length] = (
                                segment_cu_seqlens_cpu
                            )
                    segment_output, recurrent_state = self._kda_core(
                        q.reshape(head_shape),
                        k.reshape(head_shape),
                        v.reshape(head_shape),
                        raw_gate[cursor:segment_end].reshape(head_shape),
                        beta_for_core,
                        self.weights[W.linear_attn_alog],
                        self.weights[W.linear_attn_dt_b_kda],
                        recurrent_state,
                        mode="decode" if is_target_verify else "prefill",
                        cu_seqlens=segment_cu_seqlens,
                        cu_seqlens_cpu=segment_cu_seqlens_cpu,
                        output_target=(
                            None
                            if fused_output is None
                            else fused_output[:, cursor:segment_end]
                        ),
                        backend_override=target_verify_backend,
                    )
                if fused_output is None:
                    packed_outputs.append(segment_output.squeeze(0))
                segment_state = KimiKDAState(
                    q_conv_state=q_state,
                    k_conv_state=k_state,
                    v_conv_state=v_state,
                    recurrent_state=recurrent_state,
                )
                with _perf_profile(
                    f"{page_prefix}.linear_cache_update_ssm_plus_3xqkv_history"
                ):
                    if is_target_verify:
                        verify_step = cursor - start
                        self.cache_adapter.store_block_position(
                            segment_state,
                            0,
                            kv_cache,
                            attention_inputs,
                            sequence_idx,
                            target_verify_write_base + verify_step,
                            block_map=block_map,
                            recurrent_v_first=False,
                        )
                    else:
                        self.cache_adapter.store_position(
                            segment_state,
                            0,
                            kv_cache,
                            attention_inputs,
                            sequence_idx,
                            absolute_position + segment_length - 1,
                            block_map=block_map,
                            recurrent_v_first=False,
                        )
                cursor = segment_end
                absolute_position += segment_length

            q_finals.append(q_state[0])
            k_finals.append(k_state[0])
            v_finals.append(v_state[0])
            recurrent_finals.append(recurrent_state[0])

        if fused_output is not None:
            output = fused_output
        elif packed_outputs:
            output = torch.cat(packed_outputs, dim=0).unsqueeze(0)
            if trace_enabled:
                record_accuracy_tensor(
                    f"{self.trace_prefix}.q_conv",
                    torch.cat(packed_q, dim=0),
                    token_dim=0,
                )
                record_accuracy_tensor(
                    f"{self.trace_prefix}.k_conv",
                    torch.cat(packed_k, dim=0),
                    token_dim=0,
                )
                record_accuracy_tensor(
                    f"{self.trace_prefix}.v_conv",
                    torch.cat(packed_v, dim=0),
                    token_dim=0,
                )
                if prepared_q:
                    record_accuracy_tensor(
                        f"{self.trace_prefix}.prepared_q",
                        torch.cat(prepared_q, dim=1),
                        token_dim=1,
                    )
                    record_accuracy_tensor(
                        f"{self.trace_prefix}.prepared_k",
                        torch.cat(prepared_k, dim=1),
                        token_dim=1,
                    )
                    record_accuracy_tensor(
                        f"{self.trace_prefix}.alpha",
                        torch.cat(prepared_alpha, dim=1),
                        token_dim=1,
                    )
                    record_accuracy_tensor(
                        f"{self.trace_prefix}.prepared_beta",
                        torch.cat(prepared_beta, dim=1),
                        token_dim=1,
                    )
        else:
            output = q_projected.new_empty(1, 0, self.local_heads, self.head_dim)
        if _perf_fusions_enabled() and len(ranges) == 1:
            return output, KimiKDAState(
                q_conv_state=q_state,
                k_conv_state=k_state,
                v_conv_state=v_state,
                recurrent_state=recurrent_state,
            )
        return output, KimiKDAState(
            q_conv_state=torch.stack(q_finals),
            k_conv_state=torch.stack(k_finals),
            v_conv_state=torch.stack(v_finals),
            recurrent_state=torch.stack(recurrent_finals),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        mode: KDAExecutionMode,
        state: Optional[KimiKDAState] = None,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        sequence_parallel: bool = False,
        prefill_sp_layout: Optional[_TokenShardLayout] = None,
    ) -> tuple[torch.Tensor, Optional[KimiKDAState]]:
        trace_enabled = _accuracy_trace_enabled()
        if state is not None and kv_cache is not None:
            raise ValueError(
                "pass either an explicit KDA state or LayerKVCache, not both"
            )
        paged_decode_cache = self._paged_decode_cache(
            hidden_states,
            kv_cache,
            attention_inputs,
            mode=mode,
        )
        if kv_cache is not None and paged_decode_cache is None:
            if attention_inputs is None:
                raise ValueError("attention_inputs are required with a KDA cache")
            with _perf_profile(
                f"{self.trace_prefix}.cache_load_recurrent_plus_conv_history"
            ):
                state = self.cache_adapter.load(
                    kv_cache, attention_inputs, cu_seqlens, mode=mode
                )
        if trace_enabled and mode == "decode" and state is not None:
            record_accuracy_tensor(
                f"{self.trace_prefix}.cache_input.q_conv",
                state.q_conv_state,
            )
            record_accuracy_tensor(
                f"{self.trace_prefix}.cache_input.k_conv",
                state.k_conv_state,
            )
            record_accuracy_tensor(
                f"{self.trace_prefix}.cache_input.v_conv",
                state.v_conv_state,
            )
            record_accuracy_tensor(
                f"{self.trace_prefix}.cache_input.recurrent",
                state.recurrent_state,
            )
        # A2A is intentionally rejected by _kda_comm_backend(). Keep the local
        # selector until the remaining experimental post-processing code is
        # removed in a follow-up cleanup.
        a2a_prefill = False
        if prefill_sp_layout is not None and (
            mode != "prefill"
            or not sequence_parallel
            or self.attn_tp_size <= 1
            or not hidden_states.is_cuda
        ):
            raise ValueError(
                "prefill_sp_layout requires CUDA Prefill Sequence Parallel with TP>1"
            )
        local_token_count = hidden_states.shape[0]
        # 这里原本只为 canonical TP 把分片输入 all_gather 回整份;canonical 已删,
        # 分片输入直接交给下游的融合投影处理。
        token_count = hidden_states.shape[0]
        (
            q_projected,
            k_projected,
            v_projected,
            raw_gate,
            raw_beta,
            output_gate_projected,
        ) = self._project_fused_kda_inputs(
            hidden_states,
            prefill_sp_layout=prefill_sp_layout,
        )
        token_count = q_projected.shape[0]
        head_shape = (1, token_count, self.local_heads, self.head_dim)
        output_gate = output_gate_projected.reshape(head_shape)
        if trace_enabled:
            record_accuracy_tensor(
                f"{self.trace_prefix}.q_projected", q_projected, token_dim=0
            )
            record_accuracy_tensor(
                f"{self.trace_prefix}.k_projected", k_projected, token_dim=0
            )
            record_accuracy_tensor(
                f"{self.trace_prefix}.v_projected", v_projected, token_dim=0
            )
        if trace_enabled:
            record_accuracy_tensor(
                f"{self.trace_prefix}.raw_gate", raw_gate, token_dim=0
            )
            record_accuracy_tensor(
                f"{self.trace_prefix}.raw_beta", raw_beta, token_dim=0
            )
        # Target verification owns multiple reserve blocks.  Its indexed
        # kernels update those physical pages directly, just like
        # qwen3-next; do not enter the gather/one-token/scatter prefill path.
        stored_page_states = (
            mode == "prefill"
            and kv_cache is not None
            and paged_decode_cache is None
        )
        if stored_page_states:
            assert state is not None and attention_inputs is not None
            output, final_state = self._cached_chunk_prefill(
                q_projected,
                k_projected,
                v_projected,
                raw_gate,
                raw_beta,
                cu_seqlens,
                state,
                kv_cache,
                attention_inputs,
            )
        else:
            q_state = None if state is None else state.q_conv_state
            k_state = None if state is None else state.k_conv_state
            v_state = None if state is None else state.v_conv_state
            recurrent_state = None if state is None else state.recurrent_state
            if paged_decode_cache is not None:
                (
                    ssm_cache,
                    conv_cache,
                    block_map,
                    sequence_lengths_plus_one,
                    page_size,
                ) = paged_decode_cache
                is_target_verify = bool(
                    getattr(attention_inputs, "is_target_verify", False)
                )
                if is_target_verify:
                    batch = block_map.shape[0]
                    if token_count % batch != 0:
                        raise ValueError(
                            f"KDA target token count {token_count} is not "
                            f"divisible by batch {batch}"
                        )
                    sequence_length = token_count // batch
                    # Verify must be numerically identical to repeated target
                    # decode.  The generic multi-token causal_conv1d_update
                    # is not equivalent to Kimi's paged single-token kernel
                    # (including its physical state update).  Replay each
                    # speculative position through the exact decode kernel.
                    q_steps = q_projected.reshape(batch, sequence_length, -1)
                    k_steps = k_projected.reshape(batch, sequence_length, -1)
                    v_steps = v_projected.reshape(batch, sequence_length, -1)
                    conv_steps = []
                    for step in range(sequence_length):
                        reserve_base = torch.div(
                            sequence_lengths_plus_one - 2,
                            page_size,
                            rounding_mode="floor",
                        ).to(torch.long)
                        reserve_col = reserve_base + step
                        logical_col = torch.div(
                            sequence_lengths_plus_one + step - 2,
                            page_size,
                            rounding_mode="floor",
                        ).to(torch.long)
                        batch_idx = torch.arange(batch, device=block_map.device)
                        dest_ids = block_map[batch_idx, reserve_col].to(torch.long)
                        if step > 0:
                            src_ids = block_map[
                                batch_idx, reserve_col - 1
                            ].to(torch.long)
                            conv_cache[dest_ids] = conv_cache[src_ids]
                        step_block_map = block_map.clone()
                        step_block_map[batch_idx, logical_col] = dest_ids.to(
                            step_block_map.dtype
                        )
                        q_step, k_step, v_step = kimi_kda_short_conv_paged_decode(
                            q_steps[:, step, :].contiguous(),
                            k_steps[:, step, :].contiguous(),
                            v_steps[:, step, :].contiguous(),
                            self.kda_conv,
                            conv_cache,
                            step_block_map,
                            sequence_lengths_plus_one + step,
                            page_size,
                        )
                        conv_steps.append(torch.cat((q_step, k_step, v_step), dim=-1))
                    conv_output = torch.stack(conv_steps, dim=1).reshape(
                        token_count, -1
                    )
                    q, k, v = torch.split(conv_output, self.projection_size, dim=-1)
                else:
                    if not is_kimi_kda_short_conv_paged_decode_supported(
                        q_projected,
                        k_projected,
                        v_projected,
                        self.kda_conv,
                        conv_cache,
                        block_map,
                        sequence_lengths_plus_one,
                        page_size,
                    ):
                        raise RuntimeError(
                            "KDA batched decode support changed after cache selection"
                        )
                    q, k, v = kimi_kda_short_conv_paged_decode(
                        q_projected,
                        k_projected,
                        v_projected,
                        self.kda_conv,
                        conv_cache,
                        block_map,
                        sequence_lengths_plus_one,
                        page_size,
                    )
                q_final = k_final = v_final = None
            else:
                sequence_ranges = _sequence_offsets(
                    cu_seqlens,
                    token_count,
                    cu_seqlens_host=(
                        getattr(attention_inputs, "cu_seqlens_host", None)
                        if attention_inputs is not None and _host_metadata_enabled()
                        else None
                    ),
                )
                q, q_final = _packed_causal_depthwise_conv1d(
                    q_projected,
                    self.kda_q_conv,
                    cu_seqlens,
                    q_state,
                    mode=mode,
                    sequence_ranges=sequence_ranges,
                )
                k, k_final = _packed_causal_depthwise_conv1d(
                    k_projected,
                    self.kda_k_conv,
                    cu_seqlens,
                    k_state,
                    mode=mode,
                    sequence_ranges=sequence_ranges,
                )
                v, v_final = _packed_causal_depthwise_conv1d(
                    v_projected,
                    self.kda_v_conv,
                    cu_seqlens,
                    v_state,
                    mode=mode,
                    sequence_ranges=sequence_ranges,
                )
            if trace_enabled:
                prepared = self._prepared_trace_values(
                    q.reshape(head_shape),
                    k.reshape(head_shape),
                    raw_gate.reshape(head_shape),
                    raw_beta.float().reshape(1, token_count, self.local_heads),
                )
                if prepared is not None:
                    q_work, k_work, alpha, beta = prepared
                    record_accuracy_tensor(
                        f"{self.trace_prefix}.prepared_q", q_work, token_dim=1
                    )
                    record_accuracy_tensor(
                        f"{self.trace_prefix}.prepared_k", k_work, token_dim=1
                    )
                    record_accuracy_tensor(
                        f"{self.trace_prefix}.alpha", alpha, token_dim=1
                    )
                    record_accuracy_tensor(
                        f"{self.trace_prefix}.prepared_beta", beta, token_dim=1
                    )
            if paged_decode_cache is not None:
                is_target_verify = bool(
                    getattr(attention_inputs, "is_target_verify", False)
                )
                if is_target_verify and token_count > block_map.shape[0]:
                    batch = block_map.shape[0]
                    sequence_length = token_count // batch
                    q_seq = q.reshape(batch, sequence_length, -1)
                    k_seq = k.reshape(batch, sequence_length, -1)
                    v_seq = v.reshape(batch, sequence_length, -1)
                    gate_seq = raw_gate.reshape(batch, sequence_length, -1)
                    beta_seq = raw_beta.reshape(batch, sequence_length, -1)
                    one_token_cu = torch.arange(
                        batch + 1, device=cu_seqlens.device, dtype=cu_seqlens.dtype
                    )
                    output_steps = []
                    for step in range(sequence_length):
                        reserve_base = torch.div(
                            sequence_lengths_plus_one - 2,
                            page_size,
                            rounding_mode="floor",
                        ).to(torch.long)
                        reserve_col = reserve_base + step
                        logical_col = torch.div(
                            sequence_lengths_plus_one + step - 2,
                            page_size,
                            rounding_mode="floor",
                        ).to(torch.long)
                        batch_idx = torch.arange(batch, device=block_map.device)
                        dest_ids = block_map[batch_idx, reserve_col].to(torch.long)
                        if step > 0:
                            src_ids = block_map[
                                batch_idx, reserve_col - 1
                            ].to(torch.long)
                            ssm_cache[dest_ids] = ssm_cache[src_ids]
                        step_block_map = block_map.clone()
                        step_block_map[batch_idx, logical_col] = dest_ids.to(
                            step_block_map.dtype
                        )
                        step_output = self._paged_decode_core(
                            q_seq[:, step, :].reshape(batch, -1),
                            k_seq[:, step, :].reshape(batch, -1),
                            v_seq[:, step, :].reshape(batch, -1),
                            gate_seq[:, step, :].reshape(batch, -1),
                            beta_seq[:, step, :].reshape(batch, -1),
                            one_token_cu,
                            ssm_cache,
                            step_block_map,
                            sequence_lengths_plus_one + step,
                            page_size,
                        )
                        output_steps.append(step_output.squeeze(0))
                    output = torch.stack(output_steps, dim=1).reshape(
                        1, token_count, self.local_heads, self.head_dim
                    )
                else:
                    output = self._paged_decode_core(
                        q,
                        k,
                        v,
                        raw_gate,
                        raw_beta,
                        cu_seqlens,
                        ssm_cache,
                        block_map,
                        sequence_lengths_plus_one,
                        page_size,
                    )
                # Both physical state pools were updated in place.  The model
                # caller ignores this auxiliary return when LayerKVCache owns
                # state, so avoid gathering it back into canonical tensors.
                final_state = None
            else:
                output, recurrent_final = self._kda_core(
                    q.reshape(head_shape),
                    k.reshape(head_shape),
                    v.reshape(head_shape),
                    raw_gate.reshape(head_shape),
                    raw_beta.reshape(1, token_count, self.local_heads),
                    self.weights[W.linear_attn_alog],
                    self.weights[W.linear_attn_dt_b_kda],
                    recurrent_state,
                    mode=mode,
                    cu_seqlens=cu_seqlens,
                )
                assert (
                    q_final is not None and k_final is not None and v_final is not None
                )
                final_state = KimiKDAState(
                    q_conv_state=q_final,
                    k_conv_state=k_final,
                    v_conv_state=v_final,
                    recurrent_state=recurrent_final,
                )
            if trace_enabled:
                record_accuracy_tensor(f"{self.trace_prefix}.q_conv", q, token_dim=0)
                record_accuracy_tensor(f"{self.trace_prefix}.k_conv", k, token_dim=0)
                record_accuracy_tensor(f"{self.trace_prefix}.v_conv", v, token_dim=0)
        if trace_enabled:
            assert final_state is not None
            record_accuracy_tensor(
                f"{self.trace_prefix}.core_output", output, token_dim=1
            )
            record_accuracy_tensor(
                f"{self.trace_prefix}.state.q_conv", final_state.q_conv_state
            )
            record_accuracy_tensor(
                f"{self.trace_prefix}.state.k_conv", final_state.k_conv_state
            )
            record_accuracy_tensor(
                f"{self.trace_prefix}.state.v_conv", final_state.v_conv_state
            )
            record_accuracy_tensor(
                f"{self.trace_prefix}.state.recurrent",
                final_state.recurrent_state,
            )
        if a2a_prefill:
            assert output_gate is not None
            if trace_enabled:
                record_accuracy_tensor(
                    f"{self.trace_prefix}.output_gate",
                    output_gate.squeeze(0),
                    token_dim=0,
                )
            post_send = output.reshape(
                token_count,
                self.local_heads,
                self.head_dim,
            )
            if not post_send.is_contiguous():
                raise ValueError(
                    "K3 KDA A2A core output must be contiguous before exchange"
                )
            post_recv = self._a2a_buffer(
                "post_recv",
                tuple(post_send.shape),
                post_send,
            )
            with _perf_profile(
                f"{self.trace_prefix}.a2a_post.exchange_TP_heads_to_SP_tokens",
                post_send,
            ):
                all_to_all_single(post_send, Group.TP, output=post_recv)
            unpacked_output = self._a2a_buffer(
                "post_unpacked_gated",
                (
                    1,
                    local_token_count,
                    self.total_heads,
                    self.head_dim,
                ),
                output_gate,
            )
            with _perf_profile(
                f"{self.trace_prefix}."
                "a2a_post.fused_source_head_unpack_rmsnorm_sigmoid_gate",
                post_recv,
            ):
                output = kimi_k3_a2a_unpack_rms_norm_sigmoid_gate(
                    post_recv.reshape(
                        self.attn_tp_size,
                        local_token_count,
                        self.local_heads,
                        self.head_dim,
                    ),
                    output_gate,
                    self.weights[W.linear_attn_norm_w],
                    self.eps,
                    output=unpacked_output,
                )
            if trace_enabled:
                record_accuracy_tensor(
                    f"{self.trace_prefix}.gated_output",
                    output,
                    token_dim=1,
                )
            with _perf_profile(
                f"{self.trace_prefix}.a2a_post.replicated_o_projection_SP_tokens",
                output,
            ):
                output = _linear(
                    output.reshape(local_token_count, self.full_projection_size),
                    self._a2a_o_weight,
                )
        else:
            if trace_enabled:
                record_accuracy_tensor(
                    f"{self.trace_prefix}.output_gate",
                    output_gate.squeeze(0),
                    token_dim=0,
                )
            with _perf_profile(
                f"{self.trace_prefix}.rmsnorm_sigmoid_output_gate", output
            ):
                if bool(getattr(attention_inputs, "is_target_verify", False)):
                    # Target verification has only a handful of rows and can
                    # start from a non-zero recurrent state.  Keep this narrow
                    # path in explicit FP32 arithmetic: the production Triton
                    # fusion is tuned for regular prefill/decode shapes and
                    # can produce NaNs for these short speculative batches.
                    output_float = output.float()
                    output = (
                        output_float
                        * torch.rsqrt(
                            output_float.square().mean(dim=-1, keepdim=True)
                            + self.eps
                        )
                        * self.weights[W.linear_attn_norm_w].float()
                        * torch.sigmoid(output_gate.float())
                    ).to(dtype=output.dtype)
                else:
                    output = kimi_kda_rms_norm_sigmoid_gate(
                        output,
                        output_gate,
                        self.weights[W.linear_attn_norm_w],
                        self.eps,
                    )
            if trace_enabled:
                record_accuracy_tensor(
                    f"{self.trace_prefix}.gated_output", output, token_dim=1
                )
            with _perf_profile(
                f"{self.trace_prefix}.o_projection_then_token_reduce_scatter",
                output,
            ):
                projection_input = output.reshape(
                    token_count, self.projection_size
                )
                if bool(getattr(attention_inputs, "is_target_verify", False)):
                    # Match qwen3-next's verify projection boundary.  The
                    # generic K3 row-parallel helper requests an FP32-output
                    # CUDA GEMM before the collective; on tiny speculative
                    # batches that path has produced non-finite partials on
                    # SM100 even though the recurrent state and gated output
                    # are finite.  Verify is not sequence-parallel, so a BF16
                    # local projection followed by the ordinary TP reduction
                    # is sufficient and numerically stable.
                    output = _linear(
                        projection_input,
                        self.weights[W.linear_attn_out_w],
                    )
                    if self.parallelism_config.get_attn_tp_size() > 1:
                        output = all_reduce(output, group=Group.TP)
                else:
                    output = _row_parallel_linear(
                        projection_input,
                        self.weights[W.linear_attn_out_w],
                        self.parallelism_config.get_attn_tp_size(),
                        reduce_scatter_tokens=(
                            sequence_parallel
                            and self.attn_tp_size > 1
                            and hidden_states.is_cuda
                        ),
                        pad_reduce_scatter_tokens=(
                            sequence_parallel
                            and self.attn_tp_size > 1
                            and hidden_states.is_cuda
                            and (
                                mode == "decode"
                                or token_count % self.attn_tp_size != 0
                            )
                        ),
                        use_input_dtype_reduce_scatter=(mode == "prefill"),
                    )
        if trace_enabled:
            record_accuracy_tensor(f"{self.trace_prefix}.output", output, token_dim=0)
        if (
            kv_cache is not None
            and not stored_page_states
            and paged_decode_cache is None
        ):
            assert final_state is not None
            self.cache_adapter.store(
                final_state,
                kv_cache,
                attention_inputs,
                cu_seqlens,
                mode=mode,
            )
        return output, final_state


class KimiK3MLA(MlaAttention):
    """K3 NoPE MLA over RTP's packed-token and compressed-cache layouts.

    The serving path reuses the framework MLA kernels and executes one packed
    Q-A/KV-A/output-gate projection. Accuracy tracing keeps the source model's
    independent projection boundaries so it can record comparable tensors.
    """

    def __init__(
        self,
        config: KimiK3ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int = -1,
    ) -> None:
        super().__init__(
            config.attn_config,
            parallelism_config,
            weights,
            layer_idx,
            KIMI_K3_MLA_LATENT_NORM_EPS,
            config.quant_config,
        )
        self.config = config
        self.weights = weights
        self.trace_prefix = f"layer.{layer_idx}.mla" if layer_idx >= 0 else "mla"
        self._perf_profile_prefix = None
        self._perf_accepts_strided_latent = _perf_fusions_enabled()
        tp_size = int(parallelism_config.get_attn_tp_size())
        self.attn_tp_size = tp_size
        total_heads = int(config.attn_config.head_num)
        if total_heads % tp_size:
            raise ValueError(
                f"MLA heads {total_heads} must be divisible by attention TP {tp_size}"
            )
        self.local_heads = total_heads // tp_size
        self.attn_tp_rank = int(parallelism_config.get_attn_tp_rank())
        self.q_lora_rank = int(config.attn_config.q_lora_rank)
        self.kv_lora_rank = int(config.attn_config.kv_lora_rank)
        self.nope_dim = int(config.attn_config.nope_head_dim)
        self.suffix_dim = int(config.attn_config.rope_head_dim)
        self.value_dim = int(config.attn_config.v_head_dim)
        self.q_head_dim = self.nope_dim + self.suffix_dim
        self.softmax_scale = self.q_head_dim**-0.5
        # The source K3 MLA constructs q_a/kv_a KimiRMSNorm without passing
        # config.rms_norm_eps, so both intentionally use the module default
        # 1e-6.  Other decoder norms continue to use config.rms_norm_eps
        # (1e-5 for the real checkpoint).
        self.eps = KIMI_K3_MLA_LATENT_NORM_EPS
        runtime = config.k3_runtime_config
        if not runtime.mla_use_nope:
            raise ValueError(
                "Kimi K3 requires the physical MLA suffix to remain no-RoPE"
            )
        self.use_output_gate = runtime.mla_use_output_gate
        # Prefill 走 FlashMLA(dense prefill),Decode 走 FlashInfer("kernel")。
        # 与 KDA 一样按 PD 角色定,不再由 KIMI_K3_MLA_BACKEND 传入。
        self._mla_backend = (
            "flashmla" if _is_prefill_role(parallelism_config) else "kernel"
        )

        self._q_a_norm = weights[W.mla_q_a_ln_gamma]
        self._q_b_w = weights[W.mla_q_b_w]
        self._kv_a_norm = weights[W.mla_kv_a_ln_gamma]
        self._kv_b_w = weights[W.mla_kv_b_w]
        self._o_w = weights[W.attn_o_w]
        self._packed_qkv_gate_w = weights[W.mla_fusedqkrope_w]
        # These are only the two small MLA latent norms; decoder-wide norms keep
        # the framework kernel.
        self.q_a_layernorm = _KimiK3MLALatentRMSNorm(self._q_a_norm)
        self.kv_a_layernorm = _KimiK3MLALatentRMSNorm(self._kv_a_norm)
        self._sp_active_for_forward = False
        self._sp_padded_for_forward = False
        self._sp_prefill_input_is_sharded = False
        self._sp_prefill_layout_for_forward: Optional[_TokenShardLayout] = None

    def _use_source_projection_boundaries(self) -> bool:
        return (
            not _perf_fusions_enabled()
            # Only accuracy tracing needs the source implementation's
            # independent projection boundaries.
            or _accuracy_trace_requested()
        )

    def _project_source_qkv_a_input(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        q_end = self.q_lora_rank
        kv_end = q_end + self.kv_lora_rank + self.suffix_dim
        q_a = _linear(hidden_states, self._packed_qkv_gate_w[:, :q_end])
        kv_a = _linear(hidden_states, self._packed_qkv_gate_w[:, q_end:kv_end])
        output_gate = None
        if self.use_output_gate:
            output_gate = _linear(hidden_states, self._packed_qkv_gate_w[:, kv_end:])
        return torch.cat((q_a, kv_a), dim=-1), output_gate

    def _project_qkv_a_input(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        source_boundaries = self._use_source_projection_boundaries()
        prefill_layout = getattr(self, "_sp_prefill_layout_for_forward", None)
        if self._sp_prefill_input_is_sharded:
            if source_boundaries:
                hidden_states = (
                    _prefill_all_gather_input(hidden_states, self.attn_tp_size)
                    if prefill_layout is None
                    else _prefill_all_gather_input(
                        hidden_states,
                        self.attn_tp_size,
                        prefill_layout.logical_tokens,
                    )
                )
            else:
                logical_tokens = (
                    hidden_states.shape[0] * self.attn_tp_size
                    if prefill_layout is None
                    else prefill_layout.logical_tokens
                )
                packed = _prefill_all_gather_matmul(
                    hidden_states,
                    self._packed_qkv_gate_w,
                    tp_size=self.attn_tp_size,
                    logical_tokens=logical_tokens,
                )
                return torch.split(
                    packed,
                    [
                        self.q_lora_rank + self.kv_lora_rank + self.suffix_dim,
                        self.local_heads * self.value_dim,
                    ],
                    dim=-1,
                )
        if source_boundaries:
            return self._project_source_qkv_a_input(hidden_states)
        packed = self.fused_qkv_a_proj(hidden_states)
        return torch.split(
            packed,
            [
                self.q_lora_rank + self.kv_lora_rank + self.suffix_dim,
                self.local_heads * self.value_dim,
            ],
            dim=-1,
        )

    def _apply_output_gate(
        self,
        attn_output: torch.Tensor,
        output_gate: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """K3 sigmoid output gate, applied on the framework (kernel) path.

        ``attn_output`` is the framework context flattened to
        ``[tokens, local_heads * v_head_dim]`` (head-major), matching the flat
        layout of the rank-local gate projection, so the gate multiplies element
        wise per (head, value) exactly as K3 requires before o_proj.
        This runs before o_proj's TP all_reduce, so each rank gates only its
        local heads.
        """
        if not self.use_output_gate:
            return attn_output
        assert output_gate is not None
        return attn_output * torch.sigmoid(output_gate.reshape_as(attn_output))

    def _project_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        if self._sp_active_for_forward:
            tp_size = self.parallelism_config.get_attn_tp_size()
            return _row_parallel_linear(
                attn_output,
                self._o_w,
                tp_size,
                reduce_scatter_tokens=True,
                pad_reduce_scatter_tokens=(
                    self._sp_padded_for_forward
                    or (
                        self._sp_prefill_input_is_sharded
                        and attn_output.shape[0] % tp_size != 0
                    )
                ),
                use_input_dtype_reduce_scatter=(self._sp_prefill_input_is_sharded),
            )
        return super()._project_output(attn_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        sequence_parallel: bool = False,
        prefill_sp_layout: Optional[_TokenShardLayout] = None,
    ) -> torch.Tensor:
        attn_inputs = _select_mla_attention_inputs(attention_inputs, fmha_impl)
        self._sp_active_for_forward = bool(
            sequence_parallel
            and self.parallelism_config.get_attn_tp_size() > 1
            and hidden_states.is_cuda
            and attn_inputs is not None
        )
        self._sp_prefill_input_is_sharded = prefill_sp_layout is not None
        self._sp_prefill_layout_for_forward = prefill_sp_layout
        if prefill_sp_layout is not None and (
            not self._sp_active_for_forward
            or attn_inputs is None
            or not attn_inputs.is_prefill
        ):
            raise ValueError(
                "prefill_sp_layout requires production CUDA MLA Prefill "
                "Sequence Parallel with TP>1"
            )
        self._sp_padded_for_forward = bool(
            self._sp_active_for_forward
            and attn_inputs is not None
            and not attn_inputs.is_prefill
        )
        try:
            return self._forward_impl(
                hidden_states,
                fmha_impl,
                kv_cache,
                attention_inputs=attention_inputs,
            )
        finally:
            self._sp_active_for_forward = False
            self._sp_padded_for_forward = False
            self._sp_prefill_input_is_sharded = False
            self._sp_prefill_layout_for_forward = None

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
    ) -> torch.Tensor:
        if not hidden_states.is_cuda:
            raise RuntimeError("Kimi K3 MLA requires CUDA")
        if accuracy_trace_mode() is None:
            return super().forward(hidden_states, fmha_impl, kv_cache)

        # Trace the real MLA kernel/cache path.  The trace stays O(T) and
        # recomputes only the final score/probability row, so it never changes
        # the production math it is observing.
        fused_qkv, output_gate = self._project_qkv_a_input(hidden_states)
        input_shape = fused_qkv.shape[:-1]
        q, compressed = torch.split(
            fused_qkv,
            [
                self.q_lora_rank,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ],
            dim=-1,
        )
        query_latent = self.q_a_layernorm(q.contiguous())
        query_projection = self.q_b_proj(query_latent)
        query = query_projection.reshape(-1, self.num_heads, self.q_head_dim)
        compressed_kv, key_suffix = torch.split(
            compressed,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv.contiguous())
        canonical_current = torch.cat((compressed_kv, key_suffix), dim=-1)
        record_accuracy_tensor(
            f"{self.trace_prefix}.query_latent",
            query_latent,
            token_dim=0,
        )
        record_accuracy_tensor(f"{self.trace_prefix}.query", query, token_dim=0)
        record_accuracy_tensor(
            f"{self.trace_prefix}.compressed_current",
            canonical_current,
            token_dim=0,
        )

        topk_indices = self._run_sparse_indexer(
            hidden_states, query_latent, query, kv_cache, fmha_impl
        )
        context = fmha_impl.forward(
            query,
            compressed_kv,
            key_suffix,
            kv_cache,
            self.layer_idx,
            topk_indices,
        )
        if context is not None:
            context = context.reshape(*input_shape, -1).contiguous()
        else:
            context = torch.zeros(
                (*input_shape, self.num_heads * self.v_head_dim),
                dtype=fused_qkv.dtype,
                device=fused_qkv.device,
            )
        # Accuracy requests contain one packed sequence.  Recompute only its
        # final score row: O(T) storage instead of the O(T^2) full matrix.
        # Use the same group-selected attention-input view that produced this
        # layer's slot mapping.  Reading the wrapper-captured view here makes
        # canonical/trace math reuse group zero's cache for every later MLA
        # group even though concat_and_cache_mla wrote the correct slot.
        attn_inputs = _select_mla_attention_inputs(attention_inputs, fmha_impl)
        if attn_inputs is None:
            raise ValueError("MLA kernel trace requires PyAttentionInputs")
        cu_seqlens = self._trace_cu_seqlens(
            attn_inputs,
            fused_qkv.shape[0],
            bool(attn_inputs.is_prefill),
            fused_qkv.device,
        )
        ranges = _sequence_offsets(cu_seqlens, fused_qkv.shape[0])
        if len(ranges) != 1:
            raise RuntimeError(
                "K3 MLA accuracy trace currently requires one packed sequence"
            )
        cache_input, canonical_cache = self._trace_cache_snapshot(
            kv_cache,
            attn_inputs,
            current_token_count=ranges[0][1] - ranges[0][0],
            fallback_current=canonical_current,
        )
        if cache_input.numel():
            record_accuracy_tensor(f"{self.trace_prefix}.cache_input", cache_input)
        record_accuracy_tensor(f"{self.trace_prefix}.cache", canonical_cache)
        all_compressed_kv, all_key_suffix = torch.split(
            canonical_cache,
            [self.kv_lora_rank, self.suffix_dim],
            dim=-1,
        )
        expanded_input = all_compressed_kv.to(dtype=self._kv_b_w.dtype)
        expanded_projection = _linear(expanded_input, self._kv_b_w)
        expanded = expanded_projection.reshape(
            all_compressed_kv.shape[0],
            self.local_heads,
            self.nope_dim + self.value_dim,
        )
        key_nope, value = torch.split(expanded, [self.nope_dim, self.value_dim], dim=-1)
        key = torch.cat(
            (
                key_nope,
                all_key_suffix.unsqueeze(1).expand(-1, self.local_heads, -1),
            ),
            dim=-1,
        )
        context_by_head = context.reshape(
            fused_qkv.shape[0], self.num_heads, self.v_head_dim
        )
        scores = torch.einsum("thd,shd->hts", query[-1:], key) * self.softmax_scale
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
            dtype=query.dtype
        )
        record_accuracy_tensor(
            f"{self.trace_prefix}.context_last_query",
            context_by_head[-1:],
            token_dim=0,
        )
        record_accuracy_tensor(f"{self.trace_prefix}.scores_last_query", scores)
        record_accuracy_tensor(
            f"{self.trace_prefix}.probabilities_last_query", probabilities
        )

        if self.use_output_gate:
            assert output_gate is not None
            output_gate = output_gate.reshape_as(context)
            record_accuracy_tensor(
                f"{self.trace_prefix}.output_gate",
                output_gate.reshape(
                    fused_qkv.shape[0],
                    self.num_heads,
                    self.v_head_dim,
                ),
                token_dim=0,
            )
            context = context * torch.sigmoid(output_gate)
        output = self._project_output(context)
        record_accuracy_tensor(f"{self.trace_prefix}.output", output, token_dim=0)
        return output

    def _trace_cache_snapshot(
        self,
        kv_cache: Optional[LayerKVCache],
        attention_inputs: PyAttentionInputs,
        *,
        current_token_count: int,
        fallback_current: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rebuild canonical MLA history from the cache after kernel writeback.

        The native FMHA path stores normalized latent + no-RoPE suffix in the
        paged cache.  Recording only ``fallback_current`` proves the projection
        but cannot validate PD transfer.  Accuracy tracing therefore follows
        the group-specific kernel block table and snapshots the complete
        prefix plus current tokens after ``fmha_impl.forward`` has written it.
        """

        empty = fallback_current.new_empty(0, self.kv_lora_rank + self.suffix_dim)
        if kv_cache is None or KimiK3LinearCacheAdapter._is_fake_stream(
            attention_inputs
        ):
            return empty, fallback_current
        if current_token_count < 0:
            raise ValueError("MLA trace current token count cannot be negative")

        length_tensor = (
            attention_inputs.prefix_lengths
            if bool(attention_inputs.is_prefill)
            else attention_inputs.sequence_lengths
        )
        past_lengths = (
            [0]
            if length_tensor is None or length_tensor.numel() == 0
            else [int(value) for value in length_tensor.detach().cpu().tolist()]
        )
        if len(past_lengths) != 1:
            raise RuntimeError("K3 MLA accuracy cache trace requires one sequence")
        past_length = past_lengths[0]
        total_length = past_length + current_token_count
        if total_length == 0:
            return empty, empty

        page_size = int(kv_cache.seq_size_per_block)
        if page_size <= 0:
            raise ValueError("MLA cache seq_size_per_block must be positive")
        width = self.kv_lora_rank + self.suffix_dim
        base = kv_cache.kv_cache_base
        if base is None or base.numel() == 0:
            raise ValueError("MLA LayerKVCache has no backing tensor")
        if base.ndim == 3 and base.shape[-1] == width:
            cache_view = base
        else:
            if base.numel() % (base.shape[0] * width):
                raise ValueError("MLA cache storage is not divisible by latent width")
            cache_view = base.reshape(base.shape[0], -1, width)
        if cache_view.shape[1] < page_size:
            raise ValueError("MLA cache block is smaller than seq_size_per_block")

        block_map = KimiK3LinearCacheAdapter._block_map(attention_inputs)
        cached = torch.stack(
            [
                cache_view[
                    KimiK3LinearCacheAdapter._block_id(
                        block_map, 0, position, page_size
                    ),
                    position % page_size,
                ]
                for position in range(total_length)
            ]
        )
        return cached[:past_length], cached

    @staticmethod
    def _trace_cu_seqlens(
        attention_inputs: PyAttentionInputs,
        token_count: int,
        is_prefill: bool,
        device: torch.device,
    ) -> torch.Tensor:
        cu_seqlens = (
            attention_inputs.cu_seqlens
            if is_prefill
            else attention_inputs.decode_cu_seqlens_d
        )
        if cu_seqlens is None or cu_seqlens.numel() == 0:
            cu_seqlens = (
                torch.tensor([0, token_count], dtype=torch.int32, device=device)
                if is_prefill
                else torch.arange(token_count + 1, dtype=torch.int32, device=device)
            )
        return cu_seqlens

class KimiK3DenseMLP(nn.Module):
    def __init__(
        self,
        config: KimiK3ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int = -1,
    ) -> None:
        super().__init__()
        self.parallelism_config = parallelism_config
        self.weights = weights
        self.trace_prefix = f"layer.{layer_idx}.dense" if layer_idx >= 0 else "dense"
        self.ffn_tp_size = int(parallelism_config.get_ffn_tp_size())
        self.ffn_tp_rank = int(parallelism_config.get_ffn_tp_rank())
        self._full_column_weights: dict[str, torch.Tensor] = {}
        self._full_row_weights: dict[str, torch.Tensor] = {}
        runtime = config.k3_runtime_config
        self.beta = runtime.activation_situ_beta
        self.linear_beta = runtime.activation_situ_linear_beta

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        sequence_parallel: bool = False,
        valid_token_count: Optional[int] = None,
    ) -> torch.Tensor:
        sp_active = sequence_parallel and self.ffn_tp_size > 1 and hidden_states.is_cuda
        trace_enabled = _accuracy_trace_enabled()
        gate_weight = _sequence_parallel_column_weight(
            self.weights,
            K3W.DENSE_GATE,
            self.ffn_tp_size,
            self.ffn_tp_rank,
            self._full_column_weights,
            "sp_gate",
            sequence_parallel=sp_active,
        )
        up_weight = _sequence_parallel_column_weight(
            self.weights,
            K3W.DENSE_UP,
            self.ffn_tp_size,
            self.ffn_tp_rank,
            self._full_column_weights,
            "sp_up",
            sequence_parallel=sp_active,
        )
        # Sequence parallel feeds the all-gathered full-width weight here, so
        # the GEMM itself is the same call in both cases; only the label
        # distinguishes them in a profile.
        with _perf_profile(
            f"{self.trace_prefix}."
            + ("replicated_gate_and_up_gemm" if sp_active else "gate_and_up_gemm"),
            hidden_states,
        ):
            gate = _linear(hidden_states, gate_weight)
            up = _linear(hidden_states, up_weight)
        if trace_enabled:
            record_accuracy_tensor(f"{self.trace_prefix}.gate", gate, token_dim=0)
            record_accuracy_tensor(f"{self.trace_prefix}.up", up, token_dim=0)
        with _perf_profile(f"{self.trace_prefix}.situ_activation", gate):
            activated = _situ(
                gate,
                up,
                self.beta,
                self.linear_beta,
            )
        if trace_enabled:
            record_accuracy_tensor(
                f"{self.trace_prefix}.activation", activated, token_dim=0
            )
        with _perf_profile(
            f"{self.trace_prefix}.replicated_down_gemm_no_collective", activated
        ):
            down_weight = _sequence_parallel_row_weight(
                self.weights,
                K3W.DENSE_DOWN,
                self.ffn_tp_size,
                self.ffn_tp_rank,
                self._full_row_weights,
                "sp_down",
                sequence_parallel=sp_active,
            )
            output = (
                _linear(activated, down_weight)
                if sp_active
                else _row_parallel_linear(
                    activated,
                    down_weight,
                    self.parallelism_config.get_ffn_tp_size(),
                )
            )
        if trace_enabled:
            record_accuracy_tensor(f"{self.trace_prefix}.output", output, token_dim=0)
        return output


class KimiK3LatentMoE(nn.Module):
    """Correctness path for K3 latent MoE with packed MXFP4 experts.

    ``ep_size == 1`` evaluates the selected experts with transparent Torch
    operators.  ``ep_size > 1`` uses RTP's initialized DeepEP buffer: normal
    mode dispatches one token copy per destination rank, while low-latency
    mode dispatches one token copy per selected expert.  Both paths combine
    the *latent* expert sum before K3's routed RMSNorm/up projection; moving
    those nonlinear operations onto the destination rank would be wrong.

    K3's latent width (3584 in the real checkpoint) is not one of the widths
    currently admitted by RTP's low-latency router.  This first correctness
    implementation pads the communication payload to ``hidden_size`` (7168),
    then slices it back after combine.  A native K3 DeepEP/MXFP4 executor can
    remove that bandwidth overhead without changing the model semantics.
    """

    def __init__(
        self,
        config: KimiK3ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int = -1,
    ) -> None:
        super().__init__()
        self.parallelism_config = parallelism_config
        self.weights = weights
        self.trace_prefix = f"layer.{layer_idx}.moe" if layer_idx >= 0 else "moe"
        self.expert_num = int(config.expert_num)
        self.top_k = int(config.moe_k)
        self.renormalize = bool(config.has_moe_norm)
        self.routed_scaling_factor = float(config.routed_scaling_factor)
        self.num_expert_group = int(config.moe_n_group)
        self.topk_group = int(config.moe_topk_group)
        self.ep_size = int(parallelism_config.ep_size)
        self.ep_rank = int(parallelism_config.ep_rank)
        if self.expert_num % self.ep_size:
            raise ValueError(
                f"expert count {self.expert_num} must divide EP size {self.ep_size}"
            )
        self.local_expert_count = self.expert_num // self.ep_size
        self.local_expert_start = self.ep_rank * self.local_expert_count
        self.attn_tp_size = int(parallelism_config.get_attn_tp_size())
        self.attn_tp_rank = int(parallelism_config.get_attn_tp_rank())
        self.ffn_tp_size = int(parallelism_config.get_ffn_tp_size())
        self.ffn_tp_rank = int(parallelism_config.get_ffn_tp_rank())
        self._full_column_weights: dict[str, torch.Tensor] = {}
        self._full_row_weights: dict[str, torch.Tensor] = {}
        runtime = config.k3_runtime_config
        self.latent_moe_use_norm = runtime.latent_moe_use_norm
        self.beta = runtime.activation_situ_beta
        self.linear_beta = runtime.activation_situ_linear_beta
        self.eps = float(config.layernorm_eps)
        self.latent_size = int(self.weights[K3W.MOE_ROUTED_DOWN].shape[1])
        self.dispatch_hidden_size = int(config.hidden_size)
        if self.dispatch_hidden_size < self.latent_size:
            raise ValueError(
                "Kimi K3 DeepEP communication width cannot be smaller than "
                f"the routed latent width: {self.dispatch_hidden_size} < "
                f"{self.latent_size}"
            )
        self.layer_idx = int(layer_idx)
        # The gate is used only by this module. Replace the checkpoint BF16
        # tensor once so Decode does not materialize 92 FP32 copies per step.
        self.weights[K3W.MOE_GATE] = self.weights[K3W.MOE_GATE].float()
        self.weights[K3W.MOE_CORRECTION_BIAS] = self.weights[
            K3W.MOE_CORRECTION_BIAS
        ].float()
        self._group_topk = GroupTopK()
        # Kill switch 恒定,不必在每个 Decode step 重解析环境变量;trace 模式仍
        # 在 _route 里按请求判断,不能缓存到这里。
        self._fused_router_enabled = _env_flag("KIMI_K3_FUSED_ROUTER", True)
        # K3 的 MoE 只有 DeepGEMM mega 一条生产路径。DeepEP 的 Torch 专家循环把
        # 选中的专家反量化成 BF16,93 层 Decode 首次使用即耗尽显存,所以那条分支
        # 连同它的开关一并删掉 —— 不是"默认走 mega",而是只有 mega。
        self._setup_deep_gemm_mega()

    @staticmethod
    def _packed_fp4_view(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype == torch.int8:
            return tensor
        if tensor.dtype == torch.uint8:
            return tensor.view(torch.int8)
        raise TypeError(
            "K3 DeepGEMM packed expert weight must be uint8/int8, got "
            f"{tensor.dtype}"
        )

    @staticmethod
    def _ue8m0_view(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype == torch.float8_e8m0fnu:
            return tensor
        if tensor.dtype == torch.uint8:
            return tensor.view(torch.float8_e8m0fnu)
        raise TypeError(
            "K3 DeepGEMM expert scale must be uint8/float8_e8m0fnu, got "
            f"{tensor.dtype}"
        )

    def _setup_deep_gemm_mega(self) -> None:
        """Transform K3's EP-local MXFP4 weights for SiTU MegaMoE."""

        import deep_gemm
        import torch.distributed as dist

        from rtp_llm.models_py.modules.dsv4.moe.input_packer import (
            get_mega_moe_input_packer,
        )
        from rtp_llm.models_py.modules.dsv4.moe.mega_buf import (
            _get_or_create_mega_buf,
            _get_or_create_mega_output,
        )
        from rtp_llm.models_py.modules.dsv4.quant_layouts import (
            FP4_BLOCK,
            prepare_fp4_weight_scale_for_deepgemm,
        )

        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
            raise RuntimeError("K3 DeepGEMM MegaMoE requires an SM100+ CUDA GPU")
        if not dist.is_initialized():
            raise RuntimeError(
                "K3 DeepGEMM MegaMoE requires torch.distributed initialization"
            )
        world_size = int(dist.get_world_size())
        if (
            self.ep_size != 8
            or self.attn_tp_size != 8
            or world_size != 8
            or self.local_expert_count != 112
        ):
            raise RuntimeError(
                "K3 DeepGEMM MegaMoE is fixed to "
                "TP8/EP8/world8/112-local-experts; got "
                f"TP={self.attn_tp_size} EP={self.ep_size} world={world_size} "
                f"local_experts={self.local_expert_count}"
            )

        mega_signature = inspect.signature(deep_gemm.fp8_fp4_mega_moe)
        required_parameters = {
            "activation_beta",
            "activation_linear_beta",
            "fast_math",
        }
        missing_parameters = required_parameters.difference(mega_signature.parameters)
        if missing_parameters:
            raise RuntimeError(
                "Kimi K3 DeepGEMM mega resolved an old DeepGEMM "
                "without K3 SiTU support; missing parameters: "
                + ", ".join(sorted(missing_parameters))
            )
        # 守卫:MegaMoE 依赖 DeepGEMM 的一组特定 API,机器上常有多份 DeepGEMM,
        # 解析到旧的那份表现不是报错而是数值不对。原先靠 KIMI_K3_DEEPGEMM_EXPECTED_PATH
        # 钉住路径,而 launcher 里它就是照抄 KIMI_K3_OPERATOR_PYTHONPATH —— 直接读源头。
        expected_root = os.environ.get("KIMI_K3_OPERATOR_PYTHONPATH")
        deep_gemm_path = os.path.realpath(getattr(deep_gemm, "__file__", ""))
        if expected_root and not deep_gemm_path.startswith(
            os.path.realpath(expected_root) + os.sep
        ):
            raise RuntimeError(
                "K3 DeepGEMM loaded from an unexpected path: "
                f"{deep_gemm_path}; expected under {expected_root}"
            )

        st_w1_w = self.weights.pop(K3W.MOE_W1_PACKED)
        st_w1_s = self.weights.pop(K3W.MOE_W1_SCALE)
        st_w3_w = self.weights.pop(K3W.MOE_W3_PACKED)
        st_w3_s = self.weights.pop(K3W.MOE_W3_SCALE)
        if st_w1_w.ndim != 3 or st_w1_w.shape[0] != self.local_expert_count:
            raise RuntimeError(
                "unexpected K3 W1 stack for MegaMoE: " f"{tuple(st_w1_w.shape)}"
            )
        device = st_w1_w.device
        intermediate = int(st_w1_w.shape[1])
        if (
            self.latent_size != 3584
            or intermediate != 3072
            or int(st_w1_w.shape[2]) * 2 != self.latent_size
        ):
            raise RuntimeError(
                "K3 DeepGEMM MegaMoE expects latent/intermediate=3584/3072; "
                f"got {self.latent_size}/{intermediate}"
            )

        expert_count = self.local_expert_count
        w13 = torch.empty(
            (expert_count, 2 * intermediate, self.latent_size // 2),
            dtype=torch.int8,
            device=device,
        )
        s13_raw = torch.empty(
            (expert_count, 2 * intermediate, self.latent_size // FP4_BLOCK),
            dtype=torch.float8_e8m0fnu,
            device=device,
        )
        w13[:, :intermediate].copy_(self._packed_fp4_view(st_w1_w))
        w13[:, intermediate:].copy_(self._packed_fp4_view(st_w3_w))
        s13_raw[:, :intermediate].copy_(self._ue8m0_view(st_w1_s))
        s13_raw[:, intermediate:].copy_(self._ue8m0_view(st_w3_s))
        del st_w1_w, st_w1_s, st_w3_w, st_w3_s
        s13 = prepare_fp4_weight_scale_for_deepgemm(
            s13_raw,
            2 * intermediate,
            self.latent_size,
            expert_count,
        )
        del s13_raw
        torch.cuda.empty_cache()

        st_w2_w = self.weights.pop(K3W.MOE_W2_PACKED)
        st_w2_s = self.weights.pop(K3W.MOE_W2_SCALE)
        expected_w2_shape = (
            expert_count,
            self.latent_size,
            intermediate // 2,
        )
        if tuple(st_w2_w.shape) != expected_w2_shape:
            raise RuntimeError(
                "unexpected K3 W2 stack for MegaMoE: "
                f"{tuple(st_w2_w.shape)} != {expected_w2_shape}"
            )
        w2 = torch.empty(expected_w2_shape, dtype=torch.int8, device=device)
        s2_raw = torch.empty(
            (expert_count, self.latent_size, intermediate // FP4_BLOCK),
            dtype=torch.float8_e8m0fnu,
            device=device,
        )
        w2.copy_(self._packed_fp4_view(st_w2_w))
        s2_raw.copy_(self._ue8m0_view(st_w2_s))
        del st_w2_w, st_w2_s
        s2 = prepare_fp4_weight_scale_for_deepgemm(
            s2_raw,
            self.latent_size,
            intermediate,
            expert_count,
        )
        del s2_raw
        torch.cuda.empty_cache()

        (self._mega_l1_w, self._mega_l1_sf), (
            self._mega_l2_w,
            self._mega_l2_sf,
        ) = deep_gemm.transform_weights_for_mega_moe(
            (w13, s13),
            (w2, s2),
            activation="situ",
        )
        del w13, s13, w2, s2
        torch.cuda.empty_cache()

        max_tokens_per_rank = int(
            os.environ.get("KIMI_K3_MEGA_MAX_TOKENS_PER_RANK", "65536")
        )
        if max_tokens_per_rank <= 0:
            raise ValueError("KIMI_K3_MEGA_MAX_TOKENS_PER_RANK must be positive")
        self._mega_group = dist.group.WORLD
        self._mega_buf = _get_or_create_mega_buf(
            group=self._mega_group,
            num_experts=self.expert_num,
            num_max_tokens_per_rank=max_tokens_per_rank,
            num_topk=self.top_k,
            hidden=self.latent_size,
            intermediate_hidden=intermediate,
            use_fp8_dispatch=True,
            activation="situ",
        )
        output_capacity = max(
            max_tokens_per_rank,
            int(getattr(self._mega_buf, "num_max_tokens_per_rank", 0)),
        )
        self._mega_y = _get_or_create_mega_output(
            output_capacity,
            self.latent_size,
            torch.bfloat16,
            device,
        )
        self._mega_input_packer = get_mega_moe_input_packer()

        device_index = device.index if device.index is not None else 0
        if device_index not in _DEEPGEMM_MEGA_LOGGED_DEVICES:
            logging.info(
                "[KimiK3 DeepGEMM MegaMoE] enabled device=%s module=%s "
                "TP=%d EP=%d experts=%d local_experts=%d topk=%d "
                "latent=%d intermediate=%d max_tokens_per_rank=%d "
                "input_packer=%s",
                device,
                deep_gemm_path,
                self.attn_tp_size,
                self.ep_size,
                self.expert_num,
                self.local_expert_count,
                self.top_k,
                self.latent_size,
                intermediate,
                max_tokens_per_rank,
                self._mega_input_packer.name,
            )
            _DEEPGEMM_MEGA_LOGGED_DEVICES.add(device_index)

    def _deep_gemm_mega_expert_sum(
        self,
        routed_input: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        import deep_gemm

        token_count = int(routed_input.shape[0])
        capacity = int(self._mega_buf.num_max_tokens_per_rank)
        if token_count > capacity:
            raise RuntimeError(
                f"K3 MegaMoE tokens/rank={token_count} exceeds capacity={capacity}"
            )
        with _perf_profile(
            f"{self.trace_prefix}.mega_input_pack_hidden_topk_weights_ids",
            routed_input,
        ):
            self._mega_input_packer.pack(
                routed_input,
                routing_weights,
                expert_ids,
                self._mega_buf,
                token_count,
            )
        output = self._mega_y[:token_count]
        with _perf_profile(
            f"{self.trace_prefix}.deepgemm_mega_fused_a2a_dispatch"
            "_mxfp4_w13_situ_w2_combine",
            routed_input,
        ):
            deep_gemm.fp8_fp4_mega_moe(
                output,
                (self._mega_l1_w, self._mega_l1_sf),
                (self._mega_l2_w, self._mega_l2_sf),
                self._mega_buf,
                recipe=(1, 1, 32),
                activation="situ",
                activation_clamp=None,
                activation_beta=float(self.beta),
                activation_linear_beta=(
                    None if self.linear_beta is None else float(self.linear_beta)
                ),
                fast_math=True,
            )
        return output

    def _route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_weight = self.weights[K3W.MOE_GATE]
        router_logits = _linear(hidden_states.float(), router_weight)
        correction_bias = self.weights[K3W.MOE_CORRECTION_BIAS]
        if (
            self._fused_router_enabled
            and accuracy_trace_mode() is None
            and self._group_topk.fused_sigmoid_supported(
                router_logits,
                correction_bias,
                self.num_expert_group,
                self.topk_group,
                self.top_k,
            )
        ):
            expert_weights = torch.empty(
                (hidden_states.shape[0], self.top_k),
                dtype=torch.float32,
                device=hidden_states.device,
            )
            expert_ids = torch.empty(
                (hidden_states.shape[0], self.top_k),
                dtype=torch.int64,
                device=hidden_states.device,
            )
            self._group_topk.forward_fused_sigmoid(
                expert_weights,
                expert_ids,
                router_logits,
                correction_bias,
                self.top_k,
                self.renormalize,
                self.routed_scaling_factor,
            )
            return expert_ids, expert_weights

        scores = torch.sigmoid(router_logits)
        choice_scores = scores + correction_bias.unsqueeze(0)
        if self.num_expert_group > 1 and self.num_expert_group > self.topk_group:
            grouped = choice_scores.reshape(
                hidden_states.shape[0], self.num_expert_group, -1
            )
            group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
            selected_groups = group_scores.topk(
                self.topk_group, dim=-1, sorted=False
            ).indices
            group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
            group_mask.scatter_(1, selected_groups, True)
            expert_mask = (
                group_mask.unsqueeze(-1).expand_as(grouped).reshape_as(choice_scores)
            )
            choice_scores = choice_scores.masked_fill(~expert_mask, float("-inf"))
        if accuracy_trace_mode() is not None:
            boundary = choice_scores.topk(self.top_k + 1, dim=-1).values
            router_token_dim = None if tensor_dump_full_router() else 0
            record_accuracy_tensor(
                f"{self.trace_prefix}.router_scores", scores, token_dim=0
            )
            record_accuracy_tensor(
                f"{self.trace_prefix}.router_margin",
                boundary[:, self.top_k - 1] - boundary[:, self.top_k],
                token_dim=router_token_dim,
            )
        expert_ids = choice_scores.topk(self.top_k, dim=-1, sorted=False).indices
        expert_weights = scores.gather(1, expert_ids)
        if self.top_k > 1 and self.renormalize:
            expert_weights = expert_weights / (
                expert_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        return expert_ids, expert_weights * self.routed_scaling_factor

    def _expert_weight(
        self,
        packed_name: str,
        scale_name: str,
        local_expert: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        packed = self.weights[packed_name][local_expert]
        scale = self.weights[scale_name][local_expert]
        # The full 93-layer TP1/EP8 correctness model is larger than one
        # B300 when every local expert remains resident.  A default-off loader
        # mode may retain selected layers' checkpoint-native MXFP4 bytes on
        # CPU.  Copy only the routed expert here, before the existing
        # dequantization, so the mathematical path and checkpoint bytes remain
        # unchanged.
        if packed.device != device:
            packed = packed.to(device=device)
        if scale.device != device:
            scale = scale.to(device=device)
        return dequantize_mxfp4(
            packed,
            scale,
            dtype=dtype,
        )

    def _run_local_expert(
        self, expert_input: torch.Tensor, local_expert: int
    ) -> torch.Tensor:
        """Torch MXFP4 fallback for one EP-local expert."""

        w1 = self._expert_weight(
            K3W.MOE_W1_PACKED,
            K3W.MOE_W1_SCALE,
            local_expert,
            expert_input.dtype,
            expert_input.device,
        )
        w3 = self._expert_weight(
            K3W.MOE_W3_PACKED,
            K3W.MOE_W3_SCALE,
            local_expert,
            expert_input.dtype,
            expert_input.device,
        )
        activated = _situ(
            F.linear(expert_input, w1),
            F.linear(expert_input, w3),
            self.beta,
            self.linear_beta,
        )
        w2 = self._expert_weight(
            K3W.MOE_W2_PACKED,
            K3W.MOE_W2_SCALE,
            local_expert,
            expert_input.dtype,
            expert_input.device,
        )
        return F.linear(activated, w2)

    def _local_expert_sum(
        self,
        routed_input: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        *,
        ids_are_local: bool,
    ) -> torch.Tensor:
        """Sum this rank's selected expert contributions in latent space."""

        expert_slots = self._local_expert_slots(
            routed_input,
            expert_ids,
            ids_are_local=ids_are_local,
        )
        return self._reduce_expert_slots(expert_slots, routing_weights)

    def _local_expert_slots(
        self,
        routed_input: torch.Tensor,
        expert_ids: torch.Tensor,
        *,
        ids_are_local: bool,
    ) -> torch.Tensor:
        """Evaluate this rank's experts without reducing the top-k slot axis."""

        if expert_ids.ndim != 2:
            raise ValueError("expert_ids must be rank 2")
        if expert_ids.shape[0] != routed_input.shape[0]:
            raise ValueError("expert_ids must have one row per routed token")

        local_ids = (
            expert_ids if ids_are_local else expert_ids - self.local_expert_start
        )
        valid = (local_ids >= 0) & (local_ids < self.local_expert_count)
        expert_slots = routed_input.new_zeros(
            routed_input.shape[0],
            expert_ids.shape[1],
            routed_input.shape[1],
        )
        selected_local = local_ids[valid].unique()
        for local_expert_tensor in selected_local:
            local_expert = int(local_expert_tensor.item())
            matches = (local_ids == local_expert).nonzero(as_tuple=False)
            token_indices = matches[:, 0]
            slot_indices = matches[:, 1]
            expert_slots[token_indices, slot_indices] = self._run_local_expert(
                routed_input[token_indices], local_expert
            )
        return expert_slots

    @staticmethod
    def _reduce_expert_slots(
        expert_slots: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Dummy's FP32 router weighting in the original slot order."""

        if expert_slots.shape[:2] != routing_weights.shape:
            raise ValueError("expert slots and routing weights must align")
        # Match the original K3 MoE reduction exactly.  The Dummy first
        # restores every expert result to its [token, top-k slot] position,
        # then casts expert values to the FP32 router-weight dtype, multiplies,
        # sums the top-k axis in FP32, and rounds to BF16 once.  Weighting each
        # expert in BF16 and index_add_-ing directly into a BF16 token buffer
        # changes layer-1 expert_sum enough to perturb the next layer's KDA
        # gate and, eventually, routed expert selection.
        return (
            expert_slots.to(dtype=routing_weights.dtype)
            .mul(routing_weights.unsqueeze(-1))
            .sum(dim=1)
            .to(dtype=expert_slots.dtype)
        )

    def _pad_dispatch_payload(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1] != self.latent_size:
            raise ValueError(
                f"expected latent width {self.latent_size}, got {latent.shape[-1]}"
            )
        if self.dispatch_hidden_size == self.latent_size:
            return latent
        return F.pad(latent, (0, self.dispatch_hidden_size - self.latent_size))

    def _tp_token_slice(
        self,
        routed_input: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Avoid dispatching the same replicated TP tokens more than once."""

        token_count = routed_input.shape[0]
        tokens_per_tp_rank = (token_count + self.attn_tp_size - 1) // self.attn_tp_size
        begin = min(tokens_per_tp_rank * self.attn_tp_rank, token_count)
        size = min(tokens_per_tp_rank, token_count - begin)
        return (
            routed_input.narrow(0, begin, size),
            expert_ids.narrow(0, begin, size),
            routing_weights.narrow(0, begin, size),
            tokens_per_tp_rank,
        )

    def _tp_gather(
        self,
        output: torch.Tensor,
        original_token_count: int,
        tokens_per_tp_rank: int,
    ) -> torch.Tensor:
        if self.attn_tp_size == 1:
            return output
        if output.shape[0] < tokens_per_tp_rank:
            output = torch.cat(
                (
                    output,
                    output.new_zeros(
                        tokens_per_tp_rank - output.shape[0], output.shape[1]
                    ),
                ),
                dim=0,
            )
        gathered = all_gather(output, group=Group.TP).reshape(
            self.attn_tp_size * tokens_per_tp_rank, -1
        )
        return gathered[:original_token_count]

    @staticmethod
    def _deepep_wrapper():
        from rtp_llm.models_py.distributed.deepep_wrapper import DeepEPWrapper

        if not DeepEPWrapper.is_initialized() or DeepEPWrapper._instance is None:
            raise RuntimeError(
                "Kimi K3 ep_size>1 requires RTP DeepEP initialization; enable "
                "USE_DEEPEP_MOE=1 and disable USE_ALL_GATHER"
            )
        return DeepEPWrapper._instance

    def _deepep_normal(
        self,
        wrapper: Any,
        routed_input: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """One DeepEP round trip with Dummy-ordered source-side reduction.

        ACCL-EP normal combine does not instantiate a top-k-16 kernel.  The
        previous production path therefore split K3's slots into 10 + 6 and
        reduced each group before adding the two results.  That regrouped the
        FP32 additions relative to Dummy's ordered ``sum(dim=topk)`` and can
        perturb close router/logit decisions after many decode layers.

        Expand each source token into occurrence rows instead.  Each row uses
        DeepEP's supported width-four routing metadata and carries as many
        independent expert outputs as fit in the 7168-byte communication
        vector (two 3584-wide K3 latent outputs).  DeepEP combines those
        independent payload slices without router weighting.  The source rank
        then restores all original top-k slots and performs exactly one
        ordered FP32 weighting/reduction.  This preserves one dispatch and one
        combine per MoE layer, unlike the diagnostic slot-wise canonical path.
        """

        if expert_ids.shape != routing_weights.shape or expert_ids.ndim != 2:
            raise ValueError(
                "DeepEP expert ids and routing weights must be equal rank-2 tensors"
            )
        token_count, topk = expert_ids.shape
        if topk == 0:
            raise ValueError("DeepEP routing requires at least one expert slot")

        # K3's communication payload is 7168 and its latent expert output is
        # 3584, so the real model carries two ordered slots per occurrence
        # row.  Keep the construction valid for reduced-size smoke configs.
        slots_per_row = min(4, self.dispatch_hidden_size // self.latent_size)
        if slots_per_row < 1:
            raise RuntimeError(
                "Kimi K3 DeepEP payload cannot hold one latent expert output"
            )
        row_count = (topk + slots_per_row - 1) // slots_per_row
        padded_topk = row_count * slots_per_row

        padded_ids = torch.full(
            (token_count, padded_topk),
            -1,
            dtype=expert_ids.dtype,
            device=expert_ids.device,
        )
        padded_ids[:, :topk] = expert_ids
        occurrence_ids = torch.full(
            (token_count * row_count, 4),
            -1,
            dtype=expert_ids.dtype,
            device=expert_ids.device,
        )
        occurrence_ids[:, :slots_per_row] = padded_ids.reshape(
            token_count * row_count, slots_per_row
        )
        occurrence_weights = (occurrence_ids >= 0).to(dtype=routing_weights.dtype)

        occurrence_input = (
            routed_input.unsqueeze(1)
            .expand(token_count, row_count, self.latent_size)
            .reshape(token_count * row_count, self.latent_size)
        )
        dispatch_input = self._pad_dispatch_payload(occurrence_input)
        buffer = wrapper.buffer
        with _perf_profile("k3.moe.deepep_occurrence.dispatch", dispatch_input):
            (
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                num_tokens_per_expert,
                is_token_in_rank,
                _,
            ) = buffer.get_dispatch_layout(occurrence_ids, self.expert_num)
            (
                recv_x,
                recv_topk_idx,
                _,
                _,
                handle,
                _,
            ) = buffer.dispatch(
                dispatch_input,
                None,
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                occurrence_ids,
                occurrence_weights,
                expert_alignment=1,
            )
        if not isinstance(recv_x, torch.Tensor):
            raise RuntimeError("Kimi K3 DeepEP path requires BF16 dispatch")

        with _perf_profile("k3.moe.deepep_occurrence.experts", recv_x):
            local_slots = self._local_expert_slots(
                recv_x[:, : self.latent_size],
                recv_topk_idx,
                ids_are_local=True,
            )
            combine_payload = recv_x.new_zeros(
                recv_x.shape[0], self.dispatch_hidden_size
            )
            for slot_idx in range(slots_per_row):
                begin = slot_idx * self.latent_size
                combine_payload[:, begin : begin + self.latent_size] = local_slots[
                    :, slot_idx
                ]

        with _perf_profile("k3.moe.deepep_occurrence.combine", combine_payload):
            combined, _, _ = buffer.combine(
                combine_payload,
                handle,
            )

        with _perf_profile("k3.moe.deepep_occurrence.source_reduce", combined):
            source_slots = torch.stack(
                [
                    combined[
                        :,
                        slot_idx * self.latent_size : (slot_idx + 1) * self.latent_size,
                    ]
                    for slot_idx in range(slots_per_row)
                ],
                dim=1,
            )
            source_slots = source_slots.reshape(
                token_count, padded_topk, self.latent_size
            )[:, :topk]
            return self._reduce_expert_slots(source_slots, routing_weights)

    def _deepep_low_latency(
        self,
        wrapper: Any,
        routed_input: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """DeepEP low-latency occurrence dispatch -> combine in latent space."""

        buffer = wrapper.buffer
        dispatch_input = self._pad_dispatch_payload(routed_input)
        recv_x, recv_counts, handle, _, _ = buffer.low_latency_dispatch(
            x=dispatch_input,
            topk_idx=expert_ids,
            num_max_dispatch_tokens_per_rank=wrapper.ll_num_max_token_per_rank,
            num_experts=self.expert_num,
            use_fp8=False,
            async_finish=False,
            return_recv_hook=False,
        )
        if not isinstance(recv_x, torch.Tensor):
            raise RuntimeError("Kimi K3 DeepEP correctness path requires BF16 dispatch")
        if recv_x.ndim != 3 or recv_x.shape[0] != self.local_expert_count:
            raise RuntimeError(
                "unexpected DeepEP low-latency receive shape: " f"{tuple(recv_x.shape)}"
            )
        expert_output = torch.zeros_like(recv_x)
        for local_expert in range(self.local_expert_count):
            count = int(recv_counts[local_expert].item())
            if count == 0:
                continue
            expert_output[local_expert, :count, : self.latent_size] = (
                self._run_local_expert(
                    recv_x[local_expert, :count, : self.latent_size],
                    local_expert,
                )
            )
        combined, _, _ = buffer.low_latency_combine(
            x=expert_output,
            topk_idx=expert_ids,
            topk_weights=routing_weights,
            handle=handle,
            zero_copy=False,
            async_finish=False,
            return_recv_hook=False,
        )
        return combined[:, : self.latent_size]

    def _distributed_expert_sum(
        self,
        routed_input: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        *,
        sequence_parallel: bool = False,
    ) -> torch.Tensor:
        if True:
            if sequence_parallel:
                return self._deep_gemm_mega_expert_sum(
                    routed_input,
                    expert_ids,
                    routing_weights,
                )
            sliced_input, sliced_ids, sliced_weights, tokens_per_tp_rank = (
                self._tp_token_slice(routed_input, expert_ids, routing_weights)
            )
            local_output = self._deep_gemm_mega_expert_sum(
                sliced_input,
                sliced_ids,
                sliced_weights,
            )
            return self._tp_gather(
                local_output,
                routed_input.shape[0],
                tokens_per_tp_rank,
            )

        from rtp_llm.models_py.distributed.deepep_wrapper import (
            DeepEPMode,
            DeepEPWrapper,
        )

        wrapper = self._deepep_wrapper()
        sp_active = sequence_parallel and self.attn_tp_size > 1 and routed_input.is_cuda
        if sp_active:
            sliced_input = routed_input
            sliced_ids = expert_ids
            sliced_weights = routing_weights
            tokens_per_tp_rank = routed_input.shape[0]
        else:
            sliced_input, sliced_ids, sliced_weights, tokens_per_tp_rank = (
                self._tp_token_slice(routed_input, expert_ids, routing_weights)
            )
        if wrapper.mode == DeepEPMode.NORMAL:
            local_output = self._deepep_normal(
                wrapper, sliced_input, sliced_ids, sliced_weights
            )
        elif wrapper.mode == DeepEPMode.LOW_LATENCY:
            local_output = self._deepep_low_latency(
                wrapper, sliced_input, sliced_ids, sliced_weights
            )
        else:
            raise RuntimeError(
                "Kimi K3 does not yet support DeepEP LOW_LATENCY_M2N/FFN "
                f"disaggregation mode: {wrapper.mode}"
            )
        return (
            local_output
            if sp_active
            else self._tp_gather(
                local_output, routed_input.shape[0], tokens_per_tp_rank
            )
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        sequence_parallel: bool = False,
        valid_token_count: Optional[int] = None,
    ) -> torch.Tensor:
        sp_active = (
            sequence_parallel and self.attn_tp_size > 1 and hidden_states.is_cuda
        )
        trace_enabled = _accuracy_trace_enabled()
        with _perf_profile(
            f"{self.trace_prefix}.router_sigmoid_grouped_topk16", hidden_states
        ):
            expert_ids, routing_weights = self._route(hidden_states)
        if valid_token_count is not None:
            if valid_token_count < 0 or valid_token_count > hidden_states.shape[0]:
                raise ValueError(
                    "valid_token_count is outside the local token shard: "
                    f"valid={valid_token_count}, rows={hidden_states.shape[0]}"
                )
            if valid_token_count < hidden_states.shape[0]:
                expert_ids = expert_ids.clone()
                routing_weights = routing_weights.clone()
                # DeepGEMM validates every expert id before applying its
                # routing weight. Padding rows still need an in-range id;
                # zero weights and the output clear below keep them inert.
                expert_ids[valid_token_count:] = 0
                routing_weights[valid_token_count:] = 0
        if trace_enabled:
            router_token_dim = None if tensor_dump_full_router() else 0
            record_accuracy_tensor(
                f"{self.trace_prefix}.expert_ids",
                expert_ids,
                token_dim=router_token_dim,
            )
            record_accuracy_tensor(
                f"{self.trace_prefix}.routing_weights",
                routing_weights,
                token_dim=router_token_dim,
            )
            record_accuracy_tensor(
                f"{self.trace_prefix}.router_counts",
                torch.bincount(
                    (
                        expert_ids[:valid_token_count].reshape(-1)
                        if valid_token_count is not None
                        else expert_ids.reshape(-1)
                    ),
                    minlength=self.expert_num,
                ),
            )
        with _perf_profile(
            f"{self.trace_prefix}.routed_down_7168_to_3584", hidden_states
        ):
            routed_input = _linear(hidden_states, self.weights[K3W.MOE_ROUTED_DOWN])
        if trace_enabled:
            record_accuracy_tensor(
                f"{self.trace_prefix}.routed_input", routed_input, token_dim=0
            )
        with _perf_profile(
            f"{self.trace_prefix}.routed_experts_mega_dispatch_compute_combine",
            routed_input,
        ):
            routed_output = (
                self._distributed_expert_sum(
                    routed_input,
                    expert_ids,
                    routing_weights,
                    sequence_parallel=sp_active,
                )
                if self.ep_size > 1
                else self._local_expert_sum(
                    routed_input,
                    expert_ids,
                    routing_weights,
                    ids_are_local=False,
                )
            )
        if trace_enabled:
            record_accuracy_tensor(
                f"{self.trace_prefix}.expert_sum", routed_output, token_dim=0
            )
        if self.latent_moe_use_norm:
            with _perf_profile(
                f"{self.trace_prefix}.routed_latent_rmsnorm", routed_output
            ):
                routed_output = _rms_norm(
                    routed_output, self.weights[K3W.MOE_ROUTED_NORM], self.eps
                )
        if trace_enabled:
            record_accuracy_tensor(
                f"{self.trace_prefix}.routed_normalized",
                routed_output,
                token_dim=0,
            )
        with _perf_profile(
            f"{self.trace_prefix}.routed_up_3584_to_7168", routed_output
        ):
            routed_output = _linear(routed_output, self.weights[K3W.MOE_ROUTED_UP])
        if trace_enabled:
            record_accuracy_tensor(
                f"{self.trace_prefix}.routed_output", routed_output, token_dim=0
            )
        shared_gate_weight = _sequence_parallel_column_weight(
            self.weights,
            K3W.MOE_SHARED_GATE,
            self.ffn_tp_size,
            self.ffn_tp_rank,
            self._full_column_weights,
            "sp_shared_gate",
            sequence_parallel=sp_active,
        )
        shared_up_weight = _sequence_parallel_column_weight(
            self.weights,
            K3W.MOE_SHARED_UP,
            self.ffn_tp_size,
            self.ffn_tp_rank,
            self._full_column_weights,
            "sp_shared_up",
            sequence_parallel=sp_active,
        )
        with _perf_profile(
            f"{self.trace_prefix}.shared_expert_"
            + ("replicated_gate_up_gemm" if sp_active else "gate_up_gemm"),
            hidden_states,
        ):
            shared_gate = _linear(hidden_states, shared_gate_weight)
            shared_up = _linear(hidden_states, shared_up_weight)
        with _perf_profile(f"{self.trace_prefix}.shared_expert_situ", shared_gate):
            shared_activation = _situ(
                shared_gate,
                shared_up,
                self.beta,
                self.linear_beta,
            )
        if trace_enabled:
            record_accuracy_tensor(
                f"{self.trace_prefix}.shared_gate", shared_gate, token_dim=0
            )
            record_accuracy_tensor(
                f"{self.trace_prefix}.shared_up", shared_up, token_dim=0
            )
            record_accuracy_tensor(
                f"{self.trace_prefix}.shared_activation",
                shared_activation,
                token_dim=0,
            )
        with _perf_profile(
            f"{self.trace_prefix}.shared_expert_replicated_down_no_collective",
            shared_activation,
        ):
            shared_down_weight = _sequence_parallel_row_weight(
                self.weights,
                K3W.MOE_SHARED_DOWN,
                self.ffn_tp_size,
                self.ffn_tp_rank,
                self._full_row_weights,
                "sp_shared_down",
                sequence_parallel=sp_active,
            )
            shared_output = (
                _linear(shared_activation, shared_down_weight)
                if sp_active
                else _row_parallel_linear(
                    shared_activation,
                    shared_down_weight,
                    self.parallelism_config.get_ffn_tp_size(),
                )
            )
        if trace_enabled:
            record_accuracy_tensor(
                f"{self.trace_prefix}.shared_output", shared_output, token_dim=0
            )
        with _perf_profile(f"{self.trace_prefix}.sum_routed_and_shared", routed_output):
            output = routed_output + shared_output
        if valid_token_count is not None and valid_token_count < hidden_states.shape[0]:
            output = output.clone()
            output[valid_token_count:] = 0
        if trace_enabled:
            record_accuracy_tensor(f"{self.trace_prefix}.output", output, token_dim=0)
        return output


class KimiK3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiK3ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.layer_idx = int(layer_idx)
        self.eps = float(config.layernorm_eps)
        self.attn_res_block_size = config.k3_runtime_config.attn_res_block_size
        layer_type = config.hybrid_attention_config.hybrid_attention_types[layer_idx]
        self.is_kda = layer_type == HybridAttentionType.LINEAR
        self.self_attn: nn.Module = (
            KimiK3KDA(config, parallelism_config, weights, layer_idx)
            if self.is_kda
            else KimiK3MLA(config, parallelism_config, weights, layer_idx)
        )
        self.mlp: nn.Module = (
            KimiK3LatentMoE(config, parallelism_config, weights, layer_idx)
            if layer_idx in set(config.moe_layer_index)
            else KimiK3DenseMLP(config, parallelism_config, weights, layer_idx)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        mode: KDAExecutionMode,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        fmha_impl: Any = None,
        sequence_parallel: bool = False,
        prefill_sp_layout: Optional[_TokenShardLayout] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trace_prefix = f"layer.{self.layer_idx}"
        trace_enabled = _accuracy_trace_enabled()
        decode_sp = sequence_parallel and mode == "decode"
        decode_sp_debug = decode_sp and _debug_enabled()
        logical_tokens = int(hidden_states.shape[0])
        tp_size = int(self.self_attn.parallelism_config.get_attn_tp_size())
        tp_rank = int(self.self_attn.parallelism_config.get_attn_tp_rank())
        local_valid_tokens: Optional[int] = None
        if (
            prefill_sp_layout is not None
            and prefill_sp_layout.local_valid_tokens < prefill_sp_layout.local_tokens
        ):
            local_valid_tokens = prefill_sp_layout.local_valid_tokens
        if decode_sp_debug:
            logging.info(
                "[K3_DECODE_SP_DEBUG] rank=%d layer=%d enter hidden=%s block=%s",
                tp_rank,
                self.layer_idx,
                tuple(hidden_states.shape),
                tuple(block_residual.shape),
            )
        if trace_enabled:
            record_accuracy_tensor(f"{trace_prefix}.input", hidden_states, token_dim=0)
        prefix_sum: Optional[torch.Tensor] = hidden_states
        expected_previous_blocks = (
            self.layer_idx + self.attn_res_block_size - 1
        ) // self.attn_res_block_size
        previous_blocks = min(expected_previous_blocks, block_residual.shape[1])
        writes_block = self.layer_idx % self.attn_res_block_size == 0
        block_write_idx = (
            previous_blocks
            if writes_block and block_residual.shape[1] > previous_blocks
            else -1
        )
        attention_input: Optional[torch.Tensor] = None
        if previous_blocks > 0 or block_write_idx >= 0:
            with _perf_profile(f"{trace_prefix}.self_attn_residual_mix", prefix_sum):
                attention_input = _attention_residual(
                    prefix_sum,
                    block_residual,
                    self.weights[K3W.SELF_ATTN_RES_NORM],
                    self.weights[K3W.SELF_ATTN_RES_PROJ],
                    self.eps,
                    self.weights[W.pre_ln_gamma],
                    self.eps,
                    num_blocks=previous_blocks,
                    block_write_idx=block_write_idx,
                )
        if writes_block:
            if block_write_idx < 0:
                block_residual = torch.cat(
                    (block_residual, prefix_sum.unsqueeze(1)), dim=1
                )
            prefix_sum = None
        active_blocks = previous_blocks + int(writes_block)
        active_block_residual = block_residual[:, :active_blocks]
        if attention_input is None:
            with _perf_profile(
                f"{trace_prefix}.attention_input_rmsnorm", hidden_states
            ):
                attention_input = _rms_norm(
                    hidden_states, self.weights[W.pre_ln_gamma], self.eps
                )
        if trace_enabled:
            record_accuracy_tensor(
                f"{trace_prefix}.attention_input", attention_input, token_dim=0
            )
        if self.is_kda:
            with _perf_profile(
                f"{trace_prefix}.KDA_attention_complete", attention_input
            ):
                attention_output, _ = self.self_attn(
                    attention_input,
                    cu_seqlens,
                    mode=mode,
                    kv_cache=kv_cache,
                    attention_inputs=attention_inputs,
                    sequence_parallel=sequence_parallel,
                    prefill_sp_layout=prefill_sp_layout,
                )
        else:
            # MLA layers use the shared ``MlaAttention`` signature and consume
            # the framework fmha_impl built by ``prepare_fmha_impl``.
            with _perf_profile(
                f"{trace_prefix}.MLA_attention_complete", attention_input
            ):
                attention_output = self.self_attn(
                    attention_input,
                    fmha_impl,
                    kv_cache,
                    attention_inputs=attention_inputs,
                    sequence_parallel=sequence_parallel,
                    prefill_sp_layout=prefill_sp_layout,
                )
        if decode_sp:
            # Attention consumes only real decode tokens. Its row-parallel
            # output projection pads immediately before ReduceScatter; shard
            # the residual tensors with the identical contiguous plan so all
            # subsequent local math lines up with that output.
            if prefix_sum is not None:
                prefix_sum, local_valid_tokens = _padded_token_shard(
                    prefix_sum,
                    logical_tokens,
                    tp_size,
                    tp_rank,
                )
            active_block_residual, block_valid_tokens = _padded_token_shard(
                active_block_residual,
                logical_tokens,
                tp_size,
                tp_rank,
            )
            if local_valid_tokens is None:
                local_valid_tokens = block_valid_tokens
            elif local_valid_tokens != block_valid_tokens:
                raise RuntimeError(
                    "K3 Decode token-SP residual shards disagree on valid rows"
                )
        attention_delta: Optional[torch.Tensor] = None
        with _perf_profile(f"{trace_prefix}.attention_prefix_sum", attention_output):
            if prefix_sum is None:
                prefix_sum = attention_output
            elif (
                not trace_enabled
                and _perf_fusions_enabled()
                and prefix_sum.is_cuda
            ):
                attention_delta = attention_output
            else:
                prefix_sum = prefix_sum + attention_output
        mlp_input: Optional[torch.Tensor] = None
        with _perf_profile(f"{trace_prefix}.mlp_attn_residual_mix", prefix_sum):
            if trace_enabled:
                mlp_input = _attention_residual(
                    prefix_sum,
                    active_block_residual,
                    self.weights[K3W.MLP_RES_NORM],
                    self.weights[K3W.MLP_RES_PROJ],
                    self.eps,
                )
            else:
                normalized_mlp_input = _attention_residual(
                    prefix_sum,
                    active_block_residual,
                    self.weights[K3W.MLP_RES_NORM],
                    self.weights[K3W.MLP_RES_PROJ],
                    self.eps,
                    self.weights[W.post_ln_gamma],
                    self.eps,
                    attention_delta,
                    active_blocks,
                )
        if mlp_input is not None:
            record_accuracy_tensor(f"{trace_prefix}.mlp_input", mlp_input, token_dim=0)
            with _perf_profile(f"{trace_prefix}.mlp_input_rmsnorm", mlp_input):
                normalized_mlp_input = _rms_norm(
                    mlp_input, self.weights[W.post_ln_gamma], self.eps
                )
        if trace_enabled:
            record_accuracy_tensor(
                f"{trace_prefix}.normalized_mlp_input",
                normalized_mlp_input,
                token_dim=0,
            )
        mlp_kind = (
            "latent_MoE" if isinstance(self.mlp, KimiK3LatentMoE) else "dense_MLP"
        )
        with _perf_profile(f"{trace_prefix}.{mlp_kind}_complete", normalized_mlp_input):
            mlp_output = self.mlp(
                normalized_mlp_input,
                sequence_parallel=sequence_parallel,
                valid_token_count=local_valid_tokens,
            )
        with _perf_profile(f"{trace_prefix}.residual_add", mlp_output):
            output = prefix_sum + mlp_output
        if decode_sp:
            with _perf_profile(
                f"{trace_prefix}.decode_token_allgather_trim_TP8",
                output,
            ):
                output = all_gather_trim(output, logical_tokens, group=Group.TP)
        if trace_enabled:
            record_accuracy_tensor(f"{trace_prefix}.output", output, token_dim=0)
            record_accuracy_tensor(
                f"{trace_prefix}.block_residual",
                block_residual[:, :active_blocks],
                token_dim=0,
            )
        return output, block_residual


class KimiK3Model(GptModelBase):
    """Text decoder body consumed by RTP's Python model executor."""

    def __init__(
        self,
        model_config: KimiK3ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ) -> None:
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.embedding_weight = weights.get_global_weight(W.embedding)
        self.attn_res_block_size = int(
            model_config.k3_runtime_config.attn_res_block_size
        )
        self.num_attn_res_blocks = (
            self.layer_num + self.attn_res_block_size - 1
        ) // self.attn_res_block_size
        self.layers = nn.ModuleList(
            [
                KimiK3DecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[layer_idx],
                    layer_idx,
                )
                for layer_idx in range(self.layer_num)
            ]
        )
        self.final_norm_weight = weights.get_global_weight(W.final_ln_gamma)
        self.output_attn_res_norm = weights.get_global_weight(K3W.OUTPUT_ATTN_RES_NORM)
        self.output_attn_res_proj = weights.get_global_weight(K3W.OUTPUT_ATTN_RES_PROJ)
        self._layer_group_ids: Optional[tuple[int, ...]] = None
        self._kda_a2a_weights_materialized = False
        self._fused_ag_gemm_workspace_ready = False
        self._mtp_hidden_buffer: Optional[torch.Tensor] = None
        self._mtp_hidden_valid_tokens = 0

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        """Bind runtime resources and reserve the largest Prefill AG workspace."""

        super().initialize(init_resource)
        if self._fused_ag_gemm_workspace_ready:
            return True

        tp_size = int(self.parallelism_config.get_attn_tp_size())
        max_global_tokens = int(self.config.max_seq_len) * int(
            init_resource.max_context_batch_size
        )
        max_local_tokens = (max_global_tokens + tp_size - 1) // tp_size
        max_physical_tokens = max_local_tokens * tp_size
        if (
            init_resource.is_decode_role
            or tp_size <= 1
            or not _use_fused_prefill_ag_gemm(max_physical_tokens)
        ):
            return True
        workspace_bytes = (
            max_local_tokens
            * int(self.config.hidden_size)
            * self.embedding_weight.element_size()
        )
        reserve_fused_all_gather_matmul_workspace(
            get_process_group(Group.TP),
            workspace_bytes,
        )
        self._fused_ag_gemm_workspace_ready = True
        logging.info(
            "[K3_FUSED_AG_GEMM] reserved %.3f GiB symmetric workspace "
            "for %d global Prefill tokens (TP%d)",
            workspace_bytes / (1 << 30),
            max_global_tokens,
            tp_size,
        )
        return True

    def get_mtp_target_hidden_states(self, num_tokens: int) -> Optional[torch.Tensor]:
        if self._mtp_hidden_buffer is None:
            return None
        rows = (
            self._mtp_hidden_valid_tokens if int(num_tokens) < 0 else int(num_tokens)
        )
        if rows < 0 or rows > self._mtp_hidden_buffer.size(0):
            raise ValueError(
                f"Kimi K3 EAGLE hidden rows {rows} exceed buffered "
                f"rows {self._mtp_hidden_buffer.size(0)}"
            )
        return self._mtp_hidden_buffer.narrow(0, 0, rows)

    # ``prepare_fmha_impl`` is inherited from ``GptModelBase``: it builds the
    # framework MLA impl via ``AttnImplFactory.get_fmha_impl`` (identical to the
    # generic MoE path).  K3's MLA layers consume that impl through
    # ``KimiK3MLA`` (an ``MlaAttention`` subclass); K3's KDA layers ignore it.

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = F.embedding(input_ids, self.embedding_weight)
        if self.parallelism_config.get_attn_tp_size() > 1:
            tokens, local_hidden = hidden_states.shape
            hidden_states = all_gather(hidden_states, group=Group.TP)
            if _perf_fusions_enabled():
                return kimi_k3_interleave_tp_hidden(
                    hidden_states,
                    tokens,
                    self.parallelism_config.get_attn_tp_size(),
                )
            hidden_states = (
                hidden_states.reshape(
                    self.parallelism_config.get_attn_tp_size(),
                    tokens,
                    local_hidden,
                )
                .transpose(0, 1)
                .contiguous()
                .reshape(tokens, -1)
            )
        return hidden_states

    def _materialize_kda_a2a_weights(self) -> None:
        """Preflight memory and build all KDA A2A layouts in layer order."""

        if self._kda_a2a_weights_materialized:
            return
        a2a_layers = [
            layer.self_attn
            for layer in self.layers
            if layer.is_kda
            and isinstance(layer.self_attn, KimiK3KDA)
            and layer.self_attn.uses_a2a_comm
        ]
        if not a2a_layers:
            self._kda_a2a_weights_materialized = True
            return
        if len(a2a_layers) > 3:
            raise RuntimeError(
                "K3 full-model KDA A2A is disabled on 8xB300: replicating "
                f"{len(a2a_layers)} KDA layers exceeds the measured static "
                "memory budget. Use rs_ag, or the four-layer A2A timeline "
                "checkpoint."
            )
        extra_bytes = sum(layer.a2a_extra_weight_bytes() for layer in a2a_layers)
        safety_bytes = _KDA_A2A_SAFETY_BYTES
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_tensor = torch.tensor(
            [free_bytes],
            dtype=torch.int64,
            device=a2a_layers[0].weights[W.linear_attn_qkv_w].device,
        )
        minimum_free = int(all_gather(free_tensor, Group.TP).min().item())
        required_bytes = extra_bytes + safety_bytes
        logging.info(
            "[K3_KDA_A2A] preflight layers=%d extra=%.3fGiB safety=%.3fGiB "
            "minimum_free=%.3fGiB device_total=%.3fGiB",
            len(a2a_layers),
            extra_bytes / (1 << 30),
            safety_bytes / (1 << 30),
            minimum_free / (1 << 30),
            total_bytes / (1 << 30),
        )
        if required_bytes > minimum_free:
            raise RuntimeError(
                "K3 KDA A2A weight replication would exceed the memory "
                "guard: "
                f"extra={extra_bytes / (1 << 30):.3f}GiB, "
                f"safety={safety_bytes / (1 << 30):.3f}GiB, "
                f"minimum_free={minimum_free / (1 << 30):.3f}GiB"
            )
        for layer in a2a_layers:
            layer.materialize_a2a_weights()
        # Weight materialization is deliberately outside measured requests.
        # Return one-layer-at-a-time gather intermediates to the driver before
        # the representative materialization/JIT request begins.
        torch.cuda.empty_cache()
        barrier(Group.TP)
        self._kda_a2a_weights_materialized = True

    @staticmethod
    def _cu_seqlens(
        attention_inputs: PyAttentionInputs, input_ids: torch.Tensor
    ) -> torch.Tensor:
        cu_seqlens = (
            attention_inputs.cu_seqlens
            if attention_inputs.is_prefill
            else attention_inputs.decode_cu_seqlens_d
        )
        if cu_seqlens is None or cu_seqlens.numel() == 0:
            cu_seqlens = (
                torch.tensor(
                    [0, input_ids.numel()],
                    dtype=torch.int32,
                    device=input_ids.device,
                )
                if attention_inputs.is_prefill
                else torch.arange(
                    input_ids.numel() + 1,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
        graph_decode = not attention_inputs.is_prefill and (
            bool(getattr(attention_inputs, "is_cuda_graph", False))
            or (input_ids.is_cuda and torch.cuda.is_current_stream_capturing())
        )
        if graph_decode:
            # Decode has exactly one packed token per request.  Inspecting the
            # CUDA prefix sums on the host would make capture illegal and would
            # freeze replay metadata; shape validation is sufficient here.
            if cu_seqlens.numel() != input_ids.numel() + 1:
                raise ValueError(
                    "K3 CUDA Graph decode requires one cu_seqlens interval per token"
                )
        else:
            _sequence_offsets(
                cu_seqlens,
                input_ids.numel(),
                cu_seqlens_host=(
                    getattr(attention_inputs, "cu_seqlens_host", None)
                    if _host_metadata_enabled()
                    else None
                ),
            )
        return cu_seqlens

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        attention_inputs = inputs.attention_inputs
        phase = (
            "prefill"
            if attention_inputs is not None and attention_inputs.is_prefill
            else "decode"
        )
        with kimi_k3_accuracy_trace(phase):
            return self._forward_impl(inputs, fmha_impl)

    def _forward_impl(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> PyModelOutputs:
        attention_inputs = inputs.attention_inputs
        if attention_inputs is None:
            raise ValueError("Kimi K3 requires PyAttentionInputs")
        if not attention_inputs.is_prefill and self.kv_cache is None:
            raise RuntimeError("Kimi K3 decode requires an initialized hybrid cache")
        input_ids = inputs.input_ids.reshape(-1)
        trace_enabled = _accuracy_trace_enabled()
        tp_size = int(self.parallelism_config.get_attn_tp_size())
        # SP MoE 是 K3 modeling 唯一的流程,不再由开关决定 —— Decode TP8/EP8
        # 不走 SP 就在启动时 die,Prefill 侧生产配置同样一直是 SP。
        tp_rank = int(self.parallelism_config.get_attn_tp_rank())
        sp_requested = tp_size > 1
        is_target_verify = bool(
            getattr(attention_inputs, "is_target_verify", False)
        )
        # The engine represents the multi-token target verification pass with
        # Prefill-shaped metadata, but the verify kernels replay every draft
        # position on every TP rank.  It must therefore stay replicated and
        # must not enter either token-SP path.
        prefill_sp = (
            sp_requested and attention_inputs.is_prefill and not is_target_verify
        )
        prefill_sp_layout = (
            _token_shard_layout(int(input_ids.numel()), tp_size, tp_rank)
            if prefill_sp
            else None
        )
        # Target verify replays multiple speculative positions on every TP
        # rank and its KDA projection performs an ordinary TP all-reduce.  Its
        # token rows are therefore replicated, unlike normal single-token
        # Decode.  Applying Decode token-SP here shards only the residual side
        # and produces incompatible full-token/sharded-token shapes.
        decode_sp = (
            sp_requested
            and not attention_inputs.is_prefill
            and not is_target_verify
        )
        sp_active = prefill_sp or decode_sp
        if not attention_inputs.is_prefill and not getattr(
            self, "_decode_sp_startup_logged", False
        ):
            logging.info(
                "[K3_DECODE_SP] rank=%d env=%s requested=%s active=%s "
                "tokens=%d tp=%d ep=%d",
                int(self.parallelism_config.get_attn_tp_rank()),
                "1",
                sp_requested,
                decode_sp,
                input_ids.numel(),
                tp_size,
                int(self.parallelism_config.ep_size),
            )
            self._decode_sp_startup_logged = True
        cu_seqlens = self._cu_seqlens(attention_inputs, input_ids)
        if sp_active:
            ep_size = int(self.parallelism_config.ep_size)
            if ep_size != tp_size:
                raise RuntimeError(
                    "Kimi K3 Sequence Parallel currently requires TP == EP; "
                    f"got TP={tp_size}, EP={ep_size}"
                )
            if prefill_sp and any(
                layer.is_kda
                and isinstance(layer.self_attn, KimiK3KDA)
                and layer.self_attn.uses_a2a_comm
                for layer in self.layers
            ):
                with _perf_profile(
                    "model.kda_a2a_weight_materialization.exclude_from_profile"
                ):
                    self._materialize_kda_a2a_weights()
        if (
            any(
                layer.is_kda
                and isinstance(layer.self_attn, KimiK3KDA)
                and layer.self_attn.uses_a2a_comm
                for layer in self.layers
            )
            and not prefill_sp
        ):
            raise RuntimeError(
                "KIMI_K3_KDA_COMM_BACKEND=a2a is Prefill-only and requires "
                "TP8/EP8 Sequence Parallel input"
            )
        if trace_enabled:
            mark_accuracy_fake_stream(
                KimiK3LinearCacheAdapter._is_fake_stream(attention_inputs),
                input_ids.device,
            )
            record_accuracy_tensor("input_ids", input_ids.long(), token_dim=0)
            record_accuracy_tensor("cu_seqlens", cu_seqlens)
        with _perf_profile(
            "model.embedding_vocab_parallel_then_hidden_allgather", input_ids
        ):
            hidden_states = self._embed(input_ids)
        if trace_enabled:
            record_accuracy_tensor("embedding", hidden_states, token_dim=0)
        if prefill_sp:
            assert prefill_sp_layout is not None
            with _perf_profile(
                "model.embedding_to_sequence_parallel_token_shard",
                hidden_states,
            ):
                hidden_states = _prefill_token_shard(
                    hidden_states,
                    prefill_sp_layout,
                )
        block_residual = hidden_states.new_empty(
            hidden_states.shape[0],
            self.num_attn_res_blocks,
            hidden_states.shape[1],
        )
        mode: KDAExecutionMode = "prefill" if attention_inputs.is_prefill else "decode"
        write_cache_store_impl = create_write_cache_store_impl(
            attention_inputs, self.kv_cache
        )
        trace_cache_mapping = trace_enabled
        if trace_cache_mapping and attention_inputs.kv_cache_layer_to_group is not None:
            record_accuracy_tensor(
                "kv_cache_layer_to_group",
                attention_inputs.kv_cache_layer_to_group,
            )
        prepared_mla_group_id: Optional[int] = None
        if _host_metadata_enabled() and self._layer_group_ids is None:
            layer_map_host = getattr(
                attention_inputs, "kv_cache_layer_to_group_host", None
            )
            if layer_map_host is not None and layer_map_host.numel():
                self._layer_group_ids = tuple(
                    int(value) for value in layer_map_host.tolist()
                )
        eagle3_enabled = os.environ.get("SP_TYPE", "").lower() == "eagle3"
        eagle3_hidden_states = []
        if eagle3_enabled:
            raw_aux_layers = os.environ.get("KIMI_K3_EAGLE3_AUX_LAYER_IDS")
            if raw_aux_layers:
                aux_layers = [int(value) for value in raw_aux_layers.split(",")]
            else:
                aux_layers = [0, max(0, self.layer_num // 2), self.layer_num - 1]
            if len(aux_layers) != 3 or any(
                layer_id < 0 or layer_id >= self.layer_num for layer_id in aux_layers
            ):
                raise ValueError(
                    "KIMI_K3_EAGLE3_AUX_LAYER_IDS must contain three valid "
                    f"zero-based layer ids for {self.layer_num} target layers"
                )
            aux_layer_set = set(aux_layers)
        else:
            aux_layers = []
            aux_layer_set = set()
        for layer_idx, layer in enumerate(self.layers):
            static_group_id = (
                self._layer_group_ids[layer_idx]
                if self._layer_group_ids is not None
                and layer_idx < len(self._layer_group_ids)
                else None
            )
            selected_group_id = select_block_map_for_layer(
                attention_inputs, layer_idx, static_group_id
            )
            if selected_group_id is None:
                selected_group_id = 0
            layer_cache = (
                self.kv_cache.get_layer_cache(layer_idx)
                if self.kv_cache is not None
                else None
            )
            if trace_cache_mapping and layer_cache is not None:
                record_accuracy_tensor(
                    f"layer.{layer_idx}.cache_group_ids",
                    torch.tensor(
                        [selected_group_id, layer_cache.group_id],
                        dtype=torch.int32,
                        device=hidden_states.device,
                    ),
                )
                selected_kernel_blocks = (
                    attention_inputs.kv_cache_kernel_block_id_device
                )
                if selected_kernel_blocks is not None:
                    record_accuracy_tensor(
                        f"layer.{layer_idx}.cache_kernel_block_ids",
                        selected_kernel_blocks,
                    )
            if not layer.is_kda and fmha_impl is not None:
                prepared_mla_group_id = _prepare_mla_fmha_for_group(
                    fmha_impl,
                    attention_inputs,
                    selected_group_id,
                    prepared_mla_group_id,
                )
                if trace_cache_mapping:
                    selected_host_blocks = (
                        attention_inputs.kv_cache_kernel_block_id_host
                    )
                    if (
                        selected_host_blocks is not None
                        and selected_host_blocks.numel()
                    ):
                        record_accuracy_tensor(
                            f"layer.{layer_idx}.cache_kernel_block_ids_host",
                            selected_host_blocks.detach().clone(),
                        )
                    fmha_params = getattr(fmha_impl, "fmha_params", None)
                    slot_mapping = (
                        getattr(fmha_params, "slot_mapping", None)
                        if fmha_params is not None
                        else None
                    )
                    if slot_mapping is not None and slot_mapping.numel():
                        record_accuracy_tensor(
                            f"layer.{layer_idx}.mla.slot_mapping",
                            slot_mapping.detach().clone(),
                        )
            with _perf_profile(
                f"layer.{layer_idx}.decoder_layer_complete", hidden_states
            ):
                hidden_states, block_residual = layer(
                    hidden_states,
                    block_residual,
                    cu_seqlens,
                    mode=mode,
                    kv_cache=layer_cache,
                    attention_inputs=attention_inputs,
                    fmha_impl=fmha_impl,
                    sequence_parallel=sp_active,
                    prefill_sp_layout=prefill_sp_layout,
                )
            if layer_idx in aux_layer_set:
                eagle3_hidden_states.append((layer_idx, hidden_states))
            # Loop-level cache-store is only for KDA layers. MLA publishes
            # from its wrapper immediately after concat_and_cache_mla.
            if (
                layer.is_kda
                and write_cache_store_impl is not None
                and layer_cache is not None
            ):
                # The shared writer selects pinned-host length mirrors prepared
                # by PyWrappedModel.  Passing the CUDA length tensors directly
                # is unsafe because PD cache-store consumes them on a CPU
                # background thread. Its physical block table remains 3-D;
                # the C++ writer maps this layer to the KDA cache group.
                layer_cache.cache_store_segment_sizes = list(
                    layer.self_attn.cache_adapter.cache_store_segment_sizes
                )
                with _perf_profile(
                    f"layer.{layer_idx}.pd_cache_store_publish_kda_segments"
                ):
                    write_cache_store_impl(layer_cache)
        if eagle3_enabled:
            by_layer = dict(eagle3_hidden_states)
            mtp_hidden_buffer = torch.cat(
                [by_layer[layer_id] for layer_id in aux_layers], dim=-1
            ).contiguous()
            if prefill_sp:
                assert prefill_sp_layout is not None
                # Auxiliary hidden states are captured while Prefill token
                # sequence parallelism is active.  Eagle3 consumes them next
                # to the replicated full-prompt embedding, so restore the
                # framework's global token layout just like final_hidden below.
                mtp_hidden_buffer = all_gather_trim(
                    mtp_hidden_buffer,
                    prefill_sp_layout.logical_tokens,
                    group=Group.TP,
                )
            self._mtp_hidden_buffer = mtp_hidden_buffer
            self._mtp_hidden_valid_tokens = self._mtp_hidden_buffer.size(0)
        with _perf_profile("model.output_attn_residual_mix", hidden_states):
            active_block_residual = block_residual[:, : self.num_attn_res_blocks]
            if trace_enabled:
                output_attn_res = _attention_residual(
                    hidden_states,
                    active_block_residual,
                    self.output_attn_res_norm,
                    self.output_attn_res_proj,
                    self.config.layernorm_eps,
                )
            else:
                hidden_states = _attention_residual(
                    hidden_states,
                    active_block_residual,
                    self.output_attn_res_norm,
                    self.output_attn_res_proj,
                    self.config.layernorm_eps,
                    self.final_norm_weight,
                    self.config.layernorm_eps,
                )
        if trace_enabled:
            record_accuracy_tensor("output_attn_res", output_attn_res, token_dim=0)
            with _perf_profile("model.final_rmsnorm", output_attn_res):
                hidden_states = _rms_norm(
                    output_attn_res,
                    self.final_norm_weight,
                    self.config.layernorm_eps,
                )
        if prefill_sp:
            assert prefill_sp_layout is not None
            with _perf_profile(
                "model.exit_token_allgather_for_framework_contract",
                hidden_states,
            ):
                hidden_states = all_gather_trim(
                    hidden_states,
                    prefill_sp_layout.logical_tokens,
                    group=Group.TP,
                )
        if trace_enabled:
            record_accuracy_tensor("final_hidden", hidden_states, token_dim=0)
        fmha_params = getattr(fmha_impl, "fmha_params", None)
        return (
            PyModelOutputs(hidden_states, fmha_params)
            if fmha_params is not None
            else PyModelOutputs(hidden_states)
        )


__all__ = [
    "KimiK3LinearCacheAdapter",
    "KimiK3KDA",
    "KimiK3MLA",
    "KimiK3DenseMLP",
    "KimiK3LatentMoE",
    "KimiK3DecoderLayer",
    "KimiK3Model",
]
