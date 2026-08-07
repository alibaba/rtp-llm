"""RTP-LLM serving model for the text-only Kimi K3 decoder.

The hybrid decoder interleaves KDA (linear-attention) and MLA (full-attention)
layers. MLA reuses the framework cache and attention kernels while K3 owns its
NoPE convention, sigmoid output gate, packed input projection and
Sequence-Parallel projection boundary. KDA runs K3's own path (Triton kernel
with a pure-Torch reference): packed prefill dispatches to the chunk scan and
token decode dispatches to the recurrent update. KDA canonical states are
mapped onto RTP's paged linear-cache ABI; MLA uses RTP's compressed latent
cache layout, so the same layer caches can flow through PD transfer.

Both attention modules keep a pure-Torch reference selectable via env var
(``KIMI_K3_KDA_BACKEND`` / ``KIMI_K3_MLA_BACKEND``, default ``kernel``) for
precision comparison against the framework/Triton kernels.
"""

from __future__ import annotations

import inspect
import logging
import os
from contextlib import nullcontext
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
    reduce_scatter,
    reduce_scatter_padded,
)
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.kimi_k3_cuda_graph_cache import (
    load_cuda_graph_decode_tensors,
    store_cuda_graph_decode_state,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.base.common.kvcache_store import (
    create_write_cache_store_impl,
)
from rtp_llm.models_py.modules.hybrid.mla_attention import MlaAttention
from rtp_llm.models_py.modules.kimi_k3.diagnostics.accuracy_trace import (
    accuracy_trace_mode,
    kimi_k3_accuracy_trace,
    mark_accuracy_fake_stream,
    record_accuracy_tensor,
)
from rtp_llm.models_py.modules.kimi_k3.kda_state import KDAExecutionMode, KimiKDAState
from rtp_llm.models_py.modules.kimi_k3.mxfp4 import dequantize_mxfp4
from rtp_llm.models_py.modules.kimi_k3.reference.kda_reference import (
    kimi_kda,
    prepare_kimi_kda_inputs,
)
from rtp_llm.models_py.triton_kernels.kimi_kda import (
    chunk_kda,
    fused_recurrent_kda,
    is_kimi_kda_short_conv_paged_decode_supported,
    kimi_k3_a2a_unpack_rms_norm_sigmoid_gate,
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
from rtp_llm.models_py.triton_kernels.kimi_kda.fused_recurrent import (
    fused_recurrent_kda_fla37_precompiled,
)
from rtp_llm.models_py.utils.typed_storage_view import LinearCacheConverter
from rtp_llm.ops import HybridAttentionType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W
from rtp_llm.utils.util import to_torch_dtype

if TYPE_CHECKING:
    from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3ModelConfig


KIMI_K3_MLA_LATENT_NORM_EPS = 1e-6
_FLASH_KDA_WORKSPACES: dict[tuple[int, int], torch.Tensor] = {}
_FLASH_KDA_LOGGED_DEVICES: set[int] = set()
_CULA_LOGGED_DEVICES: set[int] = set()
_DEEPGEMM_MEGA_LOGGED_DEVICES: set[int] = set()


def _env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").strip() == "1"


def _perf_mode_enabled() -> bool:
    """Enable strict validation and profiler annotations, not model math."""

    return _env_flag("KIMI_K3_PERF_MODE")


def _perf_fusions_enabled() -> bool:
    """Select the explicitly staged performance-fusion implementations."""

    return _env_flag("KIMI_K3_PERF_FUSIONS")


def _batched_kda_decode_enabled() -> bool:
    """Enable the experimental indexed KDA decode path."""

    return _env_flag("KIMI_K3_BATCHED_KDA_DECODE")


def _perf_profile(name: str, tensor: Optional[torch.Tensor] = None):
    if not _perf_mode_enabled():
        return nullcontext()
    suffix = ""
    if tensor is not None:
        shape = "x".join(str(dim) for dim in tensor.shape)
        suffix = f"[shape={shape},dtype={tensor.dtype}]"
    return torch.autograd.profiler.record_function(f"{name}{suffix}")


def _validate_perf_environment() -> None:
    """Reject mixed accuracy/reference settings in a measured performance run."""

    if not _perf_mode_enabled():
        return
    conflicting_flags = [
        name
        for name in (
            "KIMI_K3_ACCURACY_CANONICAL_TP",
            "KIMI_K3_ACCURACY_CANONICAL_EP",
            "KIMI_K3_ACCURACY_CANONICAL_MLA",
            "KIMI_K3_ACCURACY_LOCAL_EAGER_MLA",
        )
        if _env_flag(name)
    ]
    if os.environ.get("KIMI_K3_ACCURACY_TRACE_DIR"):
        conflicting_flags.append("KIMI_K3_ACCURACY_TRACE_DIR")
    expected = {
        "KIMI_K3_MOE_BACKEND": "deep_gemm_mega",
        "KIMI_K3_MLA_BACKEND": "flashmla",
        "KIMI_K3_USE_HOST_METADATA": "1",
        "KIMI_K3_SP_MOE": "1",
        "KIMI_K3_PERF_FUSIONS": "1",
    }
    wrong_settings = [
        f"{name}={os.environ.get(name, '')!r}"
        for name, value in expected.items()
        if os.environ.get(name, "").strip().lower() != value
    ]
    kda_backend = os.environ.get("KIMI_K3_KDA_BACKEND", "").strip().lower()
    if kda_backend != "cula":
        wrong_settings.append(f"KIMI_K3_KDA_BACKEND={kda_backend!r}")
    if conflicting_flags or wrong_settings:
        details = ", ".join(conflicting_flags + wrong_settings)
        raise RuntimeError(
            "KIMI_K3_PERF_MODE only labels the fully optimized path and "
            "forbids accuracy/reference work; invalid settings: "
            f"{details}"
        )


def _sp_moe_enabled() -> bool:
    """Enable K3 token sequence parallelism for the selected role."""

    return os.environ.get("KIMI_K3_SP_MOE", "0").strip() == "1"


def _kda_comm_backend() -> str:
    backend = os.environ.get("KIMI_K3_KDA_COMM_BACKEND", "rs_ag").strip().lower()
    if backend != "rs_ag":
        raise RuntimeError(
            "KIMI_K3_KDA_COMM_BACKEND supports only production 'rs_ag'; "
            f"the experimental A2A path is disabled, got {backend!r}"
        )
    return backend


def _decode_sp_debug_enabled() -> bool:
    return os.environ.get("KIMI_K3_DECODE_SP_DEBUG", "0").strip() == "1"


def _flash_kda_workspace(
    flash_kda: Any,
    q: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Return a process-local reusable FlashKDA workspace."""

    device_index = q.device.index if q.device.index is not None else 0
    sequence_count = int(cu_seqlens.numel() - 1)
    required = int(
        flash_kda.get_workspace_size(
            int(q.shape[0] * q.shape[1]),
            int(q.shape[2]),
            sequence_count,
        )
    )
    key = (device_index, int(q.shape[2]))
    workspace = _FLASH_KDA_WORKSPACES.get(key)
    if workspace is None or workspace.numel() < required:
        workspace = torch.empty(required, dtype=torch.uint8, device=q.device)
        _FLASH_KDA_WORKSPACES[key] = workspace
    return workspace


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply an RTP internal-layout ``[in_features, out_features]`` weight."""

    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"linear input width {x.shape[-1]} does not match weight "
            f"shape {tuple(weight.shape)}"
        )
    return torch.matmul(x, weight)


def _accuracy_canonical_tp_enabled() -> bool:
    """Return the process-wide selector for source-GEMM TP reconstruction."""

    return os.environ.get("KIMI_K3_ACCURACY_CANONICAL_TP", "0") == "1"


def _accuracy_retain_full_tp_weights() -> bool:
    """Whether canonical TP keeps gathered full weights across forwards.

    Retaining them avoids repeated collectives, but a full 93-layer TP8 K3
    Prefill accumulates tens of GiB of duplicate weights during its first
    request. Correctness-only full-model runs can disable retention while
    preserving the same gathered GEMM for every projection.
    """

    return os.environ.get("KIMI_K3_ACCURACY_RETAIN_FULL_TP_WEIGHTS", "1") == "1"


def _accuracy_full_router_trace_enabled() -> bool:
    """Keep full O(T) router diagnostics in an otherwise boundary-only trace."""

    return os.environ.get("KIMI_K3_ACCURACY_TRACE_FULL_ROUTER", "0") == "1"


def _accuracy_trace_enabled() -> bool:
    """Return whether this forward is actively recording accuracy tensors."""

    return not _perf_mode_enabled() and accuracy_trace_mode() is not None


def _accuracy_trace_requested() -> bool:
    """Return the process-wide trace selector shared by every TP rank."""

    return bool(os.environ.get("KIMI_K3_ACCURACY_TRACE_DIR"))


def _host_metadata_enabled() -> bool:
    """Use gather-time pinned host metadata instead of synchronous D2H reads."""

    return os.environ.get("KIMI_K3_USE_HOST_METADATA", "0") == "1"


def _accuracy_canonical_mla_enabled() -> bool:
    """Reproduce the Dummy eager-MLA accumulation in accuracy runs only."""

    return os.environ.get("KIMI_K3_ACCURACY_CANONICAL_MLA", "0") == "1"


def _accuracy_local_eager_mla_enabled() -> bool:
    """Use TP-local source MLA math while retaining RTP cache writeback.

    This default-off diagnostic mode keeps the framework FMHA invocation so
    HybridCache and CacheStore observe the normal production write path, but
    replaces the FlashInfer context with K3's two-matmul eager formulation.
    Unlike ``KIMI_K3_ACCURACY_CANONICAL_MLA``, it does not all-gather all
    attention heads before the quadratic attention calculation.
    """

    return os.environ.get("KIMI_K3_ACCURACY_LOCAL_EAGER_MLA", "0") == "1"


def _accuracy_canonical_ep_enabled() -> bool:
    """Reproduce source-layout MoE math in accuracy runs only."""

    return os.environ.get("KIMI_K3_ACCURACY_CANONICAL_EP", "0") == "1"


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


def _column_parallel_linear(
    x: torch.Tensor,
    local_weight: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    full_weight_cache: dict[str, torch.Tensor],
    cache_key: str,
) -> torch.Tensor:
    """Reproduce the source full-width GEMM before selecting a TP output shard.

    Splitting a column-parallel BF16 weight changes GEMM N and can make cuBLAS
    select a different reduction path.  The resulting one-ULP activation error
    is normally harmless, but K3's near-tied routed experts can amplify it into
    a different token.  The default-off accuracy path gathers each weight once,
    runs the source-width GEMM on every TP rank, and returns the local columns.
    """

    if tp_size <= 1 or not _accuracy_canonical_tp_enabled():
        return _linear(x, local_weight)
    if x.ndim != 2 or local_weight.ndim != 2:
        raise ValueError(
            "canonical K3 TP column projection requires rank-2 input and weight"
        )
    retain_full_weight = _accuracy_retain_full_tp_weights()
    full_weight = full_weight_cache.get(cache_key) if retain_full_weight else None
    if full_weight is None:
        gathered = all_gather(local_weight.transpose(0, 1).contiguous(), group=Group.TP)
        expected_shape = (
            tp_size * local_weight.shape[1],
            local_weight.shape[0],
        )
        if tuple(gathered.shape) != expected_shape:
            raise ValueError(
                "unexpected gathered K3 column weight shape "
                f"{tuple(gathered.shape)}, expected {expected_shape}"
            )
        full_weight = gathered.transpose(0, 1).contiguous()
        if retain_full_weight:
            full_weight_cache[cache_key] = full_weight
    full_output = _linear(x, full_weight)
    local_width = local_weight.shape[1]
    begin = tp_rank * local_width
    # Several KDA Triton kernels consume flattened tensors and therefore
    # require the local projection to have a compact row stride.  A plain
    # column slice keeps the full-width GEMM stride and silently reads across
    # TP-head boundaries.
    return full_output[:, begin : begin + local_width].contiguous()


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
    """Materialize an SP replica once and replace the original TP shard.

    The same model process may later receive a non-divisible Prefill request
    that falls back to replicated-token TP.  In that case return this rank's
    logical column shard as a view of the retained full tensor.
    """

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
) -> torch.Tensor:
    """Apply and reduce a K3 row-parallel projection.

    A BF16 partial result followed by a BF16 TP reduction introduces one
    rounding point per rank.  K3's residual and routed-MoE stack can amplify
    those otherwise small errors enough to change a top-k expert.  Keep the
    CUDA GEMM output and collective in FP32, then round once after the
    reduction.  The CPU branch retains the ordinary path for unit tests and
    environments where CUDA's ``mm(out_dtype=...)`` is unavailable.
    """

    if pad_reduce_scatter_tokens and not reduce_scatter_tokens:
        raise ValueError(
            "pad_reduce_scatter_tokens requires reduce_scatter_tokens=True"
        )
    if tp_size <= 1:
        return _linear(x, weight)
    if _accuracy_canonical_tp_enabled():
        if x.ndim != 2 or weight.ndim != 2:
            raise ValueError(
                "canonical K3 TP row projection requires rank-2 input and weight"
            )
        # A sum of independently accumulated TP shards is mathematically
        # equivalent to the source model's full-width GEMM, but it is not
        # bitwise equivalent in BF16.  Near-tied MoE routes can amplify one
        # output ULP into a different expert selection.  Accuracy tracing
        # therefore reconstructs the source GEMM boundary on every TP rank.
        # This default-off diagnostic path is intentionally not a production
        # performance implementation.
        full_weight = all_gather(weight.contiguous(), group=Group.TP)
        full_x = (
            all_gather(x.transpose(0, 1).contiguous(), group=Group.TP)
            .transpose(0, 1)
            .contiguous()
        )
        if full_x.shape[1] != full_weight.shape[0]:
            raise ValueError(
                "canonical K3 TP row projection gathered incompatible shapes: "
                f"x={tuple(full_x.shape)}, weight={tuple(full_weight.shape)}"
            )
        return _linear(full_x, full_weight)
    if (
        x.is_cuda
        and x.ndim == 2
        and x.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == x.dtype
    ):
        if (
            _perf_fusions_enabled()
            and reduce_scatter_tokens
            and not pad_reduce_scatter_tokens
        ):
            # The optimized SP path keeps the projection and token
            # ReduceScatter in BF16 for long Prefill rows. Decode uses the
            # padded path below: its tiny recurrent batches are sensitive to
            # a per-layer BF16 partial/collective rounding point, while the
            # FP32 collective cost is negligible at those token counts.
            partial = torch.mm(x, weight)
            return (
                reduce_scatter_padded(partial, group=Group.TP)
                if pad_reduce_scatter_tokens
                else reduce_scatter(partial, group=Group.TP)
            )
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


def _source_eager_attention_context(
    query_by_head: torch.Tensor,
    key_by_head: torch.Tensor,
    value_by_head: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match Dummy's eager MLA matmul, mask, softmax, and value matmul.

    Inputs are head-major ``[H, T, D]``, ``[H, S, D]``, and ``[H, S, V]``.
    The leading batch dimension is retained for both matmuls because cuBLAS
    kernel selection can depend on the complete source-model shape.
    """

    if query_by_head.ndim != 3 or key_by_head.ndim != 3:
        raise ValueError("canonical eager MLA query/key must be rank-3")
    if value_by_head.ndim != 3:
        raise ValueError("canonical eager MLA value must be rank-3")
    if query_by_head.shape[0] != key_by_head.shape[0]:
        raise ValueError("canonical eager MLA query/key head counts differ")
    if key_by_head.shape[:2] != value_by_head.shape[:2]:
        raise ValueError("canonical eager MLA key/value shapes differ")
    query_tokens = query_by_head.shape[1]
    total_tokens = key_by_head.shape[1]
    past_tokens = total_tokens - query_tokens
    if past_tokens < 0:
        raise ValueError("canonical eager MLA key length is shorter than query")

    scores = (
        torch.matmul(
            query_by_head.unsqueeze(0),
            key_by_head.unsqueeze(0).transpose(2, 3),
        )
        * softmax_scale
    )
    causal = torch.zeros(
        (1, 1, query_tokens, total_tokens),
        dtype=scores.dtype,
        device=scores.device,
    )
    if query_tokens > 1:
        local = torch.triu(
            torch.full(
                (query_tokens, query_tokens),
                torch.finfo(scores.dtype).min,
                dtype=scores.dtype,
                device=scores.device,
            ),
            diagonal=1,
        )
        causal[:, :, :, past_tokens:] = local
    scores = scores + causal
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
        dtype=query_by_head.dtype
    )
    context = (
        torch.matmul(probabilities, value_by_head.unsqueeze(0))
        .transpose(1, 2)
        .contiguous()
    )
    return (
        context.squeeze(0),
        scores.squeeze(0),
        probabilities.squeeze(0),
    )


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
            "AttnRes block_residual must have shape [tokens, blocks, hidden]"
        )
    if _perf_fusions_enabled() and prefix_sum.is_cuda and block_residual.shape[1] == 1:
        return kimi_k3_two_way_attn_res(
            prefix_sum,
            block_residual,
            norm_weight,
            projection_weight,
            eps,
        )
    candidates = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    candidates_float = candidates.float()
    normalized = candidates_float * torch.rsqrt(
        candidates_float.square().mean(dim=-1, keepdim=True) + eps
    )
    score_weight = norm_weight.float() * projection_weight.reshape(-1).float()
    probabilities = torch.softmax((normalized * score_weight).sum(dim=-1), dim=-1)
    return torch.einsum("tb,tbd->td", probabilities, candidates_float).to(
        dtype=prefix_sum.dtype
    )


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
    chronological order.  CUDA prefill and decode deliberately use separate
    Triton paths matching FLA's forward and cache-update kernels; CPU retains a
    Torch reference for unit tests.
    """

    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[0]:
        raise ValueError(
            "packed causal conv expects x=[tokens,channels] and "
            "weight=[channels,kernel]"
        )
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
        elif x.is_cuda and mode == "decode":
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
        elif x.is_cuda and mode == "prefill":
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
        elif mode not in ("prefill", "decode"):
            raise ValueError(f"unsupported KDA convolution mode {mode!r}")
        else:
            # CPU-only reference.  CUDA must use the kernels above because
            # separate Torch multiply/add kernels do not preserve FLA's FMA
            # and reduction semantics.
            combined = torch.cat((history, sequence.transpose(0, 1)), dim=-1)
            token_count = end - start
            weight_float = weight.float()
            output_float = (
                combined[:, :token_count].transpose(0, 1).float() * weight_float[:, 0]
            )
            for tap in range(1, kernel_size):
                output_float = output_float + (
                    combined[:, tap : tap + token_count].transpose(0, 1).float()
                    * weight_float[:, tap]
                )
            output = (output_float * torch.sigmoid(output_float)).to(dtype=x.dtype)
            final_state = (
                combined[:, -history_size:] if history_size else combined[:, :0]
            )
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
    ``[H,V,K]`` while cuLA uses ``[H,K,V]``.  Accuracy/reference paths retain
    the canonical conversion.  Cached prefill writes a state at every physical
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
        self._accuracy_full_weight_cache: dict[str, torch.Tensor] = {}
        self._segment_cu_seqlens: dict[tuple[int, int], torch.Tensor] = {}
        self.projection_size = self.local_heads * self.head_dim
        self.full_projection_size = self.total_heads * self.head_dim
        self._full_column_weights: dict[str, torch.Tensor] = {}
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
        # KDA delta-net core backend: the ported Triton kernel by default, with
        # the pure-Torch reference reachable for precision verification.
        self._kda_backend = os.environ.get("KIMI_K3_KDA_BACKEND", "kernel").lower()
        if self._kda_backend not in (
            "kernel",
            "reference",
            "fla37_precompiled",
            "flash_kda",
            "cula",
        ):
            raise ValueError(
                "KIMI_K3_KDA_BACKEND must be 'kernel', 'reference', "
                "'fla37_precompiled', 'flash_kda', or 'cula', got "
                f"{self._kda_backend!r}"
            )
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
        if _accuracy_canonical_tp_enabled():
            # Preserve the old independent-GEMM leading dimensions in the
            # diagnostic path. Production keeps zero-copy section views.
            fused_sections = tuple(section.contiguous() for section in fused_sections)
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
        self._full_column_weights.clear()
        self._a2a_weights_ready = True
        logging.info(
            "[K3_KDA_A2A] materialized %s qkvb=%s gate=%s output=%s",
            self.trace_prefix,
            tuple(self._a2a_qkvb_weight.shape),
            tuple(self._a2a_g_weight.shape),
            tuple(self._a2a_o_weight.shape),
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
        return prepare_kimi_kda_inputs(
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
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        """Return device-resident paged KDA state for the batched decode path.

        The optimized path is deliberately isolated behind an environment flag
        and falls back to the canonical gather/compute/scatter implementation
        whenever its one-token CUDA contract is not satisfied.
        """

        if (
            not _batched_kda_decode_enabled()
            or mode != "decode"
            or kv_cache is None
            or attention_inputs is None
            or not hidden_states.is_cuda
            or _accuracy_trace_enabled()
            or _accuracy_canonical_tp_enabled()
            or self._kda_backend not in ("kernel", "flash_kda")
            or bool(getattr(attention_inputs, "is_target_verify", False))
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
            or sequence_lengths_plus_one.numel() != hidden_states.shape[0]
            or block_map.shape[0] != hidden_states.shape[0]
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
        head_shape = (1, token_count, self.local_heads, self.head_dim)
        output, _ = fused_recurrent_kda(
            q.reshape(head_shape),
            k.reshape(head_shape),
            v.reshape(head_shape),
            raw_gate.reshape(head_shape),
            raw_beta.float().reshape(1, token_count, self.local_heads),
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
        return output.reshape(head_shape).to(dtype=q.dtype)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the KDA delta-net core (l2norm + decay gate + scan).

        Drop-in for the ``kimi_kda`` reference: same ``[1,T,H,*]`` I/O and a
        ``[N,H,K,V]`` fp32 final state.  Defaults to the ported Triton kernel
        (``chunk_kda``/``fused_recurrent_kda``); the
        pure-Torch reference stays reachable via ``KIMI_K3_KDA_BACKEND=reference``
        for precision verification. Kernel/reference agreement covers the
        non-zero initial-state prefill/decode seams.
        """

        if checkpoint_interval is None and checkpoint_states is not None:
            raise ValueError("checkpoint_states requires checkpoint_interval")
        if checkpoint_interval is not None and (
            mode != "prefill" or self._kda_backend != "cula"
        ):
            raise ValueError(
                "FP32 checkpoint states are supported only by cuLA prefill"
            )

        if self._kda_backend == "reference" or not q.is_cuda:
            return kimi_kda(
                q,
                k,
                v,
                raw_gate,
                raw_beta,
                a_log,
                dt_bias,
                recurrent_state,
                mode=mode,
                lower_bound=self.gate_lower_bound,
                cu_seqlens=cu_seqlens,
            )

        copy_free_backend_prefill = (
            _perf_fusions_enabled()
            and self._kda_backend == "cula"
            and mode == "prefill"
        )
        if copy_free_backend_prefill:
            state_in = recurrent_state
            if state_in is not None and (
                state_in.dtype != torch.float32 or not state_in.is_contiguous()
            ):
                raise ValueError(
                    "K3 fused state must be contiguous FP32 in the "
                    f"{self._kda_backend} native layout"
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
            if self._kda_backend == "cula":
                try:
                    import cula
                    from cula.kda import chunk_kda as cula_chunk_kda
                except Exception as error:
                    raise RuntimeError(
                        "KIMI_K3_KDA_BACKEND=cula was requested but the "
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

            if self._kda_backend == "flash_kda":
                try:
                    import flash_kda
                    import flash_kda_C
                except Exception as error:
                    raise RuntimeError(
                        "KIMI_K3_KDA_BACKEND=flash_kda was requested but the "
                        "FlashKDA extension could not be imported"
                    ) from error
                required_api = ("fwd", "get_workspace_size")
                missing = [
                    name for name in required_api if not hasattr(flash_kda, name)
                ]
                if missing:
                    raise RuntimeError(
                        "FlashKDA overlay is missing required API: "
                        + ", ".join(missing)
                    )
                if self.gate_lower_bound is None:
                    raise RuntimeError("FlashKDA requires K3's finite gate lower bound")

                state_v_first = (
                    None
                    if state_in is None
                    else state_in.transpose(-1, -2).contiguous()
                )
                sequence_count = int(cu_seqlens.numel() - 1)
                final_state_v_first = torch.empty(
                    (
                        sequence_count,
                        self.local_heads,
                        self.head_dim,
                        self.head_dim,
                    ),
                    dtype=(
                        state_v_first.dtype
                        if state_v_first is not None
                        else torch.float32
                    ),
                    device=q.device,
                )
                output = torch.empty_like(v) if output_target is None else output_target
                if tuple(output.shape) != tuple(v.shape) or not output.is_contiguous():
                    raise ValueError(
                        "FlashKDA output target must be contiguous and match V: "
                        f"output={tuple(output.shape)} v={tuple(v.shape)}"
                    )
                workspace = _flash_kda_workspace(flash_kda, q, cu_seqlens)
                device_index = q.device.index if q.device.index is not None else 0
                if device_index not in _FLASH_KDA_LOGGED_DEVICES:
                    logging.info(
                        "[KimiK3 FlashKDA] enabled device=%s python=%s extension=%s "
                        "workspace_bytes=%d",
                        q.device,
                        getattr(flash_kda, "__file__", "<unknown>"),
                        getattr(flash_kda_C, "__file__", "<unknown>"),
                        workspace.numel(),
                    )
                    _FLASH_KDA_LOGGED_DEVICES.add(device_index)
                with torch.autograd.profiler.record_function("k3.kda.flashkda"):
                    flash_kda.fwd(
                        q.contiguous(),
                        k.contiguous(),
                        v.contiguous(),
                        raw_gate.to(dtype=q.dtype).contiguous(),
                        raw_beta.to(dtype=q.dtype).contiguous(),
                        self.head_dim**-0.5,
                        output,
                        a_log.float().contiguous(),
                        dt_bias.float()
                        .reshape(self.local_heads, self.head_dim)
                        .contiguous(),
                        float(self.gate_lower_bound),
                        initial_state=state_v_first,
                        final_state=final_state_v_first,
                        cu_seqlens=cu_seqlens.contiguous(),
                        workspace=workspace,
                    )
                final_state = final_state_v_first.transpose(-1, -2).contiguous()
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
            if self._kda_backend == "flash_kda":
                raise RuntimeError(
                    "KIMI_K3_KDA_BACKEND=flash_kda only implements Prefill; "
                    "select kernel or fla37_precompiled for Decode"
                )
            # Match Dummy's recurrent call boundary: gate activation, beta
            # sigmoid, V-first register layout and state update all stay in
            # one Triton program.  Precomputing gate/beta or transposing the
            # register tile changes enough FP32 state ULPs to accumulate into
            # a BF16 hidden-state difference after several decode steps.
            state_v_first = (
                None if state_in is None else state_in.transpose(-1, -2).contiguous()
            )
            if self._kda_backend == "fla37_precompiled":
                output, final_state_v_first = fused_recurrent_kda_fla37_precompiled(
                    q,
                    k,
                    v,
                    raw_gate,
                    raw_beta.float(),
                    initial_state=state_v_first,
                    A_log=a_log,
                    dt_bias=dt_bias,
                    cu_seqlens=cu_seqlens,
                    lower_bound=self.gate_lower_bound,
                )
            else:
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
        # Normalize to the reference's [1,T,H,V] / q.dtype output contract.
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
            and self._kda_backend == "cula"
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
            while cursor < end:
                tokens_to_page_end = page_size - (absolute_position % page_size)
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
                        mode="prefill",
                        use_initial_state=absolute_position > 0,
                        sequence_ranges=segment_ranges,
                    )
                    k, k_state = _packed_causal_depthwise_conv1d(
                        k_projected[cursor:segment_end],
                        self.kda_k_conv,
                        segment_cu_seqlens,
                        k_state,
                        mode="prefill",
                        use_initial_state=absolute_position > 0,
                        sequence_ranges=segment_ranges,
                    )
                    v, v_state = _packed_causal_depthwise_conv1d(
                        v_projected[cursor:segment_end],
                        self.kda_v_conv,
                        segment_cu_seqlens,
                        v_state,
                        mode="prefill",
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
                    and self._kda_backend in ("flash_kda", "cula")
                ):
                    beta_for_core = beta_for_core.float()
                with _perf_profile(
                    f"{page_prefix}.{self._kda_backend}_recurrence_and_output"
                ):
                    segment_cu_seqlens_cpu = None
                    if self._kda_backend == "cula":
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
                        mode="prefill",
                        cu_seqlens=segment_cu_seqlens,
                        cu_seqlens_cpu=segment_cu_seqlens_cpu,
                        output_target=(
                            None
                            if fused_output is None
                            else fused_output[:, cursor:segment_end]
                        ),
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
        local_token_count = hidden_states.shape[0]
        token_count = hidden_states.shape[0]
        if _accuracy_canonical_tp_enabled():
            with _perf_profile(
                f"{self.trace_prefix}.canonical_qkvg_column_parallel_projections",
                hidden_states,
            ):
                q_projected = _column_parallel_linear(
                    hidden_states,
                    self.kda_q_w,
                    self.attn_tp_size,
                    self.attn_tp_rank,
                    self._full_column_weights,
                    "q",
                )
                k_projected = _column_parallel_linear(
                    hidden_states,
                    self.kda_k_w,
                    self.attn_tp_size,
                    self.attn_tp_rank,
                    self._full_column_weights,
                    "k",
                )
                v_projected = _column_parallel_linear(
                    hidden_states,
                    self.kda_v_w,
                    self.attn_tp_size,
                    self.attn_tp_rank,
                    self._full_column_weights,
                    "v",
                )
                output_gate_projected = _column_parallel_linear(
                    hidden_states,
                    self.kda_g_w,
                    self.attn_tp_size,
                    self.attn_tp_rank,
                    self._full_column_weights,
                    "output_gate",
                )
            with _perf_profile(
                f"{self.trace_prefix}.canonical_forget_gate_and_beta_projections",
                hidden_states,
            ):
                forget_latent = _linear(hidden_states, self.kda_f_a_w)
                raw_gate = _column_parallel_linear(
                    forget_latent,
                    self.weights[W.linear_attn_f_b_w],
                    self.attn_tp_size,
                    self.attn_tp_rank,
                    self._full_column_weights,
                    "forget_gate_up",
                )
                full_raw_beta = _linear(hidden_states, self.kda_beta_w)
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
                raw_gate = _column_parallel_linear(
                    forget_latent,
                    self.weights[W.linear_attn_f_b_w],
                    self.attn_tp_size,
                    self.attn_tp_rank,
                    self._full_column_weights,
                    "forget_gate_up",
                )
        beta_begin = self.attn_tp_rank * self.local_heads
        raw_beta = full_raw_beta.narrow(1, beta_begin, self.local_heads)
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
        stored_page_states = mode == "prefill" and kv_cache is not None
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
                output = _row_parallel_linear(
                    output.reshape(token_count, self.projection_size),
                    self.weights[W.linear_attn_out_w],
                    self.parallelism_config.get_attn_tp_size(),
                    reduce_scatter_tokens=(
                        sequence_parallel
                        and self.attn_tp_size > 1
                        and hidden_states.is_cuda
                    ),
                    pad_reduce_scatter_tokens=(
                        sequence_parallel
                        and mode == "decode"
                        and self.attn_tp_size > 1
                        and hidden_states.is_cuda
                    ),
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

    The production path reuses the framework MLA kernels and executes one
    packed Q-A/KV-A/output-gate projection. Accuracy and reference paths keep
    the source model's independent projection boundaries. The pure-Torch
    attention implementation remains selectable with
    ``KIMI_K3_MLA_BACKEND=reference``.
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
        self._perf_profile_prefix = self.trace_prefix if _perf_mode_enabled() else None
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
        self._accuracy_full_weight_cache: dict[str, torch.Tensor] = {}
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
        self._mla_backend = os.environ.get("KIMI_K3_MLA_BACKEND", "kernel").lower()
        if self._mla_backend not in ("kernel", "flashmla", "reference"):
            raise ValueError(
                "KIMI_K3_MLA_BACKEND must be 'kernel', 'flashmla' or "
                f"'reference', "
                f"got {self._mla_backend!r}"
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

    def _use_source_projection_boundaries(self) -> bool:
        return (
            self._mla_backend == "reference"
            or not _perf_fusions_enabled()
            or _accuracy_canonical_tp_enabled()
            or _accuracy_canonical_mla_enabled()
            or _accuracy_local_eager_mla_enabled()
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
            output_gate = _column_parallel_linear(
                hidden_states,
                self._packed_qkv_gate_w[:, kv_end:],
                self.attn_tp_size,
                self.attn_tp_rank,
                self._accuracy_full_weight_cache,
                "mla_output_gate",
            )
        return torch.cat((q_a, kv_a), dim=-1), output_gate

    def _project_qkv_a_input(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._use_source_projection_boundaries():
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
        wise per (head, value) exactly as the reference path does before o_proj.
        This runs before o_proj's TP all_reduce, so each rank gates only its
        local heads.
        """
        if not self.use_output_gate:
            return attn_output
        assert output_gate is not None
        return attn_output * torch.sigmoid(output_gate.reshape_as(attn_output))

    def _project_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        if self._sp_active_for_forward:
            return _row_parallel_linear(
                attn_output,
                self._o_w,
                self.parallelism_config.get_attn_tp_size(),
                reduce_scatter_tokens=True,
                pad_reduce_scatter_tokens=self._sp_padded_for_forward,
            )
        return super()._project_output(attn_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        sequence_parallel: bool = False,
    ) -> torch.Tensor:
        attn_inputs = _select_mla_attention_inputs(attention_inputs, fmha_impl)
        self._sp_active_for_forward = bool(
            sequence_parallel
            and self.parallelism_config.get_attn_tp_size() > 1
            and hidden_states.is_cuda
            and attn_inputs is not None
        )
        self._sp_padded_for_forward = bool(
            self._sp_active_for_forward
            and attn_inputs is not None
            and not attn_inputs.is_prefill
        )
        if self._sp_active_for_forward and _accuracy_canonical_tp_enabled():
            self._sp_active_for_forward = False
            raise RuntimeError(
                "Kimi K3 Sequence Parallel is incompatible with canonical TP; "
                "disable one of KIMI_K3_SP_MOE and "
                "KIMI_K3_ACCURACY_CANONICAL_TP"
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

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
    ) -> torch.Tensor:
        if self._mla_backend == "reference" or not hidden_states.is_cuda:
            # ``PyModelInputs.attention_inputs`` and the instance captured by
            # the shared FMHA wrapper can be distinct pybind views.  K3 updates
            # the explicit view's singular block table for every HybridCache
            # group, while the wrapper-captured view may still point at group
            # zero.  Always prefer the layer-current explicit view.
            attn_inputs = _select_mla_attention_inputs(attention_inputs, fmha_impl)
            if attn_inputs is None:
                raise ValueError("MLA reference path requires PyAttentionInputs")
            is_prefill = bool(attn_inputs.is_prefill)
            fused_qkv, output_gate = self._project_qkv_a_input(hidden_states)
            cu_seqlens = self._reference_cu_seqlens(
                attn_inputs, fused_qkv.shape[0], is_prefill, fused_qkv.device
            )
            return self._reference_forward(
                fused_qkv,
                output_gate,
                cu_seqlens,
                is_prefill=is_prefill,
                kv_cache=kv_cache,
                attention_inputs=attn_inputs,
            )
        canonical_mla = _accuracy_canonical_mla_enabled()
        local_eager_mla = _accuracy_local_eager_mla_enabled()
        if accuracy_trace_mode() is None and not canonical_mla and not local_eager_mla:
            return super().forward(hidden_states, fmha_impl, kv_cache)

        # Trace the real MLA kernel/cache path, or run the explicitly enabled
        # canonical-accuracy math without requiring tensor trace persistence.
        # The regular trace remains O(T) and recomputes only the final
        # score/probability row.  Canonical mode additionally materializes
        # Dummy's eager O(T^2) attention so source and distributed accumulation
        # can be compared without changing the default production path.
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
        query_projection = (
            _column_parallel_linear(
                query_latent,
                self._q_b_w,
                self.parallelism_config.get_attn_tp_size(),
                self.attn_tp_rank,
                self._accuracy_full_weight_cache,
                "mla_q_b",
            )
            if canonical_mla or local_eager_mla
            else self.q_b_proj(query_latent)
        )
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
        cu_seqlens = self._reference_cu_seqlens(
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
        expanded_projection = (
            _column_parallel_linear(
                expanded_input,
                self._kv_b_w,
                self.parallelism_config.get_attn_tp_size(),
                self.attn_tp_rank,
                self._accuracy_full_weight_cache,
                "mla_kv_b",
            )
            if canonical_mla or local_eager_mla
            else _linear(expanded_input, self._kv_b_w)
        )
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
        if canonical_mla:
            context_by_head, scores, probabilities = self._canonical_eager_context(
                query, key, value
            )
            context = context_by_head.reshape(fused_qkv.shape[0], -1).contiguous()
        elif local_eager_mla:
            context_by_head, scores, probabilities = _source_eager_attention_context(
                query.transpose(0, 1).contiguous(),
                key.transpose(0, 1).contiguous(),
                value.transpose(0, 1).contiguous(),
                self.softmax_scale,
            )
            context = context_by_head.reshape(fused_qkv.shape[0], -1).contiguous()
        else:
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
        if canonical_mla or local_eager_mla:
            output = _row_parallel_linear(
                context,
                self._o_w,
                self.parallelism_config.get_attn_tp_size(),
                reduce_scatter_tokens=self._sp_active_for_forward,
                pad_reduce_scatter_tokens=self._sp_padded_for_forward,
            )
        else:
            output = self._project_output(context)
        record_accuracy_tensor(f"{self.trace_prefix}.output", output, token_dim=0)
        return output

    def _canonical_eager_context(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_by_head = query.transpose(0, 1).contiguous()
        key_by_head = key.transpose(0, 1).contiguous()
        value_by_head = value.transpose(0, 1).contiguous()
        tp_size = self.parallelism_config.get_attn_tp_size()
        if tp_size > 1:
            query_by_head = all_gather(query_by_head, group=Group.TP)
            key_by_head = all_gather(key_by_head, group=Group.TP)
            value_by_head = all_gather(value_by_head, group=Group.TP)
        context, scores, probabilities = _source_eager_attention_context(
            query_by_head,
            key_by_head,
            value_by_head,
            self.softmax_scale,
        )
        begin = self.attn_tp_rank * self.local_heads
        end = begin + self.local_heads
        return (
            context[:, begin:end].contiguous(),
            scores[begin:end, -1:, :].contiguous(),
            probabilities[begin:end, -1:, :].contiguous(),
        )

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
    def _reference_cu_seqlens(
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

    def _reference_forward(
        self,
        fused_qkv: torch.Tensor,
        output_gate: Optional[torch.Tensor],
        cu_seqlens: torch.Tensor,
        *,
        is_prefill: bool,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
    ) -> torch.Tensor:
        token_count = fused_qkv.shape[0]
        ranges = _sequence_offsets(cu_seqlens, token_count)
        if attention_inputs is None:
            if not is_prefill or kv_cache is not None:
                raise ValueError("MLA cache/decode requires PyAttentionInputs")
            past_lengths = [0] * len(ranges)
        else:
            length_tensor = (
                attention_inputs.prefix_lengths
                if is_prefill
                else attention_inputs.sequence_lengths
            )
            if length_tensor is None or length_tensor.numel() == 0:
                past_lengths = [0] * len(ranges)
            else:
                past_lengths = [
                    int(value) for value in length_tensor.detach().cpu().tolist()
                ]
            if len(past_lengths) != len(ranges):
                raise ValueError("MLA cache batch does not match packed sequences")
        if (not is_prefill or any(past_lengths)) and kv_cache is None:
            raise RuntimeError("MLA decode/prefix reuse requires a LayerKVCache")

        q_a, compressed = torch.split(
            fused_qkv,
            [
                self.q_lora_rank,
                self.kv_lora_rank + self.suffix_dim,
            ],
            dim=-1,
        )
        query_latent = _rms_norm(q_a.contiguous(), self._q_a_norm, self.eps)
        query = _linear(query_latent, self._q_b_w).reshape(
            token_count, self.local_heads, self.q_head_dim
        )
        record_accuracy_tensor(
            f"{self.trace_prefix}.query_latent", query_latent, token_dim=0
        )
        record_accuracy_tensor(f"{self.trace_prefix}.query", query, token_dim=0)
        compressed_kv, key_suffix = torch.split(
            compressed, [self.kv_lora_rank, self.suffix_dim], dim=-1
        )
        compressed_kv = _rms_norm(compressed_kv, self._kv_a_norm, self.eps)
        canonical_current = torch.cat((compressed_kv, key_suffix), dim=-1)
        record_accuracy_tensor(
            f"{self.trace_prefix}.compressed_current",
            canonical_current,
            token_dim=0,
        )
        cache_view: Optional[torch.Tensor] = None
        block_map: Optional[list[list[int]]] = None
        page_size = 0
        if kv_cache is not None:
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
                    raise ValueError(
                        "MLA cache storage is not divisible by latent width"
                    )
                cache_view = base.reshape(base.shape[0], -1, width)
            if cache_view.shape[1] < page_size:
                raise ValueError("MLA cache block is smaller than seq_size_per_block")
            block_map = KimiK3LinearCacheAdapter._block_map(attention_inputs)
        is_fake_stream = (
            attention_inputs is not None
            and KimiK3LinearCacheAdapter._is_fake_stream(attention_inputs)
        )

        def read_prefix(sequence_idx: int, length: int) -> torch.Tensor:
            if length == 0:
                return compressed_kv.new_empty(0, self.kv_lora_rank + self.suffix_dim)
            assert cache_view is not None and block_map is not None
            cached_tokens = []
            for position in range(length):
                block_id = KimiK3LinearCacheAdapter._block_id(
                    block_map, sequence_idx, position, page_size
                )
                cached_tokens.append(cache_view[block_id, position % page_size])
            return torch.stack(cached_tokens)

        def write_tokens(
            sequence_idx: int,
            prefix_length: int,
            values: torch.Tensor,
        ) -> None:
            if cache_view is None:
                return
            assert block_map is not None
            if is_fake_stream or KimiK3LinearCacheAdapter._is_fake_block_row(
                block_map[sequence_idx]
            ):
                return
            for token_idx in range(values.shape[0]):
                position = prefix_length + token_idx
                block_id = KimiK3LinearCacheAdapter._block_id(
                    block_map, sequence_idx, position, page_size
                )
                cache_view[block_id, position % page_size].copy_(
                    values[token_idx].to(dtype=cache_view.dtype)
                )

        outputs: list[torch.Tensor] = []
        canonical_cache_input: list[torch.Tensor] = []
        canonical_cache: list[torch.Tensor] = []
        trace_mode = accuracy_trace_mode()
        trace_scores: list[torch.Tensor] = []
        trace_probabilities: list[torch.Tensor] = []
        trace_context: list[torch.Tensor] = []
        for sequence_idx, ((start, end), prefix_length) in enumerate(
            zip(ranges, past_lengths)
        ):
            if start == end:
                outputs.append(
                    fused_qkv.new_empty((0, self.local_heads, self.value_dim))
                )
                continue
            fake_cache_row = is_fake_stream or (
                block_map is not None
                and KimiK3LinearCacheAdapter._is_fake_block_row(block_map[sequence_idx])
            )
            effective_prefix_length = 0 if fake_cache_row else prefix_length
            current_compressed = torch.cat(
                (compressed_kv[start:end], key_suffix[start:end]), dim=-1
            )
            cached = read_prefix(sequence_idx, effective_prefix_length)
            canonical_cache_input.append(cached)
            all_compressed = torch.cat((cached, current_compressed), dim=0)
            canonical_cache.append(all_compressed)
            all_latent, all_suffix = torch.split(
                all_compressed, [self.kv_lora_rank, self.suffix_dim], dim=-1
            )
            all_latent = all_latent.to(dtype=self._kv_b_w.dtype)
            expanded = _linear(all_latent, self._kv_b_w).reshape(
                all_latent.shape[0],
                self.local_heads,
                self.nope_dim + self.value_dim,
            )
            key_nope, value = torch.split(
                expanded, [self.nope_dim, self.value_dim], dim=-1
            )
            expanded_suffix = all_suffix.unsqueeze(1).expand(-1, self.local_heads, -1)
            key = torch.cat((key_nope, expanded_suffix), dim=-1)
            scores = (
                torch.einsum("thd,shd->hts", query[start:end], key) * self.softmax_scale
            )
            query_positions = effective_prefix_length + torch.arange(
                end - start, device=fused_qkv.device
            )
            key_positions = torch.arange(
                effective_prefix_length + end - start,
                device=fused_qkv.device,
            )
            causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            scores = scores.masked_fill(
                ~causal_mask.unsqueeze(0), torch.finfo(scores.dtype).min
            )
            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
                dtype=query.dtype
            )
            context = torch.einsum("hts,shv->thv", probabilities, value).to(
                dtype=fused_qkv.dtype
            )
            outputs.append(context)
            if trace_mode is not None:
                record_accuracy_tensor(
                    f"{self.trace_prefix}.scores_last_query",
                    scores[:, -1:, :],
                )
                record_accuracy_tensor(
                    f"{self.trace_prefix}.probabilities_last_query",
                    probabilities[:, -1:, :],
                )
                record_accuracy_tensor(
                    f"{self.trace_prefix}.context_last_query",
                    context[-1:, :, :],
                    token_dim=0,
                )
                if trace_mode == "semantic_full" and end - start <= 256:
                    trace_scores.append(scores)
                    trace_probabilities.append(probabilities)
                    trace_context.append(context)
            write_tokens(sequence_idx, effective_prefix_length, current_compressed)
        output = torch.cat(outputs, dim=0)
        if trace_scores:
            if len(trace_scores) != 1:
                raise RuntimeError(
                    "quadratic K3 accuracy trace currently requires one sequence"
                )
            record_accuracy_tensor(
                f"{self.trace_prefix}.scores", trace_scores[0], token_dim=1
            )
            record_accuracy_tensor(
                f"{self.trace_prefix}.probabilities",
                trace_probabilities[0],
                token_dim=1,
            )
            record_accuracy_tensor(
                f"{self.trace_prefix}.context", trace_context[0], token_dim=0
            )
        if not is_prefill and canonical_cache_input:
            record_accuracy_tensor(
                f"{self.trace_prefix}.cache_input",
                torch.cat(canonical_cache_input, dim=0),
            )
        if canonical_cache:
            record_accuracy_tensor(
                f"{self.trace_prefix}.cache", torch.cat(canonical_cache, dim=0)
            )
        if self.use_output_gate:
            assert output_gate is not None
            output_gate = output_gate.reshape_as(output)
            record_accuracy_tensor(
                f"{self.trace_prefix}.output_gate", output_gate, token_dim=0
            )
            output = output * torch.sigmoid(output_gate)
        output = _row_parallel_linear(
            output.reshape(token_count, -1),
            self._o_w,
            self.parallelism_config.get_attn_tp_size(),
            reduce_scatter_tokens=self._sp_active_for_forward,
            pad_reduce_scatter_tokens=self._sp_padded_for_forward,
        )
        record_accuracy_tensor(f"{self.trace_prefix}.output", output, token_dim=0)
        return output


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
        if sp_active:
            with _perf_profile(
                f"{self.trace_prefix}.replicated_gate_and_up_gemm", hidden_states
            ):
                gate = _linear(hidden_states, gate_weight)
                up = _linear(hidden_states, up_weight)
        else:
            gate = _column_parallel_linear(
                hidden_states,
                gate_weight,
                self.ffn_tp_size,
                self.ffn_tp_rank,
                self._full_column_weights,
                "gate",
            )
            up = _column_parallel_linear(
                hidden_states,
                up_weight,
                self.ffn_tp_size,
                self.ffn_tp_rank,
                self._full_column_weights,
                "up",
            )
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
        self._moe_backend = (
            os.environ.get("KIMI_K3_MOE_BACKEND", "deepep").strip().lower()
        )
        if self._moe_backend not in ("deepep", "deep_gemm_mega"):
            raise ValueError(
                "KIMI_K3_MOE_BACKEND must be 'deepep' or "
                f"'deep_gemm_mega', got {self._moe_backend!r}"
            )
        if self._moe_backend == "deep_gemm_mega":
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
                "KIMI_K3_MOE_BACKEND=deep_gemm_mega resolved an old DeepGEMM "
                "without K3 SiTU support; missing parameters: "
                + ", ".join(sorted(missing_parameters))
            )
        expected_root = os.environ.get("KIMI_K3_DEEPGEMM_EXPECTED_PATH")
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
        if _accuracy_canonical_ep_enabled():
            # The checkpoint owns a contiguous [experts, hidden] nn.Linear
            # weight. RTP stores its contiguous transpose [hidden, experts].
            # FP32 GEMM accumulation on SM10x is observably layout-sensitive;
            # restore the source layout for the default-off accuracy path.
            router_logits = F.linear(
                hidden_states.float(),
                router_weight.transpose(0, 1).contiguous().float(),
            )
        else:
            router_logits = _linear(hidden_states.float(), router_weight.float())
        scores = torch.sigmoid(router_logits)
        choice_scores = scores + self.weights[
            K3W.MOE_CORRECTION_BIAS
        ].float().unsqueeze(0)
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
            router_token_dim = None if _accuracy_full_router_trace_enabled() else 0
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

    def _canonical_ep_collective(
        self,
        routed_input: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """NCCL correctness fallback when CUDA13 has no DeepEP provider.

        Prefill TP8/DP1 carries a replicated token matrix, so every rank
        evaluates its local experts and an all-reduce reconstructs each top-k
        slot.  Decode TP1/DP8 carries one independent real/fake stream per
        rank; gather those streams first, evaluate every rank's local experts,
        reconstruct the slots, then return this rank's original rows.
        """

        group = Group.DP_AND_TP
        if self.attn_tp_size == self.ep_size:
            local_slots = self._local_expert_slots(
                routed_input,
                expert_ids,
                ids_are_local=False,
            )
            expert_slots = all_reduce(local_slots, group=group)
            return self._reduce_expert_slots(expert_slots, routing_weights)

        if self.attn_tp_size == 1:
            local_token_count = routed_input.shape[0]
            gathered_input = all_gather(routed_input.contiguous(), group=group)
            gathered_ids = all_gather(expert_ids.contiguous(), group=group)
            gathered_weights = all_gather(routing_weights.contiguous(), group=group)
            local_slots = self._local_expert_slots(
                gathered_input,
                gathered_ids,
                ids_are_local=False,
            )
            expert_slots = all_reduce(local_slots, group=group)
            gathered_output = self._reduce_expert_slots(
                expert_slots,
                gathered_weights,
            )
            begin = self.ep_rank * local_token_count
            return gathered_output.narrow(0, begin, local_token_count)

        raise RuntimeError(
            "Kimi K3 canonical NCCL EP fallback currently supports only "
            "TP=EP/DP1 prefill or TP1/DP=EP decode"
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

    def _deepep_normal_canonical_slots(
        self,
        wrapper: Any,
        routed_input: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Preserve Dummy's ordered top-k reduction through DeepEP.

        Normal DeepEP combines expert contributions by destination rank and
        K3 additionally splits top-k 16 into supported widths 10 + 6.  Both
        regroup the FP32 additions relative to Dummy's single ordered
        ``sum(dim=topk)``.  For accuracy tracing, dispatch one real slot at a
        time (padded to DeepEP's width four), combine its unweighted BF16
        expert output, then apply all router weights and the ordered FP32 sum
        on the source rank.  Expert ownership and the actual DeepEP transport
        remain exercised; only the reduction schedule is canonicalized.
        """

        if expert_ids.shape != routing_weights.shape:
            raise ValueError("expert ids and routing weights must have equal shape")
        buffer = wrapper.buffer
        dispatch_input = self._pad_dispatch_payload(routed_input)
        slot_outputs: list[torch.Tensor] = []
        for slot_idx in range(expert_ids.shape[1]):
            slot_ids = torch.full(
                (expert_ids.shape[0], 4),
                -1,
                dtype=expert_ids.dtype,
                device=expert_ids.device,
            )
            slot_ids[:, 0] = expert_ids[:, slot_idx]
            unit_weights = torch.zeros(
                (expert_ids.shape[0], 4),
                dtype=routing_weights.dtype,
                device=routing_weights.device,
            )
            unit_weights[:, 0] = 1
            (
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                num_tokens_per_expert,
                is_token_in_rank,
                _,
            ) = buffer.get_dispatch_layout(slot_ids, self.expert_num)
            (
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
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
                slot_ids,
                unit_weights,
                expert_alignment=1,
            )
            if not isinstance(recv_x, torch.Tensor):
                raise RuntimeError(
                    "Kimi K3 canonical DeepEP path requires BF16 dispatch"
                )
            local_latent = self._local_expert_sum(
                recv_x[:, : self.latent_size],
                recv_topk_idx,
                recv_topk_weights,
                ids_are_local=True,
            )
            combined, _, _ = buffer.combine(
                self._pad_dispatch_payload(local_latent),
                handle,
            )
            slot_outputs.append(
                combined[:, : self.latent_size].to(dtype=routed_input.dtype)
            )

        expert_slots = torch.stack(slot_outputs, dim=1)
        return (
            expert_slots.to(dtype=routing_weights.dtype)
            .mul(routing_weights.unsqueeze(-1))
            .sum(dim=1)
            .to(dtype=routed_input.dtype)
        )

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
        if self._moe_backend == "deep_gemm_mega":
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

        canonical_slots = _accuracy_canonical_ep_enabled()
        if sequence_parallel and canonical_slots and not DeepEPWrapper.supported():
            raise RuntimeError(
                "Kimi K3 Sequence Parallel requires DeepEP when canonical EP "
                "is enabled; the NCCL replicated-token fallback is incompatible"
            )
        if canonical_slots and not DeepEPWrapper.supported():
            if not getattr(self, "_canonical_ep_fallback_logged", False):
                logging.warning(
                    "Kimi K3 canonical EP is using NCCL correctness fallback "
                    "because DeepEP is unavailable"
                )
                self._canonical_ep_fallback_logged = True
            return self._canonical_ep_collective(
                routed_input,
                expert_ids,
                routing_weights,
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
            # Every EP rank must execute the same dispatch/combine schedule.
            # Decode DP uses collective-only fake streams on the non-owning
            # ranks, and those streams intentionally disable tensor capture.
            # Tying this choice to ``accuracy_trace_mode()`` therefore made the
            # owning rank run 16 slot-wise collectives while its peers ran the
            # ordinary 10+6 schedule.  The process-wide accuracy switch is the
            # only valid selector; tracing controls persistence, not math.
            local_output = (
                self._deepep_normal_canonical_slots(
                    wrapper, sliced_input, sliced_ids, sliced_weights
                )
                if canonical_slots
                else self._deepep_normal(
                    wrapper, sliced_input, sliced_ids, sliced_weights
                )
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
                expert_ids[valid_token_count:] = -1
                routing_weights[valid_token_count:] = 0
        if trace_enabled:
            router_token_dim = None if _accuracy_full_router_trace_enabled() else 0
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
        if sp_active:
            with _perf_profile(
                f"{self.trace_prefix}.shared_expert_replicated_gate_up_gemm",
                hidden_states,
            ):
                shared_gate = _linear(hidden_states, shared_gate_weight)
                shared_up = _linear(hidden_states, shared_up_weight)
        else:
            shared_gate = _column_parallel_linear(
                hidden_states,
                shared_gate_weight,
                self.ffn_tp_size,
                self.ffn_tp_rank,
                self._full_column_weights,
                "shared_gate",
            )
            shared_up = _column_parallel_linear(
                hidden_states,
                shared_up_weight,
                self.ffn_tp_size,
                self.ffn_tp_rank,
                self._full_column_weights,
                "shared_up",
            )
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trace_prefix = f"layer.{self.layer_idx}"
        trace_enabled = _accuracy_trace_enabled()
        decode_sp = sequence_parallel and mode == "decode"
        decode_sp_debug = decode_sp and _decode_sp_debug_enabled()
        logical_tokens = int(hidden_states.shape[0])
        tp_size = int(self.self_attn.parallelism_config.get_attn_tp_size())
        tp_rank = int(self.self_attn.parallelism_config.get_attn_tp_rank())
        local_valid_tokens: Optional[int] = None
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
        if block_residual.shape[1] > 0:
            with _perf_profile(f"{trace_prefix}.self_attn_residual_mix", prefix_sum):
                hidden_states = _attention_residual(
                    prefix_sum,
                    block_residual,
                    self.weights[K3W.SELF_ATTN_RES_NORM],
                    self.weights[K3W.SELF_ATTN_RES_PROJ],
                    self.eps,
                )
        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
            prefix_sum = None
        with _perf_profile(f"{trace_prefix}.attention_input_rmsnorm", hidden_states):
            attention_input = _rms_norm(
                hidden_states, self.weights[W.pre_ln_gamma], self.eps
            )
        if trace_enabled:
            record_accuracy_tensor(
                f"{trace_prefix}.attention_input", attention_input, token_dim=0
            )
        kda_a2a_prefill = (
            sequence_parallel
            and mode == "prefill"
            and self.is_kda
            and isinstance(self.self_attn, KimiK3KDA)
            and self.self_attn.uses_a2a_comm
        )
        if sequence_parallel and mode == "prefill" and not kda_a2a_prefill:
            with _perf_profile(
                f"{trace_prefix}.attention_input_token_allgather_TP8",
                attention_input,
            ):
                attention_input = all_gather(attention_input, group=Group.TP)
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
                )
        if decode_sp_debug:
            logging.info(
                "[K3_DECODE_SP_DEBUG] rank=%d layer=%d attention_done output=%s",
                tp_rank,
                self.layer_idx,
                tuple(attention_output.shape),
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
            block_residual, block_valid_tokens = _padded_token_shard(
                block_residual,
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
        if decode_sp_debug:
            logging.info(
                "[K3_DECODE_SP_DEBUG] rank=%d layer=%d residual_shard_done valid=%s",
                tp_rank,
                self.layer_idx,
                local_valid_tokens,
            )
        with _perf_profile(f"{trace_prefix}.attention_prefix_sum", attention_output):
            prefix_sum = (
                attention_output
                if prefix_sum is None
                else prefix_sum + attention_output
            )
        with _perf_profile(f"{trace_prefix}.mlp_attn_residual_mix", prefix_sum):
            mlp_input = _attention_residual(
                prefix_sum,
                block_residual,
                self.weights[K3W.MLP_RES_NORM],
                self.weights[K3W.MLP_RES_PROJ],
                self.eps,
            )
        if trace_enabled:
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
        if decode_sp_debug:
            logging.info(
                "[K3_DECODE_SP_DEBUG] rank=%d layer=%d mlp_done output=%s",
                tp_rank,
                self.layer_idx,
                tuple(mlp_output.shape),
            )
        with _perf_profile(f"{trace_prefix}.residual_add", mlp_output):
            output = prefix_sum + mlp_output
        if decode_sp:
            with _perf_profile(
                f"{trace_prefix}.decode_token_allgather_trim_TP8",
                output,
            ):
                output = all_gather_trim(output, logical_tokens, group=Group.TP)
                block_residual = all_gather_trim(
                    block_residual,
                    logical_tokens,
                    group=Group.TP,
                )
        if decode_sp_debug:
            logging.info(
                "[K3_DECODE_SP_DEBUG] rank=%d layer=%d exit output=%s block=%s",
                tp_rank,
                self.layer_idx,
                tuple(output.shape),
                tuple(block_residual.shape),
            )
        if trace_enabled:
            record_accuracy_tensor(f"{trace_prefix}.output", output, token_dim=0)
            record_accuracy_tensor(
                f"{trace_prefix}.block_residual", block_residual, token_dim=0
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
        _validate_perf_environment()

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
        safety_gib = float(os.environ.get("KIMI_K3_KDA_A2A_SAFETY_GIB", "8"))
        if safety_gib < 0:
            raise ValueError("KIMI_K3_KDA_A2A_SAFETY_GIB must be non-negative")
        safety_bytes = int(safety_gib * (1 << 30))
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
        if _perf_mode_enabled():
            return self._forward_impl(inputs, fmha_impl)
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
        if attention_inputs.is_target_verify:
            raise RuntimeError(
                "Kimi K3 target-verify cache semantics are not connected yet"
            )
        if not attention_inputs.is_prefill and self.kv_cache is None:
            raise RuntimeError("Kimi K3 decode requires an initialized hybrid cache")
        input_ids = inputs.input_ids.reshape(-1)
        trace_enabled = _accuracy_trace_enabled()
        tp_size = int(self.parallelism_config.get_attn_tp_size())
        sp_requested = _sp_moe_enabled() and tp_size > 1
        prefill_sp = (
            sp_requested
            and attention_inputs.is_prefill
            and input_ids.numel() % tp_size == 0
        )
        decode_sp = sp_requested and not attention_inputs.is_prefill
        sp_active = prefill_sp or decode_sp
        if not attention_inputs.is_prefill and not getattr(
            self, "_decode_sp_startup_logged", False
        ):
            logging.info(
                "[K3_DECODE_SP] rank=%d env=%s requested=%s active=%s "
                "tokens=%d tp=%d ep=%d",
                int(self.parallelism_config.get_attn_tp_rank()),
                os.environ.get("KIMI_K3_SP_MOE"),
                sp_requested,
                decode_sp,
                input_ids.numel(),
                tp_size,
                int(self.parallelism_config.ep_size),
            )
            self._decode_sp_startup_logged = True
        if _perf_mode_enabled():
            if not prefill_sp:
                raise RuntimeError(
                    "K3 performance profiling requires divisible TP8/EP8 "
                    f"Prefill Sequence Parallel; tokens={input_ids.numel()}, "
                    f"TP={tp_size}, is_prefill={attention_inputs.is_prefill}"
                )
            # Kineto can arm ranks at slightly different wall times.  Align
            # once before model operators so the first embedding collective
            # does not absorb scheduler skew.  This named range is excluded
            # from model-kernel latency.
            with _perf_profile(
                "profiling.rank_entry_barrier.exclude_from_model_latency"
            ):
                barrier(Group.TP)
                # NCCL barrier completion alone does not guarantee that every
                # rank's current CUDA stream has drained before Python starts
                # submitting the profiled model operators.  Without the
                # device sync, late ranks make the first A2A kernel on early
                # ranks look several milliseconds slower than the transfer
                # itself.  Keep both operations inside the excluded range.
                torch.cuda.synchronize()
        cu_seqlens = self._cu_seqlens(attention_inputs, input_ids)
        if sp_active:
            ep_size = int(self.parallelism_config.ep_size)
            if ep_size != tp_size:
                raise RuntimeError(
                    "Kimi K3 Sequence Parallel currently requires TP == EP; "
                    f"got TP={tp_size}, EP={ep_size}"
                )
            if _accuracy_canonical_tp_enabled():
                raise RuntimeError(
                    "Kimi K3 Sequence Parallel is incompatible with canonical "
                    "TP; disable one of KIMI_K3_SP_MOE and "
                    "KIMI_K3_ACCURACY_CANONICAL_TP"
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
        elif (
            sp_requested
            and attention_inputs.is_prefill
            and not getattr(self, "_sp_shape_fallback_logged", False)
        ):
            logging.warning(
                "Kimi K3 Sequence Parallel is falling back to the replicated "
                "TP path because token count %d is not divisible by TP=%d",
                input_ids.numel(),
                tp_size,
            )
            self._sp_shape_fallback_logged = True
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
                "divisible TP8/EP8 Sequence Parallel input"
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
            local_tokens = hidden_states.shape[0] // tp_size
            tp_rank = int(self.parallelism_config.get_attn_tp_rank())
            with _perf_profile(
                "model.embedding_to_sequence_parallel_token_shard",
                hidden_states,
            ):
                hidden_states = hidden_states.narrow(
                    0, tp_rank * local_tokens, local_tokens
                ).contiguous()
        block_residual = hidden_states.new_empty(
            hidden_states.shape[0], 0, hidden_states.shape[1]
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
                )
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
        with _perf_profile("model.output_attn_residual_mix", hidden_states):
            hidden_states = _attention_residual(
                hidden_states,
                block_residual,
                self.output_attn_res_norm,
                self.output_attn_res_proj,
                self.config.layernorm_eps,
            )
        if trace_enabled:
            record_accuracy_tensor("output_attn_res", hidden_states, token_dim=0)
        with _perf_profile("model.final_rmsnorm", hidden_states):
            hidden_states = _rms_norm(
                hidden_states, self.final_norm_weight, self.config.layernorm_eps
            )
        if prefill_sp:
            with _perf_profile(
                "model.exit_token_allgather_for_framework_contract",
                hidden_states,
            ):
                hidden_states = all_gather(hidden_states, group=Group.TP)
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
