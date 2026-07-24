"""RTP-LLM serving model for the text-only Kimi K3 decoder.

The hybrid decoder interleaves KDA (linear-attention) and MLA (full-attention)
layers.  MLA runs through the shared framework path -- ``KimiK3MLA`` is a thin
``MlaAttention`` subclass over ``MlaFlashInfer*Impl`` (same as deepseek's
generic MLA), adding only K3's NoPE (neutralised by an identity rope cache) and
sigmoid output gate.  KDA runs K3's own path (Triton kernel with a pure-Torch
reference), with two forms: packed prefill dispatches to the chunk scan while
token decode dispatches to the recurrent update.  KDA canonical states are
mapped onto RTP's paged linear-cache ABI; MLA uses RTP's compressed latent
cache layout, so the same layer caches can flow through PD transfer.

Both attention modules keep a pure-Torch reference selectable via env var
(``KIMI_K3_KDA_BACKEND`` / ``KIMI_K3_MLA_BACKEND``, default ``kernel``) for
precision comparison against the framework/Triton kernels.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

import rtp_llm.ops.compute_ops as compute_ops
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3WeightNames as K3W
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, all_reduce
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.base.common.kvcache_store import (
    create_write_cache_store_impl,
)
from rtp_llm.models_py.modules.kimi_k3.tmp_test.accuracy_trace import (
    accuracy_trace_mode,
    kimi_k3_accuracy_trace,
    mark_accuracy_fake_stream,
    record_accuracy_tensor,
)
from rtp_llm.models_py.modules.kimi_k3.kda_state import (
    KDAExecutionMode,
    KimiKDAState,
)
from rtp_llm.models_py.modules.kimi_k3.mxfp4 import dequantize_mxfp4
from rtp_llm.models_py.modules.kimi_k3.reference.kda_reference import (
    kimi_kda,
    prepare_kimi_kda_inputs,
)
from rtp_llm.models_py.modules.hybrid.mla_attention import MlaAttention
from rtp_llm.models_py.triton_kernels.kimi_kda import (
    chunk_kda,
    fused_kda_gate,
    fused_recurrent_kda,
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


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply an RTP internal-layout ``[in_features, out_features]`` weight."""

    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"linear input width {x.shape[-1]} does not match weight "
            f"shape {tuple(weight.shape)}"
        )
    return torch.matmul(x, weight)


def _row_parallel_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    tp_size: int,
) -> torch.Tensor:
    """Apply and reduce a K3 row-parallel projection.

    A BF16 partial result followed by a BF16 TP reduction introduces one
    rounding point per rank.  K3's residual and routed-MoE stack can amplify
    those otherwise small errors enough to change a top-k expert.  Keep the
    CUDA GEMM output and collective in FP32, then round once after the
    reduction.  The CPU branch retains the ordinary path for unit tests and
    environments where CUDA's ``mm(out_dtype=...)`` is unavailable.
    """

    if tp_size <= 1:
        return _linear(x, weight)
    if (
        x.is_cuda
        and x.ndim == 2
        and x.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == x.dtype
    ):
        output = torch.mm(x, weight, out_dtype=torch.float32)
        output = all_reduce(output, group=Group.TP)
        return output.to(dtype=x.dtype)
    output = _linear(x, weight)
    return all_reduce(output, group=Group.TP)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
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
    gate_float = gate.float()
    up_float = up.float()
    activated_gate = (
        beta * torch.tanh(gate_float / beta) * torch.sigmoid(gate_float)
    )
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
    candidates = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    candidates_float = candidates.float()
    normalized = candidates_float * torch.rsqrt(
        candidates_float.square().mean(dim=-1, keepdim=True) + eps
    )
    score_weight = norm_weight.float() * projection_weight.reshape(-1).float()
    probabilities = torch.softmax(
        (normalized * score_weight).sum(dim=-1), dim=-1
    )
    return torch.einsum("tb,tbd->td", probabilities, candidates_float).to(
        dtype=prefix_sum.dtype
    )


def _sequence_offsets(
    cu_seqlens: torch.Tensor, token_count: int
) -> list[tuple[int, int]]:
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a one-dimensional [batch + 1] tensor")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must use an integer dtype")
    offsets = [int(value) for value in cu_seqlens.detach().cpu().tolist()]
    if offsets[0] != 0 or offsets[-1] != token_count:
        raise ValueError(
            f"cu_seqlens must start at 0 and end at {token_count}, got {offsets}"
        )
    if any(left > right for left, right in zip(offsets, offsets[1:])):
        raise ValueError("cu_seqlens must be non-decreasing")
    return list(zip(offsets, offsets[1:]))


def _packed_causal_depthwise_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch fallback for packed KDA short convolution.

    The returned cache is ``[batch, channels, kernel_size - 1]`` in
    chronological order, matching the standalone K3 correctness model.
    """

    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[0]:
        raise ValueError(
            "packed causal conv expects x=[tokens,channels] and "
            "weight=[channels,kernel]"
        )
    ranges = _sequence_offsets(cu_seqlens, x.shape[0])
    channels, kernel_size = weight.shape
    history_size = kernel_size - 1
    expected_state = (len(ranges), channels, history_size)
    if initial_state is None:
        initial_state = x.new_zeros(expected_state)
    elif tuple(initial_state.shape) != expected_state:
        raise ValueError(
            f"conv state must have shape {expected_state}, got "
            f"{tuple(initial_state.shape)}"
        )

    outputs: list[torch.Tensor] = []
    final_states: list[torch.Tensor] = []
    for sequence_idx, (start, end) in enumerate(ranges):
        sequence = x[start:end].transpose(0, 1).unsqueeze(0)
        history = initial_state[sequence_idx].unsqueeze(0).to(dtype=x.dtype)
        combined = torch.cat((history, sequence), dim=-1)
        if end == start:
            output = x.new_empty((0, channels))
        else:
            # FLA's ShortConvolution Triton kernel converts every input and
            # weight tap to FP32, accumulates taps from oldest to newest, then
            # applies SiLU in FP32 and performs one RTNE cast.  cuDNN's BF16
            # depthwise F.conv1d uses a different reduction path and differs
            # by one ULP, which can perturb K3's top-k routing boundary.
            token_count = end - start
            weight_float = weight.float()
            output_float = (
                combined[0, :, :token_count].transpose(0, 1).float()
                * weight_float[:, 0]
            )
            for tap in range(1, kernel_size):
                output_float = output_float + (
                    combined[0, :, tap : tap + token_count]
                    .transpose(0, 1)
                    .float()
                    * weight_float[:, tap]
                )
            output = (
                output_float * torch.sigmoid(output_float)
            ).to(dtype=x.dtype)
        outputs.append(output)
        final_states.append(
            combined.squeeze(0)[:, -history_size:]
            if history_size
            else combined.squeeze(0)[:, :0]
        )
    return torch.cat(outputs, dim=0), torch.stack(final_states, dim=0)


class KimiK3LinearCacheAdapter:
    """Map canonical KDA state to RTP's paged linear-cache byte layout.

    RTP stores ``[H,V,K]`` followed by a single packed ``[history,QKV]``
    convolution state in each linear block.  The correctness KDA equations use
    ``[H,K,V]`` and three ``[channels,history]`` tensors, so conversion is kept
    explicit at this boundary.  Cached prefill writes a state at every physical
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
        conv_section_bytes = (
            self.projection_size * self.converter.conv_state_item_size
        )
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

    def _views(
        self, kv_cache: LayerKVCache
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        offsets = [
            int(value) for value in cu_seqlens.detach().cpu().tolist()
        ]
        new_lengths = [right - left for left, right in zip(offsets, offsets[1:])]
        source = (
            attention_inputs.prefix_lengths
            if mode == "prefill"
            else attention_inputs.sequence_lengths
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
        block_map = attention_inputs.kv_cache_kernel_block_id_device
        if block_map is None or block_map.numel() == 0 or block_map.ndim != 2:
            raise ValueError("KDA cache requires a two-dimensional kernel block map")
        return [
            [int(value) for value in row]
            for row in block_map.detach().cpu().tolist()
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

    def load(
        self,
        kv_cache: LayerKVCache,
        attention_inputs: PyAttentionInputs,
        cu_seqlens: torch.Tensor,
        *,
        mode: KDAExecutionMode,
    ) -> KimiKDAState:
        ssm_cache, conv_cache = self._views(kv_cache)
        past_lengths, _ = self._lengths(
            attention_inputs, cu_seqlens, mode=mode
        )
        block_map = self._block_map(attention_inputs)
        page_size = int(kv_cache.seq_size_per_block)
        if page_size <= 0:
            raise ValueError("linear cache seq_size_per_block must be positive")

        recurrent_states: list[torch.Tensor] = []
        q_states: list[torch.Tensor] = []
        k_states: list[torch.Tensor] = []
        v_states: list[torch.Tensor] = []
        is_fake_stream = self._is_fake_stream(attention_inputs)
        for sequence_idx, past_length in enumerate(past_lengths):
            if is_fake_stream or past_length == 0 or self._is_fake_block_row(
                block_map[sequence_idx]
            ):
                recurrent_states.append(
                    ssm_cache.new_zeros(
                        self.local_heads, self.head_dim, self.head_dim
                    )
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
            # Physical cache is [H,V,K]; KDA correctness state is [H,K,V].
            recurrent_states.append(ssm_cache[block_id].transpose(-1, -2))
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
        past_lengths, new_lengths = self._lengths(
            attention_inputs, cu_seqlens, mode=mode
        )
        block_map = self._block_map(attention_inputs)
        page_size = int(kv_cache.seq_size_per_block)
        is_fake_stream = self._is_fake_stream(attention_inputs)
        for sequence_idx, (past_length, new_length) in enumerate(
            zip(past_lengths, new_lengths)
        ):
            if is_fake_stream or new_length == 0 or self._is_fake_block_row(
                block_map[sequence_idx]
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
            )

    def store_position(
        self,
        state: KimiKDAState,
        state_index: int,
        kv_cache: LayerKVCache,
        attention_inputs: PyAttentionInputs,
        sequence_idx: int,
        absolute_position: int,
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
        block_map = self._block_map(attention_inputs)
        page_size = int(kv_cache.seq_size_per_block)
        if page_size <= 0:
            raise ValueError("linear cache seq_size_per_block must be positive")
        block_id = self._block_id_or_none(
            block_map, sequence_idx, absolute_position, page_size
        )
        if block_id is None:
            return False
        self._copy_state_to_block(
            state, state_index, block_id, ssm_cache, conv_cache
        )
        return True

    @staticmethod
    def _copy_state_to_block(
        state: KimiKDAState,
        state_index: int,
        block_id: int,
        ssm_cache: torch.Tensor,
        conv_cache: torch.Tensor,
    ) -> None:
        ssm_cache[block_id].copy_(
            state.recurrent_state[state_index]
            .transpose(-1, -2)
            .to(dtype=ssm_cache.dtype)
        )
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
        self.trace_prefix = (
            f"layer.{layer_idx}.kda" if layer_idx >= 0 else "kda"
        )
        runtime = config.k3_runtime_config
        self.head_dim = int(
            config.linear_attention_config.linear_key_head_dim
        )
        tp_size = int(parallelism_config.get_attn_tp_size())
        total_heads = int(
            config.linear_attention_config.linear_num_key_heads
        )
        if total_heads % tp_size:
            raise ValueError(
                f"KDA heads {total_heads} must be divisible by attention TP {tp_size}"
            )
        self.local_heads = total_heads // tp_size
        self.projection_size = self.local_heads * self.head_dim
        self.eps = float(config.layernorm_eps)
        self.gate_lower_bound = runtime.kda_gate_lower_bound
        # KDA delta-net core backend: the ported Triton kernel by default, with
        # the pure-Torch reference reachable for precision verification.  See
        # _kda_core; agreement is pinned by kimi_k3_kda_kernel_parity_test.
        self._kda_backend = os.environ.get("KIMI_K3_KDA_BACKEND", "kernel").lower()
        if self._kda_backend not in ("kernel", "reference"):
            raise ValueError(
                "KIMI_K3_KDA_BACKEND must be 'kernel' or 'reference', got "
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
        # KDA q/k/v projections and their depthwise convs are stored fused on
        # the shared ``W.linear_attn_*`` vocabulary (aligned with main's
        # kimi_linear).  Split them once into the per-projection views the
        # conv/cache path consumes; the KimiKDAState cache ABI is unchanged.
        fused_qkv = weights[W.linear_attn_qkv_w]
        if fused_qkv.shape[1] != 3 * self.projection_size:
            raise ValueError(
                "fused KDA qkv width "
                f"{fused_qkv.shape[1]} != 3*{self.projection_size}"
            )
        self.kda_q_w, self.kda_k_w, self.kda_v_w = torch.split(
            fused_qkv, self.projection_size, dim=1
        )
        fused_conv = weights[W.linear_attn_conv1d_w].squeeze(1)
        if fused_conv.shape[0] != 3 * self.projection_size:
            raise ValueError(
                "fused KDA conv channels "
                f"{fused_conv.shape[0]} != 3*{self.projection_size}"
            )
        self.kda_q_conv, self.kda_k_conv, self.kda_v_conv = torch.split(
            fused_conv, self.projection_size, dim=0
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the KDA delta-net core (l2norm + decay gate + scan).

        Drop-in for the ``kimi_kda`` reference: same ``[1,T,H,*]`` I/O and a
        ``[N,H,K,V]`` fp32 final state.  Defaults to the ported Triton kernel
        (``fused_kda_gate`` -> ``chunk_kda``/``fused_recurrent_kda``); the
        pure-Torch reference stays reachable via ``KIMI_K3_KDA_BACKEND=reference``
        for precision verification.  Kernel/reference agreement is pinned by
        ``rtp_llm/test/tmp_test/kimi_k3/kimi_k3_kda_kernel_parity_test.py``
        (including the non-zero
        initial_state prefill/decode seams).
        """

        if self._kda_backend == "reference":
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

        # Both kernels mutate initial_state in place; feed a private fp32 clone
        # so the cached state stays pristine and gets a fresh final-state tensor.
        state_in = (
            None
            if recurrent_state is None
            else recurrent_state.float().contiguous().clone()
        )
        if mode == "prefill":
            output, final_state = chunk_kda(
                q,
                k,
                v,
                raw_gate,
                raw_beta,
                initial_state=state_in,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                A_log=a_log,
                dt_bias=dt_bias,
                safe_gate=self.gate_lower_bound is not None,
                lower_bound=self.gate_lower_bound,
            )
        elif mode == "decode":
            # The recurrent kernel's precomputed gate/beta path is already
            # bitwise-aligned with Dummy decode. Keep it separate from the
            # chunk kernel's fused Prefill contract above.
            gate = fused_kda_gate(
                raw_gate.float(),
                a_log.float(),
                dt_bias=dt_bias.float(),
                lower_bound=self.gate_lower_bound,
                output_dtype=torch.float32,
            )
            beta = raw_beta.float().sigmoid()
            output, final_state = fused_recurrent_kda(
                q,
                k,
                v,
                gate,
                beta,
                initial_state=state_in,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=False,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise ValueError(f"unsupported KDA execution mode {mode!r}")
        # Normalize to the reference's [1,T,H,V] / q.dtype output contract.
        output = output.reshape(q.shape[0], q.shape[1], v.shape[2], v.shape[3]).to(
            dtype=q.dtype
        )
        return output, final_state

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

        ranges = _sequence_offsets(cu_seqlens, q_projected.shape[0])
        past_lengths, _ = self.cache_adapter._lengths(
            attention_inputs, cu_seqlens, mode="prefill"
        )
        page_size = int(kv_cache.seq_size_per_block)
        if page_size <= 0:
            raise ValueError("linear cache seq_size_per_block must be positive")

        packed_outputs: list[torch.Tensor] = []
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
                segment_cu_seqlens = torch.tensor(
                    [0, segment_length],
                    dtype=torch.int32,
                    device=cu_seqlens.device,
                )
                q, q_state = _packed_causal_depthwise_conv1d(
                    q_projected[cursor:segment_end],
                    self.kda_q_conv,
                    segment_cu_seqlens,
                    q_state,
                )
                k, k_state = _packed_causal_depthwise_conv1d(
                    k_projected[cursor:segment_end],
                    self.kda_k_conv,
                    segment_cu_seqlens,
                    k_state,
                )
                v, v_state = _packed_causal_depthwise_conv1d(
                    v_projected[cursor:segment_end],
                    self.kda_v_conv,
                    segment_cu_seqlens,
                    v_state,
                )
                packed_q.append(q)
                packed_k.append(k)
                packed_v.append(v)
                head_shape = (
                    1,
                    segment_length,
                    self.local_heads,
                    self.head_dim,
                )
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
                segment_output, recurrent_state = self._kda_core(
                    q.reshape(head_shape),
                    k.reshape(head_shape),
                    v.reshape(head_shape),
                    raw_gate[cursor:segment_end].reshape(head_shape),
                    raw_beta[cursor:segment_end]
                    .float()
                    .reshape(1, segment_length, self.local_heads),
                    self.weights[W.linear_attn_alog],
                    self.weights[W.linear_attn_dt_b_kda],
                    recurrent_state,
                    mode="prefill",
                    cu_seqlens=segment_cu_seqlens,
                )
                packed_outputs.append(segment_output.squeeze(0))
                segment_state = KimiKDAState(
                    q_conv_state=q_state,
                    k_conv_state=k_state,
                    v_conv_state=v_state,
                    recurrent_state=recurrent_state,
                )
                self.cache_adapter.store_position(
                    segment_state,
                    0,
                    kv_cache,
                    attention_inputs,
                    sequence_idx,
                    absolute_position + segment_length - 1,
                )
                cursor = segment_end
                absolute_position += segment_length

            q_finals.append(q_state[0])
            k_finals.append(k_state[0])
            v_finals.append(v_state[0])
            recurrent_finals.append(recurrent_state[0])

        if packed_outputs:
            output = torch.cat(packed_outputs, dim=0).unsqueeze(0)
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
            output = q_projected.new_empty(
                1, 0, self.local_heads, self.head_dim
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
    ) -> tuple[torch.Tensor, KimiKDAState]:
        if state is not None and kv_cache is not None:
            raise ValueError("pass either an explicit KDA state or LayerKVCache, not both")
        if kv_cache is not None:
            if attention_inputs is None:
                raise ValueError("attention_inputs are required with a KDA cache")
            state = self.cache_adapter.load(
                kv_cache, attention_inputs, cu_seqlens, mode=mode
            )
        if mode == "decode" and state is not None:
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
        q_projected = _linear(hidden_states, self.kda_q_w)
        k_projected = _linear(hidden_states, self.kda_k_w)
        v_projected = _linear(hidden_states, self.kda_v_w)
        record_accuracy_tensor(
            f"{self.trace_prefix}.q_projected", q_projected, token_dim=0
        )
        record_accuracy_tensor(
            f"{self.trace_prefix}.k_projected", k_projected, token_dim=0
        )
        record_accuracy_tensor(
            f"{self.trace_prefix}.v_projected", v_projected, token_dim=0
        )
        token_count = hidden_states.shape[0]
        head_shape = (1, token_count, self.local_heads, self.head_dim)
        raw_gate = _linear(
            _linear(hidden_states, self.weights[W.linear_attn_f_a_w]),
            self.weights[W.linear_attn_f_b_w],
        )
        raw_beta = _linear(hidden_states, self.weights[W.linear_attn_b_w])
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
            q, q_final = _packed_causal_depthwise_conv1d(
                q_projected,
                self.kda_q_conv,
                cu_seqlens,
                q_state,
            )
            k, k_final = _packed_causal_depthwise_conv1d(
                k_projected,
                self.kda_k_conv,
                cu_seqlens,
                k_state,
            )
            v, v_final = _packed_causal_depthwise_conv1d(
                v_projected,
                self.kda_v_conv,
                cu_seqlens,
                v_state,
            )
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
            output, recurrent_final = self._kda_core(
                q.reshape(head_shape),
                k.reshape(head_shape),
                v.reshape(head_shape),
                raw_gate.reshape(head_shape),
                raw_beta.float().reshape(1, token_count, self.local_heads),
                self.weights[W.linear_attn_alog],
                self.weights[W.linear_attn_dt_b_kda],
                recurrent_state,
                mode=mode,
                cu_seqlens=cu_seqlens,
            )
            final_state = KimiKDAState(
                q_conv_state=q_final,
                k_conv_state=k_final,
                v_conv_state=v_final,
                recurrent_state=recurrent_final,
            )
            record_accuracy_tensor(
                f"{self.trace_prefix}.q_conv", q, token_dim=0
            )
            record_accuracy_tensor(
                f"{self.trace_prefix}.k_conv", k, token_dim=0
            )
            record_accuracy_tensor(
                f"{self.trace_prefix}.v_conv", v, token_dim=0
            )
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
            f"{self.trace_prefix}.state.recurrent", final_state.recurrent_state
        )
        output_gate = _linear(
            hidden_states, self.weights[W.linear_attn_g_w]
        ).reshape(head_shape)
        record_accuracy_tensor(
            f"{self.trace_prefix}.output_gate",
            output_gate.squeeze(0),
            token_dim=0,
        )
        output_float = output.float()
        output = (
            output_float
            * torch.rsqrt(
                output_float.square().mean(dim=-1, keepdim=True) + self.eps
            )
            * self.weights[W.linear_attn_norm_w].float()
            * torch.sigmoid(output_gate.float())
        ).to(dtype=hidden_states.dtype)
        record_accuracy_tensor(
            f"{self.trace_prefix}.gated_output", output, token_dim=1
        )
        output = _row_parallel_linear(
            output.reshape(token_count, self.projection_size),
            self.weights[W.linear_attn_out_w],
            self.parallelism_config.get_attn_tp_size(),
        )
        record_accuracy_tensor(
            f"{self.trace_prefix}.output", output, token_dim=0
        )
        if kv_cache is not None and not stored_page_states:
            self.cache_adapter.store(
                final_state,
                kv_cache,
                attention_inputs,
                cu_seqlens,
                mode=mode,
            )
        return output, final_state


class KimiK3MLA(MlaAttention):
    """No-RoPE MLA over RTP's packed-token layout.

    By default this runs the *framework* MLA path -- it is a thin subclass of
    :class:`MlaAttention` (same fused down-projection, ``MlaFlashInfer*Impl``
    kernel, paged latent cache) as deepseek's generic MLA, adding only the two
    things K3 needs: NoPE (neutralised via K3's identity ``rope_cos_sin_cache``)
    and a sigmoid output gate applied through the ``_apply_output_gate`` hook.

    The original hand-written einsum path is retained verbatim as
    :meth:`_reference_forward` and selected by ``KIMI_K3_MLA_BACKEND=reference``
    (default ``kernel``), mirroring KDA's ``KIMI_K3_KDA_BACKEND`` switch, so the
    two implementations can be diffed for precision.
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
            float(config.layernorm_eps),
            config.quant_config,
        )
        self.config = config
        self.weights = weights
        self.trace_prefix = (
            f"layer.{layer_idx}.mla" if layer_idx >= 0 else "mla"
        )
        tp_size = int(parallelism_config.get_attn_tp_size())
        total_heads = int(config.attn_config.head_num)
        if total_heads % tp_size:
            raise ValueError(
                f"MLA heads {total_heads} must be divisible by attention TP {tp_size}"
            )
        self.local_heads = total_heads // tp_size
        self.q_lora_rank = int(config.attn_config.q_lora_rank)
        self.kv_lora_rank = int(config.attn_config.kv_lora_rank)
        self.nope_dim = int(config.attn_config.nope_head_dim)
        self.suffix_dim = int(config.attn_config.rope_head_dim)
        self.value_dim = int(config.attn_config.v_head_dim)
        self.q_head_dim = self.nope_dim + self.suffix_dim
        self.softmax_scale = self.q_head_dim**-0.5
        self.eps = float(config.layernorm_eps)
        runtime = config.k3_runtime_config
        if not runtime.mla_use_nope:
            raise ValueError("Kimi K3 requires the physical MLA suffix to remain no-RoPE")
        self.use_output_gate = runtime.mla_use_output_gate
        self._mla_backend = os.environ.get("KIMI_K3_MLA_BACKEND", "kernel").lower()
        if self._mla_backend not in ("kernel", "reference"):
            raise ValueError(
                f"KIMI_K3_MLA_BACKEND must be 'kernel' or 'reference', "
                f"got {self._mla_backend!r}"
            )

        # Per-projection views the reference einsum path consumes, reconstructed
        # from the shared ``W.mla_*`` layout so ``_reference_forward`` stays byte
        # identical to the pre-migration path.  ``mla_fusedqkrope_w`` is
        # ``concat([q_a_proj, kv_a_proj], dim=0).T`` (see ``concat_0_tranpose``),
        # so the leading ``q_lora_rank`` columns are the old ``MLA_Q_A`` and the
        # trailing columns the old ``MLA_KV_A``.  The full-rank sigmoid gate keeps
        # its private ``K3W.MLA_OUTPUT_GATE`` key.  (The kernel backend does not
        # use these views -- it goes through the linears built by
        # ``MlaAttention.__init__``.)
        fused_qkv_a = weights[W.mla_fusedqkrope_w]
        self._q_a_w = fused_qkv_a[:, : self.q_lora_rank].contiguous()
        self._kv_a_w = fused_qkv_a[:, self.q_lora_rank :].contiguous()
        self._q_a_norm = weights[W.mla_q_a_ln_gamma]
        self._q_b_w = weights[W.mla_q_b_w]
        self._kv_a_norm = weights[W.mla_kv_a_ln_gamma]
        self._kv_b_w = weights[W.mla_kv_b_w]
        self._o_w = weights[W.attn_o_w]

    def _apply_output_gate(
        self, attn_output: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """K3 sigmoid output gate, applied on the framework (kernel) path.

        ``attn_output`` is the framework context flattened to
        ``[tokens, local_heads * v_head_dim]`` (head-major), matching the flat
        layout of the full-rank gate projection, so the gate multiplies element
        wise per (head, value) exactly as the reference path does before o_proj.
        The gate weight is sharded ``sp_neg1`` and this runs before o_proj's TP
        all_reduce, so each rank gates only its local heads -- TP correct.
        """
        if not self.use_output_gate:
            return attn_output
        gate = _linear(
            hidden_states, self.weights[K3W.MLA_OUTPUT_GATE]
        ).reshape_as(attn_output)
        return attn_output * torch.sigmoid(gate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        if self._mla_backend == "reference":
            attn_inputs = fmha_impl.attn_inputs
            is_prefill = bool(attn_inputs.is_prefill)
            cu_seqlens = self._reference_cu_seqlens(
                attn_inputs, hidden_states.shape[0], is_prefill, hidden_states.device
            )
            return self._reference_forward(
                hidden_states,
                cu_seqlens,
                is_prefill=is_prefill,
                kv_cache=kv_cache,
                attention_inputs=attn_inputs,
            )
        if accuracy_trace_mode() is None:
            return super().forward(hidden_states, fmha_impl, kv_cache)

        # Trace the real MLA kernel path without materializing the quadratic
        # matrix used by ``_reference_forward``.  The serving operations below
        # intentionally mirror ``MlaAttention.forward``; only the last query's
        # score/probability row is recomputed for diagnosis.
        input_shape = hidden_states.shape[:-1]
        fused_qkv = self.fused_qkv_a_proj(hidden_states)
        q, compressed = torch.split(
            fused_qkv,
            [
                self.q_lora_rank,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ],
            dim=-1,
        )
        query_latent = self.q_a_layernorm(q.contiguous())
        query = self.q_b_proj(query_latent).reshape(
            -1, self.num_heads, self.q_head_dim
        )
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
        record_accuracy_tensor(
            f"{self.trace_prefix}.query", query, token_dim=0
        )
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
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        context_by_head = context.reshape(
            hidden_states.shape[0], self.num_heads, self.v_head_dim
        )
        record_accuracy_tensor(
            f"{self.trace_prefix}.context_last_query",
            context_by_head[-1:],
            token_dim=0,
        )

        # Accuracy requests contain one packed sequence.  Recompute only its
        # final score row: O(T) storage instead of the O(T^2) full matrix.
        attn_inputs = fmha_impl.attn_inputs
        cu_seqlens = self._reference_cu_seqlens(
            attn_inputs,
            hidden_states.shape[0],
            bool(attn_inputs.is_prefill),
            hidden_states.device,
        )
        ranges = _sequence_offsets(cu_seqlens, hidden_states.shape[0])
        if len(ranges) != 1:
            raise RuntimeError(
                "K3 MLA accuracy trace currently requires one packed sequence"
            )
        expanded = _linear(
            compressed_kv.to(dtype=self._kv_b_w.dtype),
            self._kv_b_w,
        ).reshape(
            compressed_kv.shape[0],
            self.local_heads,
            self.nope_dim + self.value_dim,
        )
        key_nope, _ = torch.split(
            expanded, [self.nope_dim, self.value_dim], dim=-1
        )
        key = torch.cat(
            (
                key_nope,
                key_suffix.unsqueeze(1).expand(-1, self.local_heads, -1),
            ),
            dim=-1,
        )
        scores = (
            torch.einsum("thd,shd->hts", query[-1:], key)
            * self.softmax_scale
        )
        probabilities = torch.softmax(
            scores, dim=-1, dtype=torch.float32
        ).to(dtype=query.dtype)
        record_accuracy_tensor(
            f"{self.trace_prefix}.scores_last_query", scores
        )
        record_accuracy_tensor(
            f"{self.trace_prefix}.probabilities_last_query", probabilities
        )
        record_accuracy_tensor(
            f"{self.trace_prefix}.cache", canonical_current
        )

        if self.use_output_gate:
            output_gate = _linear(
                hidden_states, self.weights[K3W.MLA_OUTPUT_GATE]
            ).reshape_as(context)
            record_accuracy_tensor(
                f"{self.trace_prefix}.output_gate",
                output_gate,
                token_dim=0,
            )
            context = context * torch.sigmoid(output_gate)
        output = self.o_proj(context)
        if self.parallelism_config.get_attn_tp_size() > 1:
            output = all_reduce(output, group=Group.TP)
        record_accuracy_tensor(
            f"{self.trace_prefix}.output", output, token_dim=0
        )
        return output

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
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        is_prefill: bool,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
    ) -> torch.Tensor:
        ranges = _sequence_offsets(cu_seqlens, hidden_states.shape[0])
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
                    int(value)
                    for value in length_tensor.detach().cpu().tolist()
                ]
            if len(past_lengths) != len(ranges):
                raise ValueError("MLA cache batch does not match packed sequences")
        if (not is_prefill or any(past_lengths)) and kv_cache is None:
            raise RuntimeError("MLA decode/prefix reuse requires a LayerKVCache")

        query_latent = _rms_norm(
            _linear(hidden_states, self._q_a_w),
            self._q_a_norm,
            self.eps,
        )
        query = _linear(query_latent, self._q_b_w).reshape(
            hidden_states.shape[0], self.local_heads, self.q_head_dim
        )
        record_accuracy_tensor(
            f"{self.trace_prefix}.query_latent", query_latent, token_dim=0
        )
        record_accuracy_tensor(
            f"{self.trace_prefix}.query", query, token_dim=0
        )
        compressed = _linear(hidden_states, self._kv_a_w)
        compressed_kv, key_suffix = torch.split(
            compressed, [self.kv_lora_rank, self.suffix_dim], dim=-1
        )
        compressed_kv = _rms_norm(
            compressed_kv, self._kv_a_norm, self.eps
        )
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
            if base.numel() % (base.shape[0] * width):
                raise ValueError("MLA cache storage is not divisible by latent width")
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
                return compressed_kv.new_empty(
                    0, self.kv_lora_rank + self.suffix_dim
                )
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
                    hidden_states.new_empty((0, self.local_heads, self.value_dim))
                )
                continue
            fake_cache_row = (
                is_fake_stream
                or (
                    block_map is not None
                    and KimiK3LinearCacheAdapter._is_fake_block_row(
                        block_map[sequence_idx]
                    )
                )
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
            expanded = _linear(
                all_latent, self._kv_b_w
            ).reshape(
                all_latent.shape[0],
                self.local_heads,
                self.nope_dim + self.value_dim,
            )
            key_nope, value = torch.split(
                expanded, [self.nope_dim, self.value_dim], dim=-1
            )
            expanded_suffix = all_suffix.unsqueeze(1).expand(
                -1, self.local_heads, -1
            )
            key = torch.cat((key_nope, expanded_suffix), dim=-1)
            scores = torch.einsum(
                "thd,shd->hts", query[start:end], key
            ) * self.softmax_scale
            query_positions = effective_prefix_length + torch.arange(
                end - start, device=hidden_states.device
            )
            key_positions = torch.arange(
                effective_prefix_length + end - start,
                device=hidden_states.device,
            )
            causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            scores = scores.masked_fill(
                ~causal_mask.unsqueeze(0), torch.finfo(scores.dtype).min
            )
            probabilities = torch.softmax(
                scores, dim=-1, dtype=torch.float32
            ).to(dtype=query.dtype)
            context = torch.einsum("hts,shv->thv", probabilities, value).to(
                dtype=hidden_states.dtype
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
            write_tokens(
                sequence_idx, effective_prefix_length, current_compressed
            )
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
            output_gate = _linear(
                hidden_states, self.weights[K3W.MLA_OUTPUT_GATE]
            ).reshape_as(output)
            record_accuracy_tensor(
                f"{self.trace_prefix}.output_gate", output_gate, token_dim=0
            )
            output = output * torch.sigmoid(output_gate)
        output = _row_parallel_linear(
            output.reshape(hidden_states.shape[0], -1),
            self._o_w,
            self.parallelism_config.get_attn_tp_size(),
        )
        record_accuracy_tensor(
            f"{self.trace_prefix}.output", output, token_dim=0
        )
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
        self.trace_prefix = (
            f"layer.{layer_idx}.dense" if layer_idx >= 0 else "dense"
        )
        runtime = config.k3_runtime_config
        self.beta = runtime.activation_situ_beta
        self.linear_beta = runtime.activation_situ_linear_beta

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = _linear(hidden_states, self.weights[K3W.DENSE_GATE])
        up = _linear(hidden_states, self.weights[K3W.DENSE_UP])
        record_accuracy_tensor(
            f"{self.trace_prefix}.gate", gate, token_dim=0
        )
        record_accuracy_tensor(f"{self.trace_prefix}.up", up, token_dim=0)
        activated = _situ(
            gate,
            up,
            self.beta,
            self.linear_beta,
        )
        record_accuracy_tensor(
            f"{self.trace_prefix}.activation", activated, token_dim=0
        )
        output = _row_parallel_linear(
            activated,
            self.weights[K3W.DENSE_DOWN],
            self.parallelism_config.get_ffn_tp_size(),
        )
        record_accuracy_tensor(
            f"{self.trace_prefix}.output", output, token_dim=0
        )
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
        self.trace_prefix = (
            f"layer.{layer_idx}.moe" if layer_idx >= 0 else "moe"
        )
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

    def _route(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.sigmoid(
            _linear(hidden_states.float(), self.weights[K3W.MOE_GATE].float())
        )
        choice_scores = (
            scores + self.weights[K3W.MOE_CORRECTION_BIAS].float().unsqueeze(0)
        )
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
            expert_mask = group_mask.unsqueeze(-1).expand_as(grouped).reshape_as(
                choice_scores
            )
            choice_scores = choice_scores.masked_fill(~expert_mask, float("-inf"))
        if accuracy_trace_mode() is not None:
            boundary = choice_scores.topk(self.top_k + 1, dim=-1).values
            record_accuracy_tensor(
                f"{self.trace_prefix}.router_scores", scores, token_dim=0
            )
            record_accuracy_tensor(
                f"{self.trace_prefix}.router_margin",
                boundary[:, self.top_k - 1] - boundary[:, self.top_k],
                token_dim=0,
            )
        expert_ids = choice_scores.topk(
            self.top_k, dim=-1, sorted=False
        ).indices
        expert_weights = scores.gather(1, expert_ids)
        if self.top_k > 1 and self.renormalize:
            expert_weights = expert_weights / (
                expert_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        return expert_ids, expert_weights * self.routed_scaling_factor

    def _expert_weight(
        self, packed_name: str, scale_name: str, local_expert: int, dtype: torch.dtype
    ) -> torch.Tensor:
        return dequantize_mxfp4(
            self.weights[packed_name][local_expert],
            self.weights[scale_name][local_expert],
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
        )
        w3 = self._expert_weight(
            K3W.MOE_W3_PACKED,
            K3W.MOE_W3_SCALE,
            local_expert,
            expert_input.dtype,
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

        if expert_ids.shape != routing_weights.shape:
            raise ValueError("expert_ids and routing_weights must have equal shape")
        if expert_ids.shape[0] != routed_input.shape[0]:
            raise ValueError("routing tensors must have one row per routed token")

        local_ids = (
            expert_ids
            if ids_are_local
            else expert_ids - self.local_expert_start
        )
        valid = (local_ids >= 0) & (local_ids < self.local_expert_count)
        routed_output = torch.zeros_like(routed_input)
        selected_local = local_ids[valid].unique()
        for local_expert_tensor in selected_local:
            local_expert = int(local_expert_tensor.item())
            matches = (local_ids == local_expert).nonzero(as_tuple=False)
            token_indices = matches[:, 0]
            slot_indices = matches[:, 1]
            expert_output = self._run_local_expert(
                routed_input[token_indices], local_expert
            )
            expert_output = expert_output * routing_weights[
                token_indices, slot_indices
            ].to(dtype=expert_output.dtype).unsqueeze(-1)
            routed_output.index_add_(0, token_indices, expert_output)
        return routed_output

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
        tokens_per_tp_rank = (
            token_count + self.attn_tp_size - 1
        ) // self.attn_tp_size
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

    @staticmethod
    def _deepep_normal_routing_groups(
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Split routing slots into ACCL-EP normal-combine widths.

        The deployed ACCL-EP generated kernel only instantiates top-k widths
        4/6/8/10.  A short final group is padded with ``(-1, 0)``; wider K3
        routing is evaluated in multiple independent groups and summed, so no
        selected expert or router weight is dropped.  In particular, the
        production K3 top-k 16 path becomes 10 + 6, while the tiny top-k 2
        smoke model becomes one padded width-4 group.
        """

        if expert_ids.shape != routing_weights.shape or expert_ids.ndim != 2:
            raise ValueError(
                "DeepEP expert ids and routing weights must be equal rank-2 tensors"
            )
        topk = expert_ids.shape[1]
        if topk == 0:
            raise ValueError("DeepEP routing requires at least one expert slot")

        supported = (4, 6, 8, 10)
        groups: list[tuple[torch.Tensor, torch.Tensor]] = []
        begin = 0
        while begin < topk:
            remaining = topk - begin
            if remaining > supported[-1]:
                group_width = supported[-1]
            else:
                group_width = next(width for width in supported if width >= remaining)
            take = min(remaining, group_width)
            ids = expert_ids[:, begin : begin + take]
            weights = routing_weights[:, begin : begin + take]
            if take < group_width:
                ids = torch.cat(
                    (
                        ids,
                        torch.full(
                            (ids.shape[0], group_width - take),
                            -1,
                            dtype=ids.dtype,
                            device=ids.device,
                        ),
                    ),
                    dim=1,
                )
                weights = torch.cat(
                    (
                        weights,
                        torch.zeros(
                            (weights.shape[0], group_width - take),
                            dtype=weights.dtype,
                            device=weights.device,
                        ),
                    ),
                    dim=1,
                )
            groups.append((ids.contiguous(), weights.contiguous()))
            begin += take
        return groups

    def _deepep_normal(
        self,
        wrapper: Any,
        routed_input: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """DeepEP normal dispatch -> weighted local sum -> combine."""

        buffer = wrapper.buffer
        dispatch_input = self._pad_dispatch_payload(routed_input)
        combined_accumulator: Optional[torch.Tensor] = None
        for group_ids, group_weights in self._deepep_normal_routing_groups(
            expert_ids, routing_weights
        ):
            (
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                num_tokens_per_expert,
                is_token_in_rank,
                _,
            ) = buffer.get_dispatch_layout(group_ids, self.expert_num)
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
                group_ids,
                group_weights,
                expert_alignment=1,
            )
            if not isinstance(recv_x, torch.Tensor):
                raise RuntimeError(
                    "Kimi K3 DeepEP correctness path requires BF16 dispatch"
                )
            local_latent = self._local_expert_sum(
                recv_x[:, : self.latent_size],
                recv_topk_idx,
                recv_topk_weights,
                ids_are_local=True,
            )
            # ``combine`` only adds x values. Router weights were already
            # applied by ``_local_expert_sum`` above, so do not pass them a
            # second time.  This is also required for production K3 top-k 16:
            # dispatch is split into 10 + 6 supported-width groups, whereas
            # the process-wide DeepEP wrapper still advertises moe_k=16.
            combined, _, _ = buffer.combine(
                self._pad_dispatch_payload(local_latent),
                handle,
            )
            combined = combined[:, : self.latent_size].float()
            combined_accumulator = (
                combined
                if combined_accumulator is None
                else combined_accumulator + combined
            )

        assert combined_accumulator is not None
        return combined_accumulator.to(dtype=routed_input.dtype)

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
                "unexpected DeepEP low-latency receive shape: "
                f"{tuple(recv_x.shape)}"
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
    ) -> torch.Tensor:
        from rtp_llm.models_py.distributed.deepep_wrapper import DeepEPMode

        wrapper = self._deepep_wrapper()
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
        return self._tp_gather(
            local_output, routed_input.shape[0], tokens_per_tp_rank
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expert_ids, routing_weights = self._route(hidden_states)
        record_accuracy_tensor(
            f"{self.trace_prefix}.expert_ids", expert_ids, token_dim=0
        )
        record_accuracy_tensor(
            f"{self.trace_prefix}.routing_weights",
            routing_weights,
            token_dim=0,
        )
        record_accuracy_tensor(
            f"{self.trace_prefix}.router_counts",
            torch.bincount(
                expert_ids.reshape(-1), minlength=self.expert_num
            ),
        )
        routed_input = _linear(
            hidden_states, self.weights[K3W.MOE_ROUTED_DOWN]
        )
        record_accuracy_tensor(
            f"{self.trace_prefix}.routed_input", routed_input, token_dim=0
        )
        routed_output = (
            self._distributed_expert_sum(
                routed_input, expert_ids, routing_weights
            )
            if self.ep_size > 1
            else self._local_expert_sum(
                routed_input,
                expert_ids,
                routing_weights,
                ids_are_local=False,
            )
        )
        record_accuracy_tensor(
            f"{self.trace_prefix}.expert_sum", routed_output, token_dim=0
        )
        if self.latent_moe_use_norm:
            routed_output = _rms_norm(
                routed_output, self.weights[K3W.MOE_ROUTED_NORM], self.eps
            )
        record_accuracy_tensor(
            f"{self.trace_prefix}.routed_normalized",
            routed_output,
            token_dim=0,
        )
        routed_output = _linear(
            routed_output, self.weights[K3W.MOE_ROUTED_UP]
        )
        record_accuracy_tensor(
            f"{self.trace_prefix}.routed_output", routed_output, token_dim=0
        )
        shared_gate = _linear(hidden_states, self.weights[K3W.MOE_SHARED_GATE])
        shared_up = _linear(hidden_states, self.weights[K3W.MOE_SHARED_UP])
        shared_activation = _situ(
            shared_gate,
            shared_up,
            self.beta,
            self.linear_beta,
        )
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
        shared_output = _row_parallel_linear(
            shared_activation,
            self.weights[K3W.MOE_SHARED_DOWN],
            self.parallelism_config.get_ffn_tp_size(),
        )
        record_accuracy_tensor(
            f"{self.trace_prefix}.shared_output", shared_output, token_dim=0
        )
        output = routed_output + shared_output
        record_accuracy_tensor(
            f"{self.trace_prefix}.output", output, token_dim=0
        )
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trace_prefix = f"layer.{self.layer_idx}"
        record_accuracy_tensor(
            f"{trace_prefix}.input", hidden_states, token_dim=0
        )
        prefix_sum: Optional[torch.Tensor] = hidden_states
        if block_residual.shape[1] > 0:
            hidden_states = _attention_residual(
                prefix_sum,
                block_residual,
                self.weights[K3W.SELF_ATTN_RES_NORM],
                self.weights[K3W.SELF_ATTN_RES_PROJ],
                self.eps,
            )
        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual = torch.cat(
                (block_residual, prefix_sum.unsqueeze(1)), dim=1
            )
            prefix_sum = None
        attention_input = _rms_norm(
            hidden_states, self.weights[W.pre_ln_gamma], self.eps
        )
        record_accuracy_tensor(
            f"{trace_prefix}.attention_input", attention_input, token_dim=0
        )
        if self.is_kda:
            attention_output, _ = self.self_attn(
                attention_input,
                cu_seqlens,
                mode=mode,
                kv_cache=kv_cache,
                attention_inputs=attention_inputs,
            )
        else:
            # MLA layers use the shared ``MlaAttention`` signature and consume
            # the framework fmha_impl built by ``prepare_fmha_impl``.
            attention_output = self.self_attn(
                attention_input,
                fmha_impl,
                kv_cache,
            )
        prefix_sum = (
            attention_output
            if prefix_sum is None
            else prefix_sum + attention_output
        )
        mlp_input = _attention_residual(
            prefix_sum,
            block_residual,
            self.weights[K3W.MLP_RES_NORM],
            self.weights[K3W.MLP_RES_PROJ],
            self.eps,
        )
        record_accuracy_tensor(
            f"{trace_prefix}.mlp_input", mlp_input, token_dim=0
        )
        normalized_mlp_input = _rms_norm(
            mlp_input, self.weights[W.post_ln_gamma], self.eps
        )
        record_accuracy_tensor(
            f"{trace_prefix}.normalized_mlp_input",
            normalized_mlp_input,
            token_dim=0,
        )
        mlp_output = self.mlp(normalized_mlp_input)
        output = prefix_sum + mlp_output
        record_accuracy_tensor(
            f"{trace_prefix}.output", output, token_dim=0
        )
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
        self.output_attn_res_norm = weights.get_global_weight(
            K3W.OUTPUT_ATTN_RES_NORM
        )
        self.output_attn_res_proj = weights.get_global_weight(
            K3W.OUTPUT_ATTN_RES_PROJ
        )

    # ``prepare_fmha_impl`` is inherited from ``GptModelBase``: it builds the
    # framework MLA impl via ``AttnImplFactory.get_fmha_impl`` (identical to the
    # generic MoE path).  K3's MLA layers consume that impl through
    # ``KimiK3MLA`` (an ``MlaAttention`` subclass); K3's KDA layers ignore it.

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = F.embedding(input_ids, self.embedding_weight)
        if self.parallelism_config.get_attn_tp_size() > 1:
            tokens, local_hidden = hidden_states.shape
            hidden_states = all_gather(hidden_states, group=Group.TP)
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
        _sequence_offsets(cu_seqlens, input_ids.numel())
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
        if attention_inputs.is_target_verify:
            raise RuntimeError(
                "Kimi K3 target-verify cache semantics are not connected yet"
            )
        if not attention_inputs.is_prefill and self.kv_cache is None:
            raise RuntimeError("Kimi K3 decode requires an initialized hybrid cache")
        input_ids = inputs.input_ids.reshape(-1)
        cu_seqlens = self._cu_seqlens(attention_inputs, input_ids)
        mark_accuracy_fake_stream(
            KimiK3LinearCacheAdapter._is_fake_stream(attention_inputs),
            input_ids.device,
        )
        record_accuracy_tensor("input_ids", input_ids, token_dim=0)
        record_accuracy_tensor("cu_seqlens", cu_seqlens)
        hidden_states = self._embed(input_ids)
        record_accuracy_tensor("embedding", hidden_states, token_dim=0)
        block_residual = hidden_states.new_empty(
            hidden_states.shape[0], 0, hidden_states.shape[1]
        )
        mode: KDAExecutionMode = (
            "prefill" if attention_inputs.is_prefill else "decode"
        )
        write_cache_store_impl = create_write_cache_store_impl(
            attention_inputs, self.kv_cache
        )
        for layer_idx, layer in enumerate(self.layers):
            select_block_map_for_layer(attention_inputs, layer_idx)
            layer_cache = (
                self.kv_cache.get_layer_cache(layer_idx)
                if self.kv_cache is not None
                else None
            )
            hidden_states, block_residual = layer(
                hidden_states,
                block_residual,
                cu_seqlens,
                mode=mode,
                kv_cache=layer_cache,
                attention_inputs=attention_inputs,
                fmha_impl=fmha_impl,
            )
            # Loop-level cache-store is only for KDA layers.  MLA layers now run
            # through ``MlaAttention`` + ``MlaFlashInfer*Impl``, whose ``forward``
            # already calls ``apply_write_cache_store`` internally -- writing here
            # too would store the MLA latent cache twice (corrupting PD transfer).
            # This matches ``GenericMoeModel``, which has no loop-level write.
            if (
                layer.is_kda
                and write_cache_store_impl is not None
                and layer_cache is not None
            ):
                # The shared writer selects pinned-host length mirrors prepared
                # by PyWrappedModel.  Passing the CUDA length tensors directly
                # is unsafe because PD cache-store consumes them on a CPU
                # background thread.  Its physical block table remains 3-D;
                # the C++ writer maps this layer to the KDA cache group.
                layer_cache.cache_store_segment_sizes = list(
                    layer.self_attn.cache_adapter.cache_store_segment_sizes
                )
                write_cache_store_impl(layer_cache)
        hidden_states = _attention_residual(
            hidden_states,
            block_residual,
            self.output_attn_res_norm,
            self.output_attn_res_proj,
            self.config.layernorm_eps,
        )
        record_accuracy_tensor("output_attn_res", hidden_states, token_dim=0)
        hidden_states = _rms_norm(
            hidden_states, self.final_norm_weight, self.config.layernorm_eps
        )
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
