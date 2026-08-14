"""Framework-facing Kimi K3 model orchestration.

This module contains the ``GptModelBase`` integration, decoder-layer
composition and the top-level packed-token/cache-group loop. Operators live under
``rtp_llm.models_py.modules.kimi_k3``:

* ``kda`` / ``cache``: linear attention and its paged recurrent state;
* ``mla``: gated MLA integration over the framework attention backends;
* ``moe``: the K3 latent-expert feed-forward layer;
* shared distributed modules: reusable execution infrastructure.

Like ``KimiLinearModel``, this module composes those components but does not
own their operator implementations.
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

import torch
from torch import nn

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3WeightNames as K3W
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather_trim,
    barrier,
    get_process_group,
)
from rtp_llm.models_py.distributed.symm_mem import (
    reserve_fused_all_gather_matmul_workspace,
)
from rtp_llm.models_py.distributed.sequence_parallel import (
    TokenShardLayout,
    shard_tokens,
    shard_tokens_with_padding,
    token_shard_layout,
)
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.kimi_k3_chunk_planner import (
    KimiK3ChunkRound,
    plan_kimi_k3_chunk_rounds,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.base import RMSNorm
from rtp_llm.models_py.modules.base.common.embedding import Embedding
from rtp_llm.models_py.modules.base.common.kvcache_store import (
    create_write_cache_store_impl,
)
from rtp_llm.models_py.modules.base.common.multimodal_embedding import (
    MultimodalEmbeddingInjector,
)
from rtp_llm.models_py.modules.hybrid.dense_mlp import (
    DenseMLP,
    DenseMLPParallelExecutor,
)
from rtp_llm.models_py.modules.factory.linear.parallel import (
    should_use_fused_all_gather_matmul,
)
from rtp_llm.models_py.modules.kimi_k3.kda.state import KDAExecutionMode
from rtp_llm.models_py.triton_kernels.common.activation import SituAndMul
from rtp_llm.ops import HybridAttentionType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W

if TYPE_CHECKING:
    from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3ModelConfig

from rtp_llm.models_py.modules.kimi_k3.kda import KimiK3KDA
from rtp_llm.models_py.modules.kimi_k3.mla import (
    KimiK3MLA,
    prepare_mla_fmha_for_group,
)
from rtp_llm.models_py.modules.kimi_k3.moe import KimiK3LatentMoE
from rtp_llm.models_py.modules.kimi_k3.residual import KimiK3AttentionResidual
from rtp_llm.models_py.modules.kimi_k3.sequence import sequence_offsets


def _prefill_chunk_tokens() -> int:
    """Return the opt-in whole-model Prefill chunk size."""

    raw = os.environ.get("KIMI_K3_PREFILL_CHUNK_TOKENS", "0").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "KIMI_K3_PREFILL_CHUNK_TOKENS must be a non-negative integer, "
            f"got {raw!r}"
        ) from exc
    if value < 0:
        raise ValueError(
            "KIMI_K3_PREFILL_CHUNK_TOKENS must be non-negative, "
            f"got {value}"
        )
    return value


def _fused_ag_workspace_global_tokens(
    max_seq_len: int,
    max_context_batch_size: int,
    chunk_tokens: int,
) -> int:
    """Bound the symmetric AG/GEMM workspace by one model invocation."""

    configured_tokens = int(max_seq_len) * int(max_context_batch_size)
    if chunk_tokens <= 0:
        return configured_tokens
    return min(
        configured_tokens,
        int(chunk_tokens) * int(max_context_batch_size),
    )


@dataclass(frozen=True)
class KimiK3DecoderMetadata:
    """Request-scoped execution state shared by every decoder layer."""

    cu_seqlens: torch.Tensor
    mode: KDAExecutionMode
    sequence_parallel: bool
    prefill_sp_layout: Optional[TokenShardLayout] = None


@dataclass(frozen=True)
class KimiK3DecoderOutput:
    """State carried from one K3 decoder layer to the next."""

    hidden_states: torch.Tensor
    block_residual: torch.Tensor


class KimiK3FinalNorm(nn.Module):
    """Finish K3's block residual and apply the decoder output RMSNorm."""

    def __init__(
        self,
        attn_res_norm: torch.Tensor,
        attn_res_proj: torch.Tensor,
        final_norm: torch.Tensor,
        eps: float,
    ) -> None:
        super().__init__()
        self.attention_residual = KimiK3AttentionResidual(
            attn_res_norm,
            attn_res_proj,
            eps,
        )
        self.final_norm = RMSNorm(final_norm, eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> torch.Tensor:
        return self.attention_residual(
            hidden_states,
            block_residual,
            output_norm_weight=self.final_norm.weight,
            output_norm_eps=self.final_norm.variance_epsilon,
        )


class KimiK3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiK3ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        hw_kernel_config: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.layer_idx = int(layer_idx)
        self.eps = float(config.layernorm_eps)
        self.attn_res_block_size = config.k3_runtime_config.attn_res_block_size
        self.layer_type = config.hybrid_attention_config.hybrid_attention_types[
            layer_idx
        ]
        self.self_attention_residual = KimiK3AttentionResidual(
            weights[K3W.SELF_ATTN_RES_NORM],
            weights[K3W.SELF_ATTN_RES_PROJ],
            self.eps,
        )
        self.mlp_residual = KimiK3AttentionResidual(
            weights[K3W.MLP_RES_NORM],
            weights[K3W.MLP_RES_PROJ],
            self.eps,
        )
        self.attention_norm = RMSNorm(weights[W.pre_ln_gamma], self.eps)
        self.mlp_norm = RMSNorm(weights[W.post_ln_gamma], self.eps)
        self.self_attn: nn.Module = (
            KimiK3KDA(config, parallelism_config, weights, layer_idx)
            if self.is_kda
            else KimiK3MLA(config, parallelism_config, weights, layer_idx)
        )
        self.mlp: nn.Module = (
            KimiK3LatentMoE(config, parallelism_config, weights, layer_idx)
            if layer_idx in set(config.moe_layer_index)
            else DenseMLP(
                config.activation_type,
                parallelism_config,
                weights,
                config.quant_config,
                hw_kernel_config=hw_kernel_config,
                gated_activation=SituAndMul(
                    config.k3_runtime_config.activation_situ_beta,
                    config.k3_runtime_config.activation_situ_linear_beta,
                ),
                merge_gate_up=False,
                parallel_executor=DenseMLPParallelExecutor(
                    weights,
                    parallelism_config,
                    gated=True,
                ),
            )
        )

    @property
    def is_kda(self) -> bool:
        return self.layer_type == HybridAttentionType.LINEAR

    def prepare_mla_fmha_for_group(
        self,
        fmha_impl: Any,
        attention_inputs: PyAttentionInputs,
        group_id: int,
        prepared_group_id: Optional[int],
    ) -> Optional[int]:
        if self.is_kda or fmha_impl is None:
            return prepared_group_id
        return prepare_mla_fmha_for_group(
            fmha_impl,
            attention_inputs,
            group_id,
            prepared_group_id,
        )

    def prepare_kda_cache_store(self, kv_cache: LayerKVCache) -> None:
        if not self.is_kda:
            raise RuntimeError("only KDA layers publish recurrent cache segments")
        kv_cache.cache_store_segment_sizes = list(
            self.self_attn.cache_store_segment_sizes
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        *,
        attn_meta: KimiK3DecoderMetadata,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        fmha_impl: Any = None,
    ) -> KimiK3DecoderOutput:
        cu_seqlens = attn_meta.cu_seqlens
        mode = attn_meta.mode
        sequence_parallel = attn_meta.sequence_parallel
        prefill_sp_layout = attn_meta.prefill_sp_layout
        decode_sp = sequence_parallel and mode == "decode"
        logical_tokens = int(hidden_states.shape[0])
        tp_size = int(self.self_attn.parallelism_config.get_attn_tp_size())
        tp_rank = int(self.self_attn.parallelism_config.get_attn_tp_rank())
        local_valid_tokens: Optional[int] = None
        if (
            prefill_sp_layout is not None
            and prefill_sp_layout.local_valid_tokens < prefill_sp_layout.local_tokens
        ):
            local_valid_tokens = prefill_sp_layout.local_valid_tokens
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
        if previous_blocks > 0 or block_write_idx >= 0:
            attention_input = self.self_attention_residual(
                prefix_sum,
                block_residual,
                output_norm_weight=self.attention_norm.weight,
                output_norm_eps=self.attention_norm.variance_epsilon,
                num_blocks=previous_blocks,
                block_write_idx=block_write_idx,
            )
        else:
            attention_input = self.attention_norm(hidden_states.contiguous())
        if writes_block:
            if block_write_idx < 0:
                raise RuntimeError("K3 AttnRes residual bank has no writable block")
            prefix_sum = None
        active_blocks = previous_blocks + int(writes_block)
        active_block_residual = block_residual[:, :active_blocks]
        if self.is_kda:
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
            attention_output = self.self_attn(
                attention_input,
                fmha_impl,
                kv_cache,
                attention_inputs=attention_inputs,
                sequence_parallel=sequence_parallel,
                prefill_sp_layout=prefill_sp_layout,
            )
        if decode_sp:
            if prefix_sum is not None:
                prefix_sum, local_valid_tokens = shard_tokens_with_padding(
                    prefix_sum,
                    logical_tokens,
                    tp_size,
                    tp_rank,
                )
            active_block_residual, block_valid_tokens = shard_tokens_with_padding(
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
        if prefix_sum is None:
            prefix_sum = attention_output
        elif prefix_sum.is_cuda:
            attention_delta = attention_output
        else:
            prefix_sum = prefix_sum + attention_output
        normalized_mlp_input = self.mlp_residual(
            prefix_sum,
            active_block_residual,
            output_norm_weight=self.mlp_norm.weight,
            output_norm_eps=self.mlp_norm.variance_epsilon,
            delta=attention_delta,
            num_blocks=active_blocks,
        )
        mlp_output = self.mlp(
            normalized_mlp_input,
            sequence_parallel=sequence_parallel,
            valid_token_count=local_valid_tokens,
        )
        output = prefix_sum + mlp_output
        if decode_sp:
            output = all_gather_trim(output, logical_tokens, group=Group.TP)
        return KimiK3DecoderOutput(output, block_residual)


def _mask_multimodal_token_ids(
    input_ids: torch.Tensor,
    multimodal_features: Sequence[torch.Tensor],
    multimodal_locs: torch.Tensor,
) -> torch.Tensor:
    """Zero the token ids that ``MultimodalEmbeddingInjector`` will overwrite.

    Multimodal rows do not hold vocab ids: ``MultimodalProcessor::expandTokenIds``
    replaces them with per-row feature hashes (``featureHashToTokenId``, an arbitrary
    ``int32`` that is routinely negative or ``>= vocab_size``) so the prefix cache can
    tell two images apart. Feeding those to the embedding op indexes out of bounds, so
    they must be masked before lookup. The zeroed rows are overwritten by the injector.
    """
    locs = multimodal_locs.to(device="cpu", dtype=torch.long).view(-1).tolist()
    masked_ids = input_ids.clone()
    for feature, loc in zip(multimodal_features, locs):
        if feature is None:
            continue
        # loc < 0 means the head rows already live in the reused KV prefix and only
        # the tail lands in this chunk, at token 0 -- same convention as the injector.
        offset = max(loc, 0)
        length = feature.size(0) - min(max(-loc, 0), feature.size(0))
        # Out-of-range spans are clipped rather than rejected here; the injector
        # raises the canonical IndexError after the embedding lookup.
        length = min(length, masked_ids.size(0) - offset)
        if length > 0:
            masked_ids.narrow(0, offset, length).fill_(0)
    return masked_ids


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
        self.embed_tokens = Embedding(
            model_config,
            parallelism_config,
            weights.get_global_weight(W.embedding),
        )
        self.embedding_weight = self.embed_tokens.weight
        self.attn_res_block_size = int(
            model_config.k3_runtime_config.attn_res_block_size
        )
        self.num_attn_res_blocks = (
            self.layer_num + self.attn_res_block_size - 1
        ) // self.attn_res_block_size
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        self.layers = nn.ModuleList(
            [
                KimiK3DecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[layer_idx],
                    layer_idx,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for layer_idx in range(self.layer_num)
            ]
        )
        self.norm = KimiK3FinalNorm(
            weights.get_global_weight(K3W.OUTPUT_ATTN_RES_NORM),
            weights.get_global_weight(K3W.OUTPUT_ATTN_RES_PROJ),
            weights.get_global_weight(W.final_ln_gamma),
            model_config.layernorm_eps,
        )
        self._layer_group_ids: Optional[tuple[int, ...]] = None
        self._fused_ag_gemm_workspace_ready = False
        self._max_generate_batch_size = int(max_generate_batch_size)
        self._is_decode_role = False
        self._mtp_hidden_buffer: Optional[torch.Tensor] = None
        self._mtp_hidden_valid_tokens = 0
        self._prefill_static_attn_res_bank: Optional[torch.Tensor] = None

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        """Bind runtime resources and reserve the largest Prefill AG workspace."""

        super().initialize(init_resource)
        self._is_decode_role = bool(init_resource.is_decode_role)
        if (
            self._is_decode_role
            and self._mtp_hidden_buffer is None
            and os.environ.get("SP_TYPE", "").lower() == "eagle3"
        ):
            tokens_per_batch = max(int(self.config.gen_num_per_cycle) + 1, 1)
            token_capacity = self._max_generate_batch_size * tokens_per_batch
            self._mtp_hidden_buffer = self.embedding_weight.new_empty(
                token_capacity,
                3 * int(self.config.hidden_size),
            )
            logging.info(
                "[K3_EAGLE3] allocated Decode hidden buffer shape=%s",
                tuple(self._mtp_hidden_buffer.shape),
            )
        if self._fused_ag_gemm_workspace_ready:
            return True

        tp_size = int(self.parallelism_config.get_attn_tp_size())
        max_global_tokens = _fused_ag_workspace_global_tokens(
            int(self.config.max_seq_len),
            int(init_resource.max_context_batch_size),
            _prefill_chunk_tokens(),
        )
        max_local_tokens = (max_global_tokens + tp_size - 1) // tp_size
        max_physical_tokens = max_local_tokens * tp_size
        if (
            init_resource.is_decode_role
            or tp_size <= 1
            or not should_use_fused_all_gather_matmul(max_physical_tokens)
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

    def _write_mtp_hidden_buffer(
        self, hidden_states: torch.Tensor, *, is_cuda_graph: bool
    ) -> None:
        rows = int(hidden_states.size(0))
        if self._is_decode_role:
            buffer = self._mtp_hidden_buffer
            if buffer is None:
                raise RuntimeError(
                    "Kimi K3 EAGLE Decode hidden buffer was not initialized"
                )
            if hidden_states.shape[1:] != buffer.shape[1:]:
                raise ValueError(
                    "Kimi K3 EAGLE hidden width does not match the Decode buffer: "
                    f"hidden={tuple(hidden_states.shape)}, buffer={tuple(buffer.shape)}"
                )
            if rows > buffer.size(0):
                raise ValueError(
                    f"Kimi K3 EAGLE hidden rows {rows} exceed Decode capacity "
                    f"{buffer.size(0)}"
                )
            # CUDA Graph replay does not execute Python. Every captured graph must
            # therefore write into the same model-owned storage instead of replacing
            # this attribute with the tensor from one captured batch shape.
            buffer.narrow(0, 0, rows).copy_(hidden_states)
        else:
            # Prefill can have far more rows than the Decode graph budget and is not
            # graph-captured, so retain its exact-size tensor without a large reserve.
            self._mtp_hidden_buffer = hidden_states
        if not is_cuda_graph:
            self._mtp_hidden_valid_tokens = rows

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

    def _ensure_prefill_static_attn_res_bank(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Return a model-owned AttnRes bank reused across Prefill chunks."""

        tp_size = int(self.parallelism_config.get_attn_tp_size())
        chunk_tokens = _prefill_chunk_tokens()
        chunk_local_rows = (
            (chunk_tokens + tp_size - 1) // tp_size if chunk_tokens > 0 else 0
        )
        required_rows = int(hidden_states.shape[0])
        capacity_rows = max(required_rows, chunk_local_rows)
        required_shape = (
            capacity_rows,
            int(self.num_attn_res_blocks),
            int(hidden_states.shape[1]),
        )
        bank = self._prefill_static_attn_res_bank
        if (
            bank is None
            or bank.device != hidden_states.device
            or bank.dtype != hidden_states.dtype
            or bank.shape[1:] != required_shape[1:]
            or bank.shape[0] < capacity_rows
        ):
            bank = hidden_states.new_empty(required_shape)
            self._prefill_static_attn_res_bank = bank
            logging.info(
                "[K3_PREFILL_ATTN_RES_BANK] allocated shape=%s bytes=%.3fGiB",
                tuple(bank.shape),
                bank.numel() * bank.element_size() / float(1 << 30),
            )
        return bank.narrow(0, 0, required_rows)

    # ``prepare_fmha_impl`` is inherited from ``GptModelBase``: it builds the
    # framework MLA impl via ``AttnImplFactory.get_fmha_impl`` (identical to the
    # generic MoE path).  K3's MLA layers consume that impl through
    # ``KimiK3MLA`` (an ``MlaAttention`` subclass); K3's KDA layers ignore it.

    def _embed(self, input_ids: torch.Tensor, multimodal_inputs: Any) -> torch.Tensor:
        multimodal_features = multimodal_inputs.multimodal_features
        mm_features_locs = multimodal_inputs.mm_features_locs_host
        if multimodal_features:
            if (
                mm_features_locs is None
                or mm_features_locs.numel() != len(multimodal_features)
            ):
                raise ValueError(
                    "Kimi K3 multimodal feature locations must match the feature count"
                )
            input_ids = _mask_multimodal_token_ids(
                input_ids, multimodal_features, mm_features_locs
            )

        hidden_states = self.embed_tokens(input_ids)
        # Vision features use full hidden space, so inject after the TP all-gather
        # in Embedding rather than into its rank-local vocabulary projection.
        return self.multimodal_embedding_injector(
            hidden_states, multimodal_features, mm_features_locs
        )

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
            sequence_offsets(
                cu_seqlens,
                input_ids.numel(),
                cu_seqlens_host=getattr(attention_inputs, "cu_seqlens_host", None),
            )
        return cu_seqlens

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        attention_inputs = inputs.attention_inputs
        input_ids = inputs.input_ids.reshape(-1)
        chunk_tokens = _prefill_chunk_tokens()
        if (
            chunk_tokens > 0
            and attention_inputs is not None
            and attention_inputs.is_prefill
            and input_ids.numel() > chunk_tokens
        ):
            return self._forward_whole_chunk_prefill(inputs, fmha_impl, chunk_tokens)
        return self._forward_impl_one(inputs, fmha_impl)

    def _validate_whole_chunk_prefill(
        self,
        inputs: PyModelInputs,
        chunk_tokens: int,
    ) -> None:
        attention_inputs = inputs.attention_inputs
        assert attention_inputs is not None
        input_ids = inputs.input_ids.reshape(-1)
        tp_size = int(self.parallelism_config.get_attn_tp_size())
        ep_size = int(self.parallelism_config.ep_size)
        if self.kv_cache is None:
            raise RuntimeError("whole-model K3 Prefill requires an initialized cache")
        page_size = int(self.kv_cache.seq_size_per_block)
        if page_size <= 0 or page_size % 64:
            raise RuntimeError(
                "whole-model K3 Prefill requires a positive cache page size "
                "divisible by the cuLA checkpoint step 64; "
                f"page_size={page_size}"
            )
        if chunk_tokens <= 0 or chunk_tokens % tp_size:
            raise RuntimeError(
                "KIMI_K3_PREFILL_CHUNK_TOKENS must be divisible by attention TP; "
                f"chunk={chunk_tokens}, TP={tp_size}"
            )
        if ep_size != tp_size:
            raise RuntimeError(
                "whole-model K3 Prefill requires TP == EP Sequence Parallel; "
                f"TP={tp_size}, EP={ep_size}"
            )
        if bool(getattr(attention_inputs, "is_target_verify", False)):
            raise RuntimeError("whole-model K3 Prefill does not support target verify")
        if bool(getattr(attention_inputs, "is_cuda_graph", False)):
            raise RuntimeError("whole-model K3 Prefill does not support CUDA Graph")
        if getattr(attention_inputs, "context_parallel_info", None) is not None:
            raise RuntimeError(
                "whole-model K3 Prefill does not support framework Prefill CP"
            )
        if bool(getattr(attention_inputs, "need_all_logits", False)):
            raise RuntimeError("whole-model K3 Prefill does not support all logits")
        if bool(getattr(attention_inputs, "need_all_hidden_states", False)):
            raise RuntimeError(
                "whole-model K3 Prefill does not support all hidden states"
            )
        if os.environ.get("SP_TYPE", "").lower() == "eagle3":
            raise RuntimeError("whole-model K3 Prefill does not support EAGLE3/MTP")
        multimodal = inputs.multimodal_inputs
        if multimodal.multimodal_features or (
            multimodal.mm_features_locs_host is not None
            and multimodal.mm_features_locs_host.numel()
        ):
            raise RuntimeError("whole-model K3 Prefill does not support multimodal input")

    @staticmethod
    def _host_lengths(value: torch.Tensor, name: str) -> list[int]:
        if value is None or not value.numel():
            raise RuntimeError(f"whole-model K3 Prefill requires {name}")
        source = value if value.device.type == "cpu" else value.detach().cpu()
        return [int(item) for item in source.tolist()]

    @staticmethod
    def _select_batch_rows(value: torch.Tensor, indices: list[int]) -> torch.Tensor:
        if value is None or not value.numel():
            return value
        index = torch.tensor(indices, dtype=torch.long, device=value.device)
        return value.index_select(0, index).contiguous()

    @classmethod
    def _select_group_batch_rows(
        cls, values: Sequence[torch.Tensor], indices: list[int]
    ) -> list[torch.Tensor]:
        return [cls._select_batch_rows(value, indices) for value in values]

    @staticmethod
    def _chunk_attention_inputs(
        attention_inputs: PyAttentionInputs,
        *,
        round_plan: KimiK3ChunkRound,
        device: torch.device,
    ) -> PyAttentionInputs:
        lengths = [item.new_length for item in round_plan.slices]
        prefixes = [item.absolute_start for item in round_plan.slices]
        sequence_lengths = [item.absolute_end for item in round_plan.slices]
        batch_indices = [item.original_batch_idx for item in round_plan.slices]
        total_tokens = sum(lengths)
        chunk = copy.copy(attention_inputs)
        cu_seqlens = [0]
        cu_kv_seqlens = [0]
        for length, sequence_length in zip(lengths, sequence_lengths):
            cu_seqlens.append(cu_seqlens[-1] + length)
            cu_kv_seqlens.append(cu_kv_seqlens[-1] + sequence_length)
        chunk.cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
        chunk.cu_kv_seqlens = torch.tensor(
            cu_kv_seqlens, dtype=torch.int32, device=device
        )
        chunk.input_lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
        chunk.prefix_lengths = torch.tensor(prefixes, dtype=torch.int32, device=device)
        chunk.sequence_lengths = torch.tensor(
            sequence_lengths, dtype=torch.int32, device=device
        )
        chunk.sequence_lengths_plus_1_d = chunk.sequence_lengths + 1
        max_length = max(lengths)
        padding_offset: list[int] = []
        cumulative_padding = 0
        for length in lengths:
            padding_offset.extend([cumulative_padding] * length)
            cumulative_padding += max_length - length
        chunk.padding_offset = torch.tensor(
            padding_offset, dtype=torch.int32, device=device
        )
        chunk.cu_seqlens_host = torch.tensor(cu_seqlens, dtype=torch.int32)
        chunk.input_lengths_host = torch.tensor(lengths, dtype=torch.int32)
        chunk.prefix_lengths_host = torch.tensor(prefixes, dtype=torch.int32)
        chunk.sequence_lengths_host = torch.tensor(
            sequence_lengths, dtype=torch.int32
        )
        chunk.original_batch_indices_host = torch.tensor(
            batch_indices, dtype=torch.int64
        )
        chunk.total_tokens = int(total_tokens)
        chunk.context_total_kv_length = int(sum(sequence_lengths))
        chunk.is_prefill = True
        chunk.is_cuda_graph = False
        chunk.cache_store_inputs = None
        singular_block_tables = (
            "kv_cache_block_id_host",
            "kv_cache_block_id_device",
            "kv_cache_kernel_block_id_host",
            "kv_cache_kernel_block_id_device",
        )
        for name in singular_block_tables:
            selected = KimiK3Model._select_batch_rows(
                getattr(attention_inputs, name), batch_indices
            )
            # Undefined C++ tensors are exposed as None, but pybind's Tensor
            # setter does not accept assigning None back to the copied object.
            if selected is not None:
                setattr(chunk, name, selected)
        chunk.kv_cache_block_id_host_by_group = (
            KimiK3Model._select_group_batch_rows(
                attention_inputs.kv_cache_block_id_host_by_group, batch_indices
            )
        )
        chunk.kv_cache_kernel_block_id_host_by_group = (
            KimiK3Model._select_group_batch_rows(
                attention_inputs.kv_cache_kernel_block_id_host_by_group,
                batch_indices,
            )
        )
        chunk.kv_cache_kernel_block_id_device_by_group = (
            KimiK3Model._select_group_batch_rows(
                attention_inputs.kv_cache_kernel_block_id_device_by_group,
                batch_indices,
            )
        )
        return chunk

    def _publish_whole_chunk_cache(
        self, attention_inputs: PyAttentionInputs
    ) -> None:
        writer = create_write_cache_store_impl(attention_inputs, self.kv_cache)
        if writer is None or self.kv_cache is None:
            return
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = self.kv_cache.get_layer_cache(layer_idx)
            if layer.is_kda:
                layer.prepare_kda_cache_store(layer_cache)
            writer(layer_cache)

    def _forward_whole_chunk_prefill(
        self,
        inputs: PyModelInputs,
        fmha_impl: Any,
        chunk_tokens: int,
    ) -> PyModelOutputs:
        self._validate_whole_chunk_prefill(inputs, chunk_tokens)
        input_ids = inputs.input_ids.reshape(-1)
        attention_inputs = inputs.attention_inputs
        assert attention_inputs is not None
        total_tokens = int(input_ids.numel())
        input_lengths = self._host_lengths(
            attention_inputs.input_lengths_host, "input_lengths_host"
        )
        prefix_lengths = self._host_lengths(
            attention_inputs.prefix_lengths_host, "prefix_lengths_host"
        )
        if sum(input_lengths) != total_tokens:
            raise RuntimeError(
                "whole-model K3 packed lengths do not cover input tokens: "
                f"lengths={sum(input_lengths)} tokens={total_tokens}"
            )
        page_size = int(self.kv_cache.seq_size_per_block)
        rounds = plan_kimi_k3_chunk_rounds(
            input_lengths,
            prefix_lengths,
            chunk_budget=chunk_tokens,
            page_size=page_size,
        )
        barrier(Group.TP)
        logging.info(
            "[K3_WHOLE_CHUNK_PREFILL] enabled total_tokens=%d "
            "requests=%d rounds=%d chunk_tokens=%d page_size=%d TP=%d EP=%d",
            total_tokens,
            len(input_lengths),
            len(rounds),
            chunk_tokens,
            page_size,
            int(self.parallelism_config.get_attn_tp_size()),
            int(self.parallelism_config.ep_size),
        )
        if self._layer_group_ids is None:
            layer_map_host = getattr(
                attention_inputs, "kv_cache_layer_to_group_host", None
            )
            if layer_map_host is not None and layer_map_host.numel():
                self._layer_group_ids = tuple(
                    int(value) for value in layer_map_host.tolist()
                )
        terminal_hidden: list[Optional[torch.Tensor]] = [None] * len(input_lengths)
        final_params: Any = None
        initialized_kda_modules: list[KimiK3KDA] = []
        try:
            for layer_idx, layer in enumerate(self.layers):
                if layer.is_kda and isinstance(layer.self_attn, KimiK3KDA):
                    static_group_id = (
                        self._layer_group_ids[layer_idx]
                        if self._layer_group_ids is not None
                        and layer_idx < len(self._layer_group_ids)
                        else None
                    )
                    select_block_map_for_layer(
                        attention_inputs, layer_idx, static_group_id
                    )
                    layer_cache = self.kv_cache.get_layer_cache(layer_idx)
                    if int(layer_cache.seq_size_per_block) != page_size:
                        raise RuntimeError(
                            "whole-model K3 KDA/cache checkpoint step mismatch: "
                            f"layer={layer_idx} linear_page="
                            f"{layer_cache.seq_size_per_block} "
                            f"physical_page={page_size}"
                        )
                    layer.self_attn.begin_whole_chunk_prefill(
                        layer_cache,
                        attention_inputs,
                        attention_inputs.cu_seqlens,
                        len(input_lengths),
                    )
                    initialized_kda_modules.append(layer.self_attn)
            for round_plan in rounds:
                chunk_attention = self._chunk_attention_inputs(
                    attention_inputs,
                    round_plan=round_plan,
                    device=input_ids.device,
                )
                chunk_inputs = PyModelInputs()
                chunk_inputs.input_ids = torch.cat(
                    [
                        input_ids.narrow(
                            0, item.source_start, item.new_length
                        )
                        for item in round_plan.slices
                    ],
                    dim=0,
                )
                chunk_inputs.attention_inputs = chunk_attention
                round_output = self._forward_impl_one(chunk_inputs, fmha_impl)
                final_params = getattr(round_output, "params_ptr", None)
                packed_end = 0
                for item in round_plan.slices:
                    packed_end += item.new_length
                    if item.terminal:
                        terminal_hidden[item.original_batch_idx] = (
                            round_output.hidden_states[packed_end - 1 : packed_end]
                        )
                del chunk_inputs
                del chunk_attention
                del round_output
            self._publish_whole_chunk_cache(attention_inputs)
        finally:
            for module in initialized_kda_modules:
                module.end_whole_chunk_prefill()
        if any(value is None for value in terminal_hidden):
            missing = [idx for idx, value in enumerate(terminal_hidden) if value is None]
            raise RuntimeError(f"whole-model K3 missing terminal rows for {missing}")
        hidden = torch.cat(
            [value for value in terminal_hidden if value is not None], dim=0
        )
        result = (
            PyModelOutputs(hidden, final_params)
            if final_params is not None
            else PyModelOutputs(hidden)
        )
        result.lm_output_already_selected = True
        return result

    def _forward_impl_one(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> PyModelOutputs:
        attention_inputs = inputs.attention_inputs
        if attention_inputs is None:
            raise ValueError("Kimi K3 requires PyAttentionInputs")
        if not attention_inputs.is_prefill and self.kv_cache is None:
            raise RuntimeError("Kimi K3 decode requires an initialized hybrid cache")
        input_ids = inputs.input_ids.reshape(-1)
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
            token_shard_layout(int(input_ids.numel()), tp_size, tp_rank)
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
                "[K3_DECODE_SP] rank=%d requested=%s active=%s "
                "tokens=%d tp=%d ep=%d",
                int(self.parallelism_config.get_attn_tp_rank()),
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
        hidden_states = self._embed(input_ids, inputs.multimodal_inputs)
        if prefill_sp:
            assert prefill_sp_layout is not None
            hidden_states = shard_tokens(
                hidden_states,
                prefill_sp_layout,
            )
        block_residual = (
            self._ensure_prefill_static_attn_res_bank(hidden_states)
            if prefill_sp
            else hidden_states.new_empty(
                hidden_states.shape[0],
                self.num_attn_res_blocks,
                hidden_states.shape[1],
            )
        )
        # MTP target verification is represented as a packed multi-token
        # attention batch, so generic attention metadata may classify it as
        # prefill-shaped. KDA must nevertheless replay it through the paged
        # Decode path and update the Decode-owned recurrent state.
        mode: KDAExecutionMode = (
            "prefill"
            if attention_inputs.is_prefill and not is_target_verify
            else "decode"
        )
        attn_meta = KimiK3DecoderMetadata(
            cu_seqlens=cu_seqlens,
            mode=mode,
            sequence_parallel=sp_active,
            prefill_sp_layout=prefill_sp_layout,
        )
        write_cache_store_impl = create_write_cache_store_impl(
            attention_inputs, self.kv_cache
        )
        prepared_mla_group_id: Optional[int] = None
        if self._layer_group_ids is None:
            layer_map_host = getattr(
                attention_inputs, "kv_cache_layer_to_group_host", None
            )
            if layer_map_host is not None and layer_map_host.numel():
                self._layer_group_ids = tuple(
                    int(value) for value in layer_map_host.tolist()
                )
        eagle3_hidden_states = []
        eagle3_enabled = os.environ.get("SP_TYPE", "").lower() == "eagle3"
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
            if not layer.is_kda and fmha_impl is not None:
                prepared_mla_group_id = layer.prepare_mla_fmha_for_group(
                    fmha_impl,
                    attention_inputs,
                    selected_group_id,
                    prepared_mla_group_id,
                )
            layer_output = layer(
                hidden_states,
                block_residual,
                attn_meta=attn_meta,
                kv_cache=layer_cache,
                attention_inputs=attention_inputs,
                fmha_impl=fmha_impl,
            )
            hidden_states = layer_output.hidden_states
            block_residual = layer_output.block_residual
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
                layer.prepare_kda_cache_store(layer_cache)
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
            self._write_mtp_hidden_buffer(
                mtp_hidden_buffer,
                is_cuda_graph=(
                    bool(getattr(attention_inputs, "is_cuda_graph", False))
                    or (input_ids.is_cuda and torch.cuda.is_current_stream_capturing())
                ),
            )
        hidden_states = self.norm(hidden_states, block_residual)
        if prefill_sp:
            assert prefill_sp_layout is not None
            hidden_states = all_gather_trim(
                hidden_states,
                prefill_sp_layout.logical_tokens,
                group=Group.TP,
            )
        fmha_params = getattr(fmha_impl, "fmha_params", None)
        return (
            PyModelOutputs(hidden_states, fmha_params)
            if fmha_params is not None
            else PyModelOutputs(hidden_states)
        )

__all__ = [
    "KimiK3DecoderLayer",
    "KimiK3DecoderMetadata",
    "KimiK3DecoderOutput",
    "KimiK3FinalNorm",
    "KimiK3Model",
]
