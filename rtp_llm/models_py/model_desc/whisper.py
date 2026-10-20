"""Whisper decoder (A2/A3): 32-layer pre-LN decoder with per-layer cross attention.

Engine contract (GptModelBase): forward(inputs, fmha_impl) -> PyModelOutputs
returning post-final-norm hidden [total_tokens, hidden]; C++ applies lm_head.

Cross-attention KV (Phase B, engine cache group):
- The cross K/V live in the engine KV cache group tagged ``cross`` declared by
  ``Whisper._post_build_model_config`` (kv_len_from_mm_features=True): blocks
  are request-private, sized by the audio frame count, written once at prefill
  and read by paged decode every step. No first-block-id aliasing: the group
  never joins the block tree.
- At prefill, ``PyMultimodalInputs.multimodal_features`` carries the audio
  encoder hidden [L, 1280] per request (mm_skip_token_expansion keeps the
  token stream untouched). The model computes per-layer cross K/V once,
  writes them into the ``cross`` group, and computes prefill cross attention
  from the freshly computed (contiguous, zero-copy) tensors.
- fp16 inf/nan clamp after every residual add, like modeling_whisper.py:635.

Fused kernels (Phase C, whisper_fused_kernels.py):
- Every residual add runs as one Triton kernel (add + fp16 nan_to_num guard,
  bit-exact vs the eager pair) instead of the ATen add + nan_to_num launches.
- Prefill cross attention runs through a shared FlashInfer ragged-prefill
  runner (non-causal, per-row K/V, sm_scale in the kernel) instead of the
  eager matmul-softmax-matmul; decode cross attention reads the ``cross``
  cache group through FlashInfer paged decode. The eager reference math
  remains as the fallback for non-CUDA devices, non-fp16 dtypes and
  FlashInfer failures.
"""

import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import (
    get_attention_inputs_value,
    select_attention_inputs_for_tag,
    select_fmha_impl_for_group,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.whisper_fused_kernels import (
    WhisperCrossAttnRaggedRunner,
    fused_residual_clamp,
    fuse_row_kv,
)
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FMHAImplBase,
    LayerNorm,
)
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)


def _clamped_residual_add(residual: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # Phase C: one Triton kernel replaces the ATen add + nan_to_num pair
    # (bit-exact; eager fallback for non-fp16/CPU/non-contiguous inputs).
    return fused_residual_clamp(residual, x)


CROSS_KV_TAG = "cross"


class _GraphDecodeState:
    """Per-batch-size CUDA graph decode state (params + wrapper pair).

    One FlashInfer paged-decode wrapper can only hold one fixed batch size,
    so each captured graph bucket owns its pair; the buffers stay at fixed
    addresses across replays.
    """

    def __init__(self, params, wrapper, workspace, page_size: int):
        self.params = params
        self.wrapper = wrapper
        self.workspace = workspace
        self.page_size = page_size
        self._plan_indptr_snapshot = None
        self.plan_kv_len = 0
        # Device-resident constant kv-length vector: torch's non_blocking
        # host-to-device .to() rides the allocator copy stream, which is not
        # ordered with the graph stream (an async-pipeline race). Keep the
        # frames on device and refresh the constant with a device-side fill
        # on the current (graph) stream instead.
        self.frames_dev = None

    def bind(self, kv_lens: torch.Tensor, block_table: torch.Tensor) -> None:
        """Capture-time fill + fixed-buffer binding for one batch bucket."""
        self.plan_kv_len = int(kv_lens.max()) if kv_lens.numel() > 0 else 0
        self.frames_dev = torch.full(
            (max(int(kv_lens.numel()), 1),), self.plan_kv_len, dtype=torch.int32, device=block_table.device
        )
        seq_minus_one = (kv_lens - 1).clamp(min=0)
        empty_prefix = torch.empty(0, dtype=torch.int32)
        self.params.fill_params(empty_prefix, seq_minus_one, seq_minus_one, block_table, self.page_size, False)
        if self.wrapper._fixed_batch_size == 0:
            self.wrapper._use_cuda_graph = True
            self.wrapper._paged_kv_indptr_buf = self.params.decode_page_indptr_d
            self.wrapper._paged_kv_last_page_len_buf = self.params.paged_kv_last_page_len_d
            self.wrapper._paged_kv_indices_buf = self.params.page_indice_d
            self.wrapper._fixed_batch_size = int(kv_lens.numel())
            self.wrapper._qo_indptr_buf = torch.arange(
                int(kv_lens.numel()) + 1, dtype=torch.int32, device=self.params.page_indice_d.device
            )

    def plan(self, head_num: int, head_dim: int, dtype: torch.dtype) -> None:
        self.wrapper.plan(
            self.params.decode_page_indptr_d,
            self.params.page_indice_d,
            self.params.paged_kv_last_page_len_d,
            head_num,
            head_num,
            head_dim,
            self.page_size,
            q_data_type=dtype,
            kv_data_type=dtype,
            o_data_type=dtype,
        )
        self._plan_indptr_snapshot = self.params.decode_page_indptr_h.clone()

    def replan_if_needed(self, head_num: int, head_dim: int, dtype: torch.dtype) -> None:
        current = self.params.decode_page_indptr_h
        if self._plan_indptr_snapshot is None or not torch.equal(self._plan_indptr_snapshot, current):
            self.plan(head_num, head_dim, dtype)


class WhisperCrossKVCacheGroup:
    """Engine ``cross`` cache group access: write-once at prefill, paged decode.

    The FlashInfer metadata (positions, page table, indptr, last-page lens) is
    derived through :class:`rtp_llm_ops.FlashInferMlaAttnParams` exactly like
    the standard attention path, but with cross semantics: prefix is always
    zero and the KV length is the audio frame count, never the token stream.
    """

    def __init__(
        self,
        head_num: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
    ) -> None:
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op import (
            KVCacheWriteOp,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
            get_py_flashinfer_workspace_buffer,
        )
        from rtp_llm.ops.compute_ops import rtp_llm_ops
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

        self.head_num = head_num
        self.head_dim = head_dim
        self.page_size = page_size
        self.dtype = dtype
        self.params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.write_op = KVCacheWriteOp(head_num, head_dim, page_size)
        self.write_op.set_params(self.params)
        self.workspace = get_py_flashinfer_workspace_buffer()
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(self.workspace, "HND")
        self.active = True
        # CUDA graph decode states, one per captured batch size: each graph
        # owns its params/wrapper pair so the fixed buffers never alias across
        # batch buckets (the eager path keeps using self.params/decode_wrapper).
        self._graph_states: dict = {}
        # Per-step dispatch flag: True only while a captured graph step owns
        # the cross attention (capture warmup/capture forwards); eager steps
        # always re-plan self.decode_wrapper and must not pick a stale
        # graph-state wrapper even when the batch size matches a bucket.
        self._graph_decode_step = False

    @staticmethod
    def _host_i32(tensor: torch.Tensor) -> torch.Tensor:
        if tensor is None:
            return torch.empty(0, dtype=torch.int32)
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.to(torch.int32).contiguous()

    def _block_table_host(self, cross_inputs) -> torch.Tensor:
        table = cross_inputs.kv_cache_kernel_block_id
        if table is None or table.numel() == 0:
            table = cross_inputs.kv_cache_kernel_block_id_device
        return self._host_i32(table)

    def write_prefill(
        self,
        cross_inputs,
        kv_lens: torch.Tensor,
        layer_kvs: List[Tuple[torch.Tensor, torch.Tensor]],
        layer_caches: List[LayerKVCache],
    ) -> None:
        """Write each layer's fused [total_frames, hidden] K/V into the group."""
        kv_lens = self._host_i32(kv_lens)
        zero_prefix = torch.zeros_like(kv_lens)
        block_table = self._block_table_host(cross_inputs)
        # Prefill semantics with zero prefix: positions 0..frames-1 per row,
        # page table from the cross group's block table.
        self.params.fill_params(zero_prefix, kv_lens, kv_lens, block_table, self.page_size, False)
        for (k, v), cache in zip(layer_kvs, layer_caches):
            if cache is None:
                continue
            k = k.view(-1, self.head_num, self.head_dim)
            v = v.view(-1, self.head_num, self.head_dim)
            self.write_op.forward(k, v, cache)

    def begin_decode(self, cross_inputs, kv_lens: torch.Tensor, is_cuda_graph: bool = False) -> None:
        """Plan the paged decode wrapper for this step's batch."""
        kv_lens = self._host_i32(kv_lens)
        # Decode branch: kv_len = sequence_lengths + 1, so pass frames - 1 to
        # make the cross KV length exactly the audio frame count.
        seq_minus_one = (kv_lens - 1).clamp(min=0)
        block_table = self._block_table_host(cross_inputs)
        empty_prefix = torch.empty(0, dtype=torch.int32)
        self.params.fill_params(empty_prefix, seq_minus_one, seq_minus_one, block_table, self.page_size, False)
        if is_cuda_graph:
            batch_size = int(kv_lens.numel())
            state = self._graph_states.get(batch_size)
            if state is None:
                from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
                from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
                    get_py_flashinfer_workspace_buffer,
                )
                from rtp_llm.ops.compute_ops import rtp_llm_ops as _ops

                workspace = get_py_flashinfer_workspace_buffer()
                state = _GraphDecodeState(
                    params=_ops.FlashInferMlaAttnParams(),
                    wrapper=BatchDecodeWithPagedKVCacheWrapper(workspace, "HND"),
                    workspace=workspace,
                    page_size=self.page_size,
                )
                self._graph_states[batch_size] = state
            state.bind(kv_lens, block_table)
            state.plan(self.head_num, self.head_dim, self.dtype)
            return
        self.decode_wrapper.plan(
            self.params.decode_page_indptr_d,
            self.params.page_indice_d,
            self.params.paged_kv_last_page_len_d,
            self.head_num,
            self.head_num,
            self.head_dim,
            self.page_size,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
            o_data_type=self.dtype,
        )

    def decode(self, q: torch.Tensor, layer_idx: int, layer_cache: Optional[LayerKVCache]) -> Optional[torch.Tensor]:
        """q [batch, hidden] -> attention output [batch, hidden], or None when
        this batch has no cross KV (the sublayer is skipped with zeros)."""
        if layer_cache is None:
            # No cross KV resolved for this batch (no audio / warmup rows):
            # the reference semantics skip the whole cross sublayer — the
            # caller must NOT run o_proj (its bias would leak in).
            return None
        q = q.view(-1, self.head_num, self.head_dim)
        state = self._graph_states.get(int(q.shape[0])) if self._graph_decode_step else None
        wrapper = state.wrapper if state is not None else self.decode_wrapper
        out = wrapper.run(q, layer_cache.kv_cache_base)
        result = out.reshape(-1, self.head_num * self.head_dim)
        return result

    def prepare_cuda_graph(self, attn_inputs) -> None:
        """Replay-time in-place refresh (called by CudaGraphRunner per step).

        The runner has already D2D-refreshed this tag's captured block table;
        the cross KV length is constant (audio frame capacity), so the device
        fill mainly re-stages indptr/last-page-length from the refreshed
        table. ``kv_len = sequence_lengths`` (frames), not seq+1.
        """
        kv_lens = attn_inputs.sequence_lengths
        if kv_lens is None or kv_lens.numel() == 0:
            return
        state = self._graph_states.get(int(kv_lens.numel()))
        if state is None:
            return
        block_id = attn_inputs.kv_cache_kernel_block_id_device
        if block_id is None or not block_id.is_cuda:
            block_id = attn_inputs.kv_cache_kernel_block_id.cuda()
        # Keep the device-fill metadata consistent with the captured plan
        # (kv = planned capacity): flashinfer's work partition is derived at
        # plan time, so a shorter indptr would desynchronize it. No-audio rows
        # carry an all-zero block-table row (reserved block 0) and attention
        # over zero pages yields zeros, matching the eager skip semantics.
        if (
            state.frames_dev is None
            or state.frames_dev.device != block_id.device
            or state.frames_dev.numel() != int(kv_lens.numel())
        ):
            state.frames_dev = torch.full(
                (int(kv_lens.numel()),), state.plan_kv_len, dtype=torch.int32, device=block_id.device
            )
        else:
            # Constant value: a device-side fill on the current (graph) stream
            # — never a host-to-device copy, whose non_blocking form rides the
            # allocator copy stream and races the replay under stream-async.
            state.frames_dev.fill_(state.plan_kv_len)
        state.params.fill_decode_cuda_graph_params(state.frames_dev, block_id, self.page_size)
        state.replan_if_needed(self.head_num, self.head_dim, self.dtype)

    def __del__(self):
        try:
            from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
                release_py_flashinfer_workspace_buffer,
            )

            release_py_flashinfer_workspace_buffer(self.workspace)
            for state in self._graph_states.values():
                release_py_flashinfer_workspace_buffer(state.workspace)
        except Exception:
            pass


class WhisperCrossAttention(nn.Module):
    """Cross attention: q <- decoder hidden, k/v <- encoder output (no mask)."""

    def __init__(
        self,
        attn_config,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        ragged_runner: Optional[WhisperCrossAttnRaggedRunner] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.parallelism_config = parallelism_config
        self.tp_size = parallelism_config.get_attn_tp_size()
        self.head_num = attn_config.head_num
        self.head_dim = attn_config.size_per_head
        self.scaling = self.head_dim**-0.5
        # Shared FlashInfer ragged runner (one per model, planned once per
        # forward by WhisperModel; None keeps the eager reference math).
        self.ragged_runner = ragged_runner
        self.q_proj = LinearFactory.create_linear_from_weights(
            weights, W.cross_attn_q_w, None, W.cross_attn_q_b,
            quant_config=quant_config, hw_kernel_config=hw_kernel_config,
        )
        self.k_proj = LinearFactory.create_linear_from_weights(
            weights, W.cross_attn_k_w, None, None,
            quant_config=quant_config, hw_kernel_config=hw_kernel_config,
        )
        self.v_proj = LinearFactory.create_linear_from_weights(
            weights, W.cross_attn_v_w, None, None,
            quant_config=quant_config, hw_kernel_config=hw_kernel_config,
        )
        self.o_proj = LinearFactory.create_linear_from_weights(
            weights, W.cross_attn_o_w, None, W.cross_attn_o_b,
            quant_config=quant_config, hw_kernel_config=hw_kernel_config,
        )

    def compute_kv(self, encoder_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """encoder_hidden [L, D] -> k/v [L, num_heads * head_dim] (fp16/eager)."""
        return self.k_proj(encoder_hidden), self.v_proj(encoder_hidden)

    def paged_forward(
        self,
        hidden_states: torch.Tensor,
        cross_group: Optional["WhisperCrossKVCacheGroup"],
        cross_layer_cache: Optional[LayerKVCache],
    ) -> torch.Tensor:
        """Decode path: q from hidden, K/V read paged from the ``cross`` group.

        The group itself owns the no-cache semantics (the engine group reads
        its paged blocks, an injected test double reads captured K/V), so a
        present group is always consulted.
        """
        if cross_group is None:
            # No engine cross group (no audio / warmup): pass through zeros,
            # matching the reference cross-attn skip for text-only rows.
            return torch.zeros_like(hidden_states)
        q = self.q_proj(hidden_states)
        attn = cross_group.decode(q, self.layer_idx, cross_layer_cache)
        if attn is None:
            # No cross KV for this batch: skip the sublayer entirely (zeros,
            # no o_proj — its bias would otherwise leak into every row).
            return torch.zeros_like(hidden_states)
        return self.o_proj(attn)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rows: List[Tuple[slice, Optional[Tuple[torch.Tensor, torch.Tensor]]]],
    ) -> torch.Tensor:
        """hidden_states [T, D] concatenated batch rows; rows carry per-row (k, v)."""
        runner = self.ragged_runner
        if runner is not None and runner.active:
            # Phase C fused path: q goes in unscaled (the head_dim**-0.5
            # softmax scale is the kernel's sm_scale), the shared plan covers
            # this layer's rows, and text-only rows come back as zeros in
            # assemble_output. Structure mismatch (None) falls through to
            # the eager reference math below.
            attn = runner.attention(self.layer_idx, rows, self.q_proj(hidden_states))
            if attn is not None:
                output = self.o_proj(attn)
                return runner.assemble_output(output, hidden_states)
        outputs = []
        for row_slice, kv in rows:
            h = hidden_states[row_slice]
            if kv is None:
                # text-only request inside a mixed batch: cross attention is
                # skipped (no audio), pass the sublayer through as zeros.
                outputs.append(torch.zeros_like(h))
                continue
            k, v = kv
            tgt_len = h.shape[0]
            q = self.q_proj(h) * self.scaling
            q = q.view(1, tgt_len, self.head_num, self.head_dim).transpose(1, 2)
            k = k.view(1, -1, self.head_num, self.head_dim).transpose(1, 2)
            v = v.view(1, -1, self.head_num, self.head_dim).transpose(1, 2)
            attn_weights = torch.matmul(q, k.transpose(2, 3))
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).reshape(tgt_len, -1)
            outputs.append(self.o_proj(attn_output))
        # ATen has no single-input fast path: torch.cat over a one-element
        # list still allocates and D2D-copies (32 memcpys per decode step at
        # bs1, one per layer). Returning the tensor itself is value-identical
        # (o_proj output / zeros_like are fresh, unaliased tensors and every
        # downstream consumer — reduce_output, _clamped_residual_add,
        # LayerNorm — is functional, never in-place).
        if len(outputs) == 1:
            return outputs[0]
        return torch.cat(outputs, dim=0)

    def reduce_output(self, output: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

            output = all_reduce(output, group=Group.TP)
        return output


class WhisperDecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        layer_idx: int,
        weights: Dict[str, torch.Tensor],
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        ragged_runner: Optional[WhisperCrossAttnRaggedRunner] = None,
    ):
        super().__init__()
        attn_configs = config.getAttentionConfigs(parallelism_config.get_attn_tp_size())
        self.self_attn = CausalAttention(
            attn_configs, parallelism_config, weights, config.layernorm_eps,
            quant_config, hw_kernel_config, layer_idx,
        )
        self.encoder_attn = WhisperCrossAttention(
            attn_configs, parallelism_config, weights, layer_idx,
            quant_config, hw_kernel_config, ragged_runner,
        )
        self.mlp = DenseMLP(
            config.activation_type, parallelism_config, weights, quant_config, hw_kernel_config,
        )
        self.self_attn_layer_norm = LayerNorm(
            weight=weights[W.pre_ln_gamma], beta=weights[W.pre_ln_beta], eps=config.layernorm_eps
        )
        self.encoder_attn_layer_norm = LayerNorm(
            weight=weights[W.cross_attn_pre_ln_gamma],
            beta=weights[W.cross_attn_pre_ln_beta],
            eps=config.layernorm_eps,
        )
        self.final_layer_norm = LayerNorm(
            weight=weights[W.post_ffn_ln_gamma], beta=weights[W.post_ffn_ln_beta], eps=config.layernorm_eps
        )

    def compute_cross_kv(self, encoder_hidden: torch.Tensor):
        return self.encoder_attn.compute_kv(encoder_hidden)

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        cross_rows: Optional[List[Tuple[slice, Optional[Tuple[torch.Tensor, torch.Tensor]]]]],
        cross_group: Optional["WhisperCrossKVCacheGroup"] = None,
        cross_layer_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states, fmha_impl=fmha_impl, kv_cache=kv_cache
        )
        hidden_states = _clamped_residual_add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        if cross_rows is not None:
            hidden_states = self.encoder_attn(hidden_states, cross_rows)
        else:
            hidden_states = self.encoder_attn.paged_forward(
                hidden_states, cross_group, cross_layer_cache
            )
        hidden_states = self.encoder_attn.reduce_output(hidden_states)
        hidden_states = _clamped_residual_add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = _clamped_residual_add(residual, hidden_states)
        return hidden_states


class WhisperModel(GptModelBase):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        quant_config: Optional[object] = None,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.embed_tokens = Embedding(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        # learned decoder positional embedding [448, 1280]
        pos_weight = weights.get_global_weight(W.positional_embedding)
        self.positional_embedding = nn.Embedding.from_pretrained(
            pos_weight.clone().to(pos_weight.dtype), freeze=True
        )
        # One ragged-prefill runner shared by all decoder layers (planned
        # once per forward in WhisperModel.forward; the FlashInfer workspace
        # is allocated lazily on the first CUDA forward).
        attn_configs = config.getAttentionConfigs(parallelism_config.get_attn_tp_size())
        self.cross_attn_runner: WhisperCrossAttnRaggedRunner = WhisperCrossAttnRaggedRunner(
            attn_configs.head_num, attn_configs.size_per_head
        )
        self.layers = nn.ModuleList(
            [
                WhisperDecoderLayer(
                    config, parallelism_config, idx, weights.weights[idx],
                    quant_config, py_hw_kernel_config, self.cross_attn_runner,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = LayerNorm(
            weight=weights.get_global_weight(W.final_ln_gamma),
            beta=weights.get_global_weight(W.final_ln_beta),
            eps=config.layernorm_eps,
        )
        # Engine ``cross`` cache group accessor (write at prefill, paged read
        # at decode); created lazily on the first forward that sees the group.
        self.cross_group: Optional[WhisperCrossKVCacheGroup] = None
        self._cross_group_init_failed = False

    def _get_cross_group(self) -> Optional[WhisperCrossKVCacheGroup]:
        if self.cross_group is not None or self._cross_group_init_failed:
            return self.cross_group
        try:
            if self.kv_cache is None or CROSS_KV_TAG not in self.kv_cache.group_tags:
                self._cross_group_init_failed = True
                return None
            page_size = self.kv_cache.get_kernel_seq_size_per_block(CROSS_KV_TAG)
            attn_configs = self.config.getAttentionConfigs(
                self.parallelism_config.get_attn_tp_size()
            )
            self.cross_group = WhisperCrossKVCacheGroup(
                attn_configs.head_num, attn_configs.size_per_head, page_size,
                torch.float16,
            )
            # The pool backing is torch::empty (uninitialized): CUDA-graph
            # dummy/no-audio rows point at reserved block 0 through their
            # zero block-table rows, so that block must read as zeros to
            # reproduce the eager cross-skip semantics.
            for i in range(model_config.num_layers):
                self.kv_cache.get_layer_cache(i, CROSS_KV_TAG).kv_cache_base[0].zero_()
        except Exception as error:  # FlashInfer unavailable -> eager fallback
            self._cross_group_init_failed = True
            return None
        return self.cross_group

    @staticmethod
    def _default_attention_inputs(attention_inputs):
        """Token-stream attention inputs: the ``default`` group's entry when
        the engine routes inputs by cache-group tag."""
        if isinstance(attention_inputs, Mapping):
            return select_attention_inputs_for_tag(attention_inputs, "default")
        return attention_inputs

    def cuda_graph_position_id_len_factor(self) -> int:
        # The decoder adds a learned positional embedding indexed by
        # combo_position_ids (no RoPE): the decode CUDA graph must capture
        # that tensor and refresh it per replay, exactly like Mrope models.
        return 1

    def prepare_fmha_impl(
        self,
        inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode: Optional[str] = None,
    ):
        # The ``cross`` group rides along as a dict member so CudaGraphRunner's
        # callPrepareCudaGraph reaches its prepare_cuda_graph hook; layers
        # never select it as an FMHA impl (self-attention uses "default").
        result = super().prepare_fmha_impl(inputs, is_cuda_graph, cuda_graph_selection_mode)
        cross_group = self._get_cross_group()
        if cross_group is None:
            return result
        if isinstance(result, Mapping):
            result = dict(result)
            result[CROSS_KV_TAG] = cross_group
            if is_cuda_graph:
                # Capture-time: plan with the capture tag inputs (fake frame
                # capacity) and flag them so the captured forward skips the
                # per-step host plan.
                attention_inputs = get_attention_inputs_value(inputs)
                if isinstance(attention_inputs, Mapping):
                    cross_inputs = select_attention_inputs_for_tag(attention_inputs, CROSS_KV_TAG)
                    cross_inputs.is_cuda_graph = True
                    kv_lens = cross_inputs.sequence_lengths
                    if kv_lens is not None and kv_lens.numel() > 0 and int(kv_lens.max()) > 0:
                        cross_group.begin_decode(cross_inputs, kv_lens, is_cuda_graph=True)
        return result

    def _get_fmha_group_tags(self) -> Optional[List[str]]:
        # The ``cross`` group is served by the model-private paged decoder
        # (WhisperCrossKVCacheGroup); only the token-stream groups go through
        # the standard FMHA factory.
        if self.kv_cache is None:
            return None
        return [tag for tag in self.kv_cache.group_tags if tag != CROSS_KV_TAG]

    def _row_slices_prefill(self, attention_inputs) -> List[slice]:
        cu = attention_inputs.cu_seqlens
        if cu is not None and cu.numel() > 1:
            bounds = cu.tolist()
        else:
            bounds = [0, attention_inputs.total_tokens]
        return [slice(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    def _prepare_cross_rows_prefill(
        self, inputs: PyModelInputs, default_inputs, attention_inputs_by_group
    ) -> List[List[Tuple[slice, Optional[Tuple[torch.Tensor, torch.Tensor]]]]]:
        """Prefill per-layer row descriptors: [(row_slice, (k_i, v_i) or None)].

        Also writes each layer's fused cross K/V into the engine ``cross``
        cache group so decode steps can read them paged. Prefill-only: the
        decode path reads the group through paged decode instead.
        """
        assert default_inputs.is_prefill, "cross row preparation is prefill-only"
        row_slices = self._row_slices_prefill(default_inputs)
        mm = inputs.multimodal_inputs
        features = list(mm.multimodal_features)
        if features:
            locs = mm.mm_features_locs.tolist()
        else:
            locs = []
        # map each feature to its batch row via its merged-batch position
        row_of_feature = []
        bounds = [s.start for s in row_slices] + [row_slices[-1].stop]
        for loc in locs:
            row_idx = 0
            for b in range(len(bounds) - 1):
                if bounds[b] <= loc < bounds[b + 1]:
                    row_idx = b
                    break
            row_of_feature.append(row_idx)
        row_encoder_hidden: List[Optional[torch.Tensor]] = [None] * len(row_slices)
        for feat_idx, (feat, row_idx) in enumerate(zip(features, row_of_feature)):
            row_encoder_hidden[row_idx] = feat
        # compute per-layer cross K/V per row; the prefill compute reads the
        # freshly computed tensors directly (zero-copy views), the engine
        # ``cross`` group gets the same values written once for decode.
        row_layer_kvs: List[Optional[List[Tuple[torch.Tensor, torch.Tensor]]]] = [None] * len(row_slices)
        for row_idx, enc in enumerate(row_encoder_hidden):
            if enc is None:
                continue
            row_layer_kvs[row_idx] = [layer.compute_cross_kv(enc) for layer in self.layers]
        # Fuse each layer's per-row K/V into one contiguous ragged buffer and
        # keep per-row views into it. The ragged cross-attention runner reads
        # the shared storage zero-copy, and the engine-group write consumes
        # the same ragged buffers (identical values, same lifetime).
        kv_row_indices = [r for r, kvs in enumerate(row_layer_kvs) if kvs is not None]
        layer_ragged_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(self.layer_num):
            pairs = [row_layer_kvs[r][layer_idx] for r in kv_row_indices]
            if not pairs:
                layer_ragged_kvs.append((None, None))
                continue
            views, k_ragged, v_ragged = fuse_row_kv(pairs)
            for r, view in zip(kv_row_indices, views):
                row_layer_kvs[r][layer_idx] = view
            layer_ragged_kvs.append((k_ragged, v_ragged))
        # Write once into the engine ``cross`` cache group (paged blocks owned
        # by the request; decode reads them back without any model-side copy).
        cross_group = self._get_cross_group()
        if cross_group is not None and kv_row_indices:
            cross_inputs = select_attention_inputs_for_tag(attention_inputs_by_group, CROSS_KV_TAG)
            kv_lens = cross_inputs.sequence_lengths
            if kv_lens is None or kv_lens.numel() != len(row_slices):
                kv_lens = torch.tensor(
                    [enc.shape[0] if enc is not None else 0 for enc in row_encoder_hidden],
                    dtype=torch.int32,
                )
            if self.kv_cache is not None and CROSS_KV_TAG in self.kv_cache.group_tags:
                layer_caches = [
                    self.kv_cache.get_layer_cache(i, CROSS_KV_TAG) for i in range(self.layer_num)
                ]
            else:
                # Test double installed without the engine cache: the group
                # skips layers without a real cache object.
                layer_caches = [None] * self.layer_num
            cross_group.write_prefill(cross_inputs, kv_lens, layer_ragged_kvs, layer_caches)
        per_layer_rows = []
        for layer_idx in range(self.layer_num):
            rows = []
            for row_idx in range(len(row_slices)):
                layer_kvs = row_layer_kvs[row_idx]
                if layer_kvs is None:
                    rows.append((row_slices[row_idx], None))
                else:
                    rows.append((row_slices[row_idx], layer_kvs[layer_idx]))
            per_layer_rows.append(rows)
        return per_layer_rows

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        positions = inputs.combo_position_ids
        hidden_states = self.embed_tokens(input_ids)
        if positions is not None and positions.numel() == input_ids.numel():
            positions = positions.to(device=hidden_states.device, dtype=torch.long)
            # Same Triton add as the residual path: the fp16 guard is a no-op
            # for finite sums (embedding magnitudes are ~1e-2), so this is
            # bit-identical to the reference ``embed + pos_embed`` fp16 add.
            hidden_states = _clamped_residual_add(
                hidden_states, self.positional_embedding(positions)
            )
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        attention_inputs = get_attention_inputs_value(inputs)
        default_inputs = self._default_attention_inputs(attention_inputs)
        cross_group = self._get_cross_group()
        if default_inputs.is_prefill:
            per_layer_rows = self._prepare_cross_rows_prefill(inputs, default_inputs, attention_inputs)
            cross_layer_caches = None
            # Plan the shared cross-attention ragged runner once from the row
            # structure (identical for every layer); inactive (non-CUDA / no
            # K/V rows / FlashInfer unavailable) leaves the eager path in place.
            if per_layer_rows:
                self.cross_attn_runner.begin_forward(per_layer_rows[0], hidden_states)
        else:
            per_layer_rows = None
            cross_layer_caches = None
            # Graph mode needs both the captured batch state and this step's
            # graph flag (set by prepare_fmha_impl): the eager fallback after
            # a canRun rejection must take the eager branch even though the
            # graph states exist.
            cross_inputs_for_gate = select_attention_inputs_for_tag(attention_inputs, CROSS_KV_TAG)
            in_graph_mode = (
                cross_group is not None
                and int(default_inputs.sequence_lengths.numel()) in cross_group._graph_states
                and bool(cross_inputs_for_gate.is_cuda_graph)
            )
            if cross_group is not None and not in_graph_mode:
                cross_group._graph_decode_step = False
                cross_inputs = select_attention_inputs_for_tag(attention_inputs, CROSS_KV_TAG)
                kv_lens = cross_inputs.sequence_lengths
                if (
                    kv_lens is not None
                    and kv_lens.numel() > 0
                    and int(kv_lens.max()) > 0
                    and self.kv_cache is not None
                    and CROSS_KV_TAG in self.kv_cache.group_tags
                ):
                    cross_group.begin_decode(cross_inputs, kv_lens)
                    cross_layer_caches = [
                        self.kv_cache.get_layer_cache(i, CROSS_KV_TAG)
                        for i in range(self.layer_num)
                    ]
            elif in_graph_mode and self.kv_cache is not None and CROSS_KV_TAG in self.kv_cache.group_tags:
                # Capture: the paged kernels must be recorded with the graph-
                # owned metadata; prepare_cuda_graph refreshes it per replay.
                cross_group._graph_decode_step = True
                cross_layer_caches = [
                    self.kv_cache.get_layer_cache(i, CROSS_KV_TAG)
                    for i in range(self.layer_num)
                ]
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            # Self-attention consumes the token-stream group's impl; the
            # ``cross`` group is served by the model-private paged decoder.
            if isinstance(fmha_impl, Mapping):
                layer_fmha_impl = select_fmha_impl_for_group(fmha_impl, "default")
            else:
                layer_fmha_impl = fmha_impl
            hidden_states = decoder_layer(
                hidden_states,
                layer_fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i, "default") if self.kv_cache else None,
                cross_rows=per_layer_rows[i] if per_layer_rows is not None else None,
                cross_group=cross_group if per_layer_rows is None else None,
                cross_layer_cache=(
                    cross_layer_caches[i] if cross_layer_caches is not None else None
                ),
            )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states)


__all__ = ["WhisperModel"]
