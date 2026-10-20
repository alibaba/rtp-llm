"""Whisper decoder (A2/A3): 32-layer pre-LN decoder with per-layer cross attention.

Engine contract (GptModelBase): forward(inputs, fmha_impl) -> PyModelOutputs
returning post-final-norm hidden [total_tokens, hidden]; C++ applies lm_head.

Cross-attention state (Phase A):
- At prefill, `PyMultimodalInputs.multimodal_features` carries the audio
  encoder hidden [L, 1280] per request (mm_skip_token_expansion keeps the
  token stream untouched). The model computes per-layer cross K/V once and
  stores them in a slot pool keyed by the request's first KV-cache kernel
  block id (unique for the request's lifetime; prefill overwrites stale
  entries). Decode steps look the slot up by the same key.
- fp16 inf/nan clamp after every residual add, like modeling_whisper.py:635.

Fused kernels (Phase C, whisper_fused_kernels.py):
- Every residual add runs as one Triton kernel (add + fp16 nan_to_num guard,
  bit-exact vs the eager pair) instead of the ATen add + nan_to_num launches.
- Cross attention runs through a shared FlashInfer ragged-prefill runner
  (non-causal, per-row K/V, sm_scale in the kernel) instead of the eager
  matmul-softmax-matmul; each layer's per-row K/V are fused into one
  contiguous ragged buffer per layer at prefill so decode reads them without
  per-step copies. The eager reference math remains as the fallback for
  non-CUDA devices, non-fp16 dtypes and FlashInfer failures.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import (
    get_attention_inputs_value,
    select_fmha_impl_for_layer,
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
        cross_rows: List[Tuple[slice, Optional[Tuple[torch.Tensor, torch.Tensor]]]],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states, fmha_impl=fmha_impl, kv_cache=kv_cache
        )
        hidden_states = _clamped_residual_add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(hidden_states, cross_rows)
        hidden_states = self.encoder_attn.reduce_output(hidden_states)
        hidden_states = _clamped_residual_add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = _clamped_residual_add(residual, hidden_states)
        return hidden_states


class WhisperCrossKVPool:
    """Per-request cross-attention K/V slots keyed by first kernel block id.

    The key (attention_inputs.kv_cache_kernel_block_id[row, 0]) is unique for
    a request's lifetime: pages are not reassigned while allocated, and a
    prefill overwrites any stale entry left on a reused block. Limitation
    (Phase A): concurrent live requests sharing one prefix block (prefix
    cache on, identical tiny prompts) can alias; production correctness comes
    with the engine cross-KV cache group (Phase B).

    Each slot holds 32 layers x 2 x L x 1280 fp16 (~245 MB for a 30 s audio),
    so the default cap bounds model-side memory at ~15 GB; raise with
    WHISPER_CROSS_KV_POOL_CAP when the GPU budget allows.
    """

    def __init__(self, capacity: Optional[int] = None):
        if capacity is None:
            try:
                capacity = int(os.environ.get("WHISPER_CROSS_KV_POOL_CAP", "64"))
            except ValueError:
                capacity = 64
        self.capacity = capacity
        self._slots: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

    def put(self, key: int, layer_kvs: List[Tuple[torch.Tensor, torch.Tensor]]):
        if len(self._slots) >= self.capacity and key not in self._slots:
            self._slots.clear()  # simple bounded eviction; smokes are short-lived
            logger.info("whisper cross-KV pool cleared (%d slots)", self.capacity)
        self._slots[key] = layer_kvs

    def get(self, key: int) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        return self._slots.get(key)


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
        self.cross_kv_pool = WhisperCrossKVPool()

    def _first_block_ids(self, attention_inputs) -> List[int]:
        block_table = attention_inputs.kv_cache_kernel_block_id
        if block_table is None or block_table.numel() == 0:
            block_table = attention_inputs.kv_cache_block_id
        ids = block_table[:, 0].tolist()
        return ids

    def _row_slices_prefill(self, attention_inputs) -> List[slice]:
        cu = attention_inputs.cu_seqlens
        if cu is not None and cu.numel() > 1:
            bounds = cu.tolist()
        else:
            bounds = [0, attention_inputs.total_tokens]
        return [slice(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    def _row_slices_decode(self, attention_inputs, input_ids: torch.Tensor) -> List[slice]:
        # Phase A: plain decode — one token per batch row. The engine sets
        # total_tokens=0 for decode batches, so derive rows from
        # sequence_lengths and tokens from input_ids. (MTP verify batches are
        # not supported until the Phase B engine cross-KV cache group.)
        batch = attention_inputs.sequence_lengths.numel()
        total = input_ids.numel()
        if total != batch:
            raise RuntimeError(
                f"whisper Phase A decode expects one token per row, got tokens={total} batch={batch}"
            )
        return [slice(i, i + 1) for i in range(total)]

    def _prepare_cross_rows(
        self, inputs: PyModelInputs, attention_inputs
    ) -> List[List[Tuple[slice, Optional[Tuple[torch.Tensor, torch.Tensor]]]]]:
        """Per-layer row descriptors: [(row_slice, (k_i, v_i) or None)]."""
        is_prefill = attention_inputs.is_prefill
        if is_prefill:
            row_slices = self._row_slices_prefill(attention_inputs)
            first_blocks = self._first_block_ids(attention_inputs)
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
            # compute per-layer cross K/V per row; build this forward's rows
            # from the freshly computed tensors directly (the pool may evict
            # under capacity pressure mid-batch — never read back what we
            # just put).
            row_layer_kvs: List[Optional[List[Tuple[torch.Tensor, torch.Tensor]]]] = [None] * len(row_slices)
            for row_idx, enc in enumerate(row_encoder_hidden):
                if enc is None:
                    continue
                row_layer_kvs[row_idx] = [layer.compute_cross_kv(enc) for layer in self.layers]
            # Phase C: fuse each layer's per-row K/V into one contiguous
            # ragged buffer and keep per-row views into it. The ragged
            # cross-attention runner reads the shared storage zero-copy, and
            # so do later decode steps whose batch matches this prefill; the
            # pool stores the views (identical values, same lifetime).
            kv_row_indices = [r for r, kvs in enumerate(row_layer_kvs) if kvs is not None]
            for layer_idx in range(self.layer_num):
                pairs = [row_layer_kvs[r][layer_idx] for r in kv_row_indices]
                if not pairs:
                    continue
                views, _, _ = fuse_row_kv(pairs)
                for r, view in zip(kv_row_indices, views):
                    row_layer_kvs[r][layer_idx] = view
            for r in kv_row_indices:
                self.cross_kv_pool.put(first_blocks[r], row_layer_kvs[r])
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
        # decode: per-row pool lookup
        row_slices = self._row_slices_decode(attention_inputs, inputs.input_ids)
        first_blocks = self._first_block_ids(attention_inputs)
        per_layer_rows = []
        for layer_idx in range(self.layer_num):
            rows = []
            for row_idx, row_slice in enumerate(row_slices):
                slot = self.cross_kv_pool.get(first_blocks[row_idx])
                if slot is None:
                    logger.warning(
                        "whisper decode row %d has no cross-KV slot (block %s); "
                        "cross attention output will be zero",
                        row_idx, first_blocks[row_idx],
                    )
                    rows.append((row_slice, None))
                else:
                    rows.append((row_slice, slot[layer_idx]))
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
        per_layer_rows = self._prepare_cross_rows(inputs, attention_inputs)
        # Plan the shared cross-attention ragged runner once from the row
        # structure (identical for every layer); inactive (non-CUDA / no K/V
        # rows / FlashInfer unavailable) leaves the eager path in place.
        if per_layer_rows:
            self.cross_attn_runner.begin_forward(per_layer_rows[0], hidden_states)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            layer_fmha_impl = select_fmha_impl_for_layer(fmha_impl, self.kv_cache, i)
            hidden_states = decoder_layer(
                hidden_states,
                layer_fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                cross_rows=per_layer_rows[i],
            )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states)


__all__ = ["WhisperModel"]
