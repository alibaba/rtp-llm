"""Windowed Prefill over fresh MLA KV and the replicated SWA history tail."""

from itertools import accumulate
from typing import Dict, List, Optional

import torch
from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashInferImplBase,
    MlaFlashInferPrefillImpl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_dense_prefill import (
    build_flashmla_device_params,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
    MlaKVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.rope_emb_new import (
    NewMlaRotaryEmbeddingOp,
)
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
from rtp_llm.models_py.modules.factory.linear.quantized_activation import retained_bf16
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    get_py_flashinfer_workspace_buffer,
    release_py_flashinfer_workspace_buffer,
)
from rtp_llm.utils.model_weight import W
from rtp_llm.ops import AttentionConfigs, FMHAConfig, KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import PyAttentionInputs


class SlidingWindowMlaPrefillOp:
    def __init__(self, config, weights, quant_config):
        if (
            config.is_sparse
            or config.kv_cache_dtype != KvCacheDataType.BASE
            or config.sliding_window <= 0
        ):
            raise ValueError(
                "SWA MLA Prefill requires dense queries, BASE KV and a positive window"
            )
        self.num_heads = config.head_num
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.rope_head_dim
        self.qk_nope_head_dim = config.nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.token_per_block = config.tokens_per_block
        self.window = config.sliding_window
        self.scale = (config.nope_head_dim + config.rope_head_dim) ** -0.5
        self.scale *= config.softmax_extra_scale
        self.weights = weights
        self.quant_config = quant_config

        self.workspace = get_py_flashinfer_workspace_buffer()
        self.attention = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace, "NHD", backend="fa2"
        )

    def __del__(self):
        workspace = getattr(self, "workspace", None)
        if workspace is not None:
            release_py_flashinfer_workspace_buffer(workspace)

    def plan(self, params):
        self.mla_params = params
        lengths = [
            q + min(prefix, self.window - 1)
            for q, prefix in zip(params.q_lens_host, params.prefix_lens_host)
        ]
        self.kv_indptr = torch.tensor(
            [0, *accumulate(lengths)],
            dtype=torch.int32,
            device=params.qo_indptr_d.device,
        )
        self.attention.plan(
            params.qo_indptr_d, self.kv_indptr,
            self.num_heads, self.num_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim, self.v_head_dim,
            causal=True, window_left=self.window - 1, sm_scale=self.scale,
            q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
        )

    def forward(self, q, compressed_kv, k_pe, kv_cache, layer_id):
        compressed_kv, k_pe = self._reuse_kv_cache_indexed_batched(
            retained_bf16(compressed_kv), k_pe, kv_cache
        )
        projection = LinearFactory.create_linear_from_weights(
            self.weights[layer_id], W.mla_kv_b_w, W.mla_kv_b_s,
            None, self.quant_config,
        )
        kv = projection(compressed_kv).view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        key = torch.cat((
            kv[..., :self.qk_nope_head_dim],
            k_pe.view(-1, 1, self.qk_rope_head_dim).expand(-1, self.num_heads, -1),
        ), dim=-1)
        return self.attention.run(q, key, kv[..., self.qk_nope_head_dim:])

    def _reuse_kv_cache_indexed_batched(self, compressed_kv, k_pe, kv_cache):
        params = self.mla_params
        if not any(params.prefix_lens_host):
            return compressed_kv, k_pe
        latent_parts, rope_parts = [], []
        fresh_start = 0
        flat_rope = k_pe.view(-1, self.qk_rope_head_dim)
        table = params.attn_inputs.kv_cache_kernel_block_id_device
        cache = kv_cache.kv_cache_base
        for request, (length, prefix) in enumerate(
            zip(params.q_lens_host, params.prefix_lens_host)
        ):
            if prefix:
                positions = torch.arange(
                    max(0, prefix - self.window + 1), prefix, device=table.device
                )
                physical = table[request, positions // self.token_per_block].long()
                slots = (
                    physical * self.token_per_block + positions % self.token_per_block
                )
                # The table names P pages; the tensor may expose their K-page view.
                history = cache[slots // cache.shape[1], slots % cache.shape[1]]
                latent_parts.append(history[:, : self.kv_lora_rank])
                rope_parts.append(history[:, self.kv_lora_rank :])
            fresh_end = fresh_start + length
            latent_parts.append(compressed_kv[fresh_start:fresh_end])
            rope_parts.append(flat_rope[fresh_start:fresh_end])
            fresh_start = fresh_end
        return torch.cat(latent_parts), torch.cat(rope_parts)


class SlidingWindowMlaPrefillImpl(MlaFlashInferPrefillImpl):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        weights: List[Dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor,
        fmha_config: Optional[FMHAConfig] = None,
        quant_config: Optional[object] = None,
        max_seq_len: int = 0,
        is_cuda_graph: bool = False,
        parallelism_config: Optional[ParallelismConfig] = None,
    ):
        if not self.support_parallelism_config(parallelism_config):
            raise ValueError(
                "SWA MLA Prefill requires a compute-CP consumer for zigzag inputs"
            )
        MlaFlashInferImplBase.__init__(
            self,
            SlidingWindowMlaPrefillOp(attn_configs, weights, quant_config),
            NewMlaRotaryEmbeddingOp(
                cos_sin_cache, attn_configs.rope_config.is_neox_style
            ),
            MlaKVCacheWriteOp(attn_configs.kv_cache_dtype),
            attn_inputs,
            attn_configs.tokens_per_block,
            attn_configs,
            weights,
            cos_sin_cache,
            fmha_config,
            quant_config=quant_config,
            max_seq_len=max_seq_len,
            is_cuda_graph=is_cuda_graph,
            parallelism_config=parallelism_config,
            warmup_flashinfer=False,
        )
        self.absorb_fmha = None

    def create_params(self, attn_inputs: PyAttentionInputs):
        self.prepare(attn_inputs)

    def prepare(self, attn_inputs: PyAttentionInputs, forbid_realloc: bool = False):
        self.attn_inputs = attn_inputs
        params = build_flashmla_device_params(attn_inputs, self.seq_size_per_block)
        self.fmha_params = self.rope_params = params
        self.fmha_impl.plan(params)

    def forward(self, q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=None):
        select_block_map_for_layer(self.attn_inputs, layer_id)
        return super().forward(q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices)

    def _device_slot_mapping(self):
        slots = super()._device_slot_mapping()
        # NULL SWA pages are intentionally absent; never write their reserved row.
        return torch.where(slots >= self.seq_size_per_block, slots, -1)
