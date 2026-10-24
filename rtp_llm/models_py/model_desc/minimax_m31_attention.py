"""MiniMax-M3.1 model-owned attention projections and native Decode."""

from collections.abc import Mapping

import torch
from torch import nn

from rtp_llm.models.minimax_m31_weights import (
    IDX_K_NORM,
    IDX_K_SCALE,
    IDX_K_WEIGHT,
    IDX_Q_NORM,
    IDX_Q_SCALE,
    IDX_Q_WEIGHT,
    RAW_K_NORM,
    RAW_Q_NORM,
)
from rtp_llm.models_py.model_desc.minimax_m31_cache import m31_cache_views
from rtp_llm.models_py.model_desc.minimax_m31_decode import (
    m31_decode_projected_attention,
)
from rtp_llm.models_py.model_desc.minimax_m31_prefill import (
    m31_prefill_projected_attention,
)
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.models_py.modules.factory.linear.impl.cuda.mxfp8_linear import (
    CudaMxfp8Linear,
)
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs
from rtp_llm.utils.model_weight import W


def _decode_index_scale(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if (
        weight.ndim != 2
        or weight.dtype != torch.float8_e4m3fn
        or scale.shape != (weight.shape[0], weight.shape[1] // 32)
        or scale.dtype != torch.uint8
    ):
        raise ValueError("M3.1 index projection requires E4M3 weight and UE8M0 scale")
    return torch.exp2(scale.to(torch.float32) - 127.0).contiguous()


class M31Attention(nn.Module):
    def __init__(
        self,
        config,
        parallelism_config,
        weights,
        layer_idx,
        hw_kernel_config=None,
    ):
        super().__init__()
        if parallelism_config.get_attn_tp_size() != 1:
            raise ValueError(
                "M3.1 native sparse attention currently requires attention TP=1"
            )
        attn = config.getAttentionConfigs(1)
        if (attn.head_num, attn.kv_head_num, attn.size_per_head) != (64, 4, 128):
            raise ValueError("M3.1 native sparse attention requires 64Q/4KV/D128")
        if config.msa_sparse_config["num_idx_heads"] != 4:
            raise ValueError("M3.1 native sparse attention requires four index Q heads")
        self.layer_idx = layer_idx
        self.norm_eps = config.layernorm_eps
        self.init_blocks = config.msa_sparse_config["init_blocks"]
        self.local_blocks = config.msa_sparse_config["local_blocks"]

        qkv_w = weights[W.attn_qkv_w]
        qkv_s = weights[W.attn_qkv_s]
        index_q_w = weights[IDX_Q_WEIGHT]
        index_q_s = _decode_index_scale(index_q_w, weights[IDX_Q_SCALE])
        index_k_w = weights[IDX_K_WEIGHT]
        index_k_s = _decode_index_scale(index_k_w, weights[IDX_K_SCALE])
        if (
            qkv_w.shape != (72 * 128, config.hidden_size)
            or qkv_w.dtype != torch.float8_e4m3fn
            or qkv_s.shape != (72 * 128, config.hidden_size // 32)
            or qkv_s.dtype != torch.float32
            or index_q_w.shape != (4 * 128, config.hidden_size)
            or index_k_w.shape != (128, config.hidden_size)
        ):
            raise ValueError("M3.1 fused QKV/index MXFP8 projection geometry mismatch")
        self.qkv_index_proj = CudaMxfp8Linear(
            torch.cat((qkv_w, index_q_w, index_k_w), dim=0).contiguous(),
            torch.cat((qkv_s, index_q_s, index_k_s), dim=0).contiguous(),
            None,
            None,
            config.quant_config,
        )
        self.qkv_index_proj._packed_weight_scale()
        self.o_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.attn_o_w,
            W.attn_o_s,
            W.attn_o_b,
            quant_config=config.quant_config,
            hw_kernel_config=hw_kernel_config,
            weight_scale_2_key=W.attn_o_s2,
            input_scale_key=W.attn_o_i_s,
        )
        self.raw_norms = (
            weights[RAW_Q_NORM],
            weights[RAW_K_NORM],
            weights[IDX_Q_NORM],
            weights[IDX_K_NORM],
        )
        from rtp_llm.ops import get_rope_cache_once

        cache = get_rope_cache_once(
            attn.rope_config,
            attn.max_seq_len + attn.gen_num_per_cycle + 1,
            is_cuda=True,
            interleave=False,
        )
        self.cos_sin_cache = cache.data
        self.output_mxfp8 = isinstance(
            self.o_proj, CudaMxfp8Linear
        ) and torch.cuda.get_device_capability(qkv_w.device) == (10, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_inputs: Mapping[str, PyAttentionInputs],
        kv_cache: Mapping[str, LayerKVCache],
    ) -> torch.Tensor:
        from rtp_llm.models.minimax_m31_kv_cache import INDEXER_KV_TAG, MAIN_KV_TAG

        main_inputs = attention_inputs[MAIN_KV_TAG]
        index_inputs = attention_inputs[INDEXER_KV_TAG]
        main_cache = kv_cache[MAIN_KV_TAG]
        index_cache = kv_cache[INDEXER_KV_TAG]
        views = m31_cache_views(
            main_cache.kv_cache_base,
            index_cache.kv_cache_base,
            kv_heads=4,
            head_dim=128,
            index_dim=128,
            page_size=128,
        )
        fused = self.qkv_index_proj(hidden_states)
        qkv, index_q, index_k = fused.split((72 * 128, 4 * 128, 128), dim=-1)
        if main_inputs.is_prefill:
            attended = m31_prefill_projected_attention(
                qkv,
                index_q,
                index_k,
                self.raw_norms,
                self.cos_sin_cache,
                views,
                main_inputs,
                index_inputs,
                main_cache,
                index_cache,
                norm_eps=self.norm_eps,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
            )
        else:
            attended = m31_decode_projected_attention(
                qkv,
                index_q,
                index_k,
                self.raw_norms,
                self.cos_sin_cache,
                views,
                main_inputs,
                index_inputs,
                norm_eps=self.norm_eps,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                output_mxfp8=self.output_mxfp8 and 0 < hidden_states.shape[0] <= 100,
            )
        if isinstance(attended, tuple):
            q8, scales = attended
            return self.o_proj(
                q8.reshape(hidden_states.shape[0], -1), input_scales=scales
            )
        return self.o_proj(attended.reshape(hidden_states.shape[0], -1))
