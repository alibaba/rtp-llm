from typing import Any, Dict, List, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FMHAImplBase,
    RMSNorm,
)
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

import aiter
from aiter import rmsnorm_quant as _rmsnorm_quant
from aiter import add_rmsnorm_quant as _add_rmsnorm_quant
from aiter import dynamic_per_token_scaled_quant as _quant
from aiter import silu_and_mul as _silu_and_mul
from aiter import rms_norm as _rms_norm
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle_cktile as _gemm_cktile


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        layer_idx: int,
        weights: Dict[str, torch.Tensor],
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        attn_configs = config.getAttentionConfigs(parallelism_config.get_attn_tp_size())
        self.self_attn = CausalAttention(
            attn_configs,
            parallelism_config,
            weights,
            config.layernorm_eps,
            quant_config,
            hw_kernel_config,
            layer_idx,
        )
        self.mlp = DenseMLP(
            config.activation_type,
            parallelism_config,
            weights,
            quant_config,
            hw_kernel_config,
        )
        self.input_layernorm = RMSNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states, fmha_impl=fmha_impl, kv_cache=kv_cache
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3Model(GptModelBase):
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
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    config,
                    parallelism_config,
                    idx,
                    weights.weights[idx],
                    quant_config,
                    py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )

    def _init_fused(self):
        """Extract weights for fused forward path."""
        self._fused_ok = False
        l0 = self.layers[0].self_attn.qkv_proj
        if not (hasattr(l0, 'weight') and hasattr(l0, 'weight_scales')
                and l0.weight.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn)):
            return
        if self.kv_cache is not None:
            return

        self._ln1_w = [l.input_layernorm.weight.data for l in self.layers]
        self._ln2_w = [l.post_attention_layernorm.weight.data for l in self.layers]
        self._qkv_w = [l.self_attn.qkv_proj.weight for l in self.layers]
        self._qkv_s = [l.self_attn.qkv_proj.weight_scales for l in self.layers]
        self._o_w = [l.self_attn.o_proj.weight for l in self.layers]
        self._o_s = [l.self_attn.o_proj.weight_scales for l in self.layers]
        self._up_w = [l.mlp.up_proj.weight for l in self.layers]
        self._up_s = [l.mlp.up_proj.weight_scales for l in self.layers]
        self._down_w = [l.mlp.down_proj.weight for l in self.layers]
        self._down_s = [l.mlp.down_proj.weight_scales for l in self.layers]
        self._final_w = self.norm.weight.data
        self._eps = self.layers[0].input_layernorm.variance_epsilon
        self._nheads = self.layers[0].self_attn.head_num
        self._nkv = self.layers[0].self_attn.qkv_proj.weight.shape[0] // self.layers[0].self_attn.head_dim - self._nheads
        self._nkv = self._nkv // 2
        self._hdim = self.layers[0].self_attn.head_dim
        self._q_size = self._nheads * self._hdim
        self._kv_size = self._nkv * self._hdim
        self._K = self._qkv_w[0].shape[1]
        self._N_up = self._up_w[0].shape[0]
        self._K_down = self._down_w[0].shape[1]
        self._buf_M = 0
        self._fused_ok = True

    def _ensure_bufs(self, M, device):
        if M <= self._buf_M:
            return
        new_M = max(M, self._buf_M * 2, 256)
        fp8 = self._qkv_w[0].dtype
        K = self._K
        N_qkv = self._qkv_w[0].shape[0]
        K_down = self._K_down
        self._xq = torch.empty(new_M, K, dtype=fp8, device=device)
        self._xs = torch.empty(new_M, dtype=torch.float32, device=device)
        self._xq2 = torch.empty(new_M, K, dtype=fp8, device=device)
        self._xs2 = torch.empty(new_M, dtype=torch.float32, device=device)
        self._xq_d = torch.empty(new_M, K_down, dtype=fp8, device=device)
        self._xs_d = torch.empty(new_M, dtype=torch.float32, device=device)
        self._qkv_buf = torch.empty(new_M, N_qkv, dtype=torch.bfloat16, device=device)
        self._o_buf = torch.empty(new_M, K, dtype=torch.bfloat16, device=device)
        self._up_buf = torch.empty(new_M, self._N_up, dtype=torch.bfloat16, device=device)
        self._silu_buf = torch.empty(new_M, self._N_up // 2, dtype=torch.bfloat16, device=device)
        self._down_buf = torch.empty(new_M, K, dtype=torch.bfloat16, device=device)
        self._res_buf = torch.empty(new_M, K, dtype=torch.bfloat16, device=device)
        self._buf_M = new_M

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        if not hasattr(self, '_fused_ok'):
            self._init_fused()

        _block_map = inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group
        _has_block_map = _block_map is not None and len(_block_map) > 0
        if not self._fused_ok or _has_block_map:
            for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
                select_block_map_for_layer(inputs.attention_inputs, i)
                hidden_states = decoder_layer(
                    hidden_states,
                    fmha_impl,
                    kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                )
            hidden_states = self.norm(hidden_states)
            return PyModelOutputs(hidden_states, fmha_impl.fmha_params)

        # Fused forward: fewer kernel launches, minimal Python between ops
        M = hidden_states.shape[0]
        self._ensure_bufs(M, hidden_states.device)
        eps = self._eps
        xq = self._xq[:M]
        xs = self._xs[:M]
        xq2 = self._xq2[:M]
        xs2 = self._xs2[:M]
        xq_d = self._xq_d[:M]
        xs_d = self._xs_d[:M]
        qkv_buf = self._qkv_buf[:M]
        o_buf = self._o_buf[:M]
        up_buf = self._up_buf[:M]
        silu_buf = self._silu_buf[:M]
        down_buf = self._down_buf[:M]
        res_buf = self._res_buf[:M]
        q_size = self._q_size
        kv_size = self._kv_size
        nh = self._nheads
        nk = self._nkv
        hd = self._hdim
        ss = [q_size, kv_size, kv_size]

        for i in range(self.layer_num):
            # fused: rmsnorm + quant → 1 kernel (replaces norm + quant = 2)
            _rmsnorm_quant(xq, hidden_states, xs, self._ln1_w[i], eps)
            # gemm qkv
            _gemm_cktile(xq, self._qkv_w[i], xs, self._qkv_s[i], qkv_buf)
            # attention (rope + flash_attn via fmha_impl)
            attn_out = fmha_impl.forward(qkv_buf, None, i)
            # quant attn output for o_proj
            _quant(xq2, attn_out.reshape(M, q_size), xs2)
            # gemm o_proj
            _gemm_cktile(xq2, self._o_w[i], xs2, self._o_s[i], o_buf)
            # fused: residual_add + rmsnorm + quant → 1 kernel (replaces add + norm + quant = 3)
            _add_rmsnorm_quant(xq, o_buf, hidden_states, res_buf, xs, self._ln2_w[i], eps)
            # gemm gate_up
            _gemm_cktile(xq, self._up_w[i], xs, self._up_s[i], up_buf)
            # silu_and_mul
            _silu_and_mul(silu_buf, up_buf)
            # quant for down_proj
            _quant(xq_d, silu_buf, xs_d)
            # gemm down
            _gemm_cktile(xq_d, self._down_w[i], xs_d, self._down_s[i], down_buf)
            # residual add
            hidden_states = down_buf
            hidden_states.add_(res_buf)

        hidden_states = _rms_norm(hidden_states, self._final_w, eps)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = [
    "Qwen3Model",
]
