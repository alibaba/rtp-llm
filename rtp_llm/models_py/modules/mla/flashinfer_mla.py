from typing import Any, Dict, List, Optional
import logging
import rtp_llm.models_py.modules.utils as utils

try:
    import flashinfer
    logging.info("mla flashinfer_mla import flashinfer succeed!")
except ImportError:
    flashinfer = None
    logging.warning("can't import flashinfer, only support cuda12+!")

# import flashinfer
import torch
import torch.nn.functional as F

# from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.ops import KVCache, PyAttentionInputs
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.linear_factory import LinearFactory
from rtp_llm.utils.model_weight import W
from rtp_llm.ops import rtp_llm_ops

class MlaFlashInferPrefillOp(object):
    def __init__(
        self,
        config: GptInitModelParameters, # for LinearFactory
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        page_size: int,
        softmax_extra_scale: float,
        use_mla: bool,
        weights: List[Dict[str, torch.Tensor]] | None,
    ):
        super().__init__()
        if weights is None:
            raise Exception(f"MlaAbsorbAttention need weights but got none")
        self.config = config
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5
        self.weights = weights
        self.token_per_block = page_size
        self.softmax_extra_scale = softmax_extra_scale
        self.use_mla = use_mla
        g_workspace_buffer = torch.empty(
            512 * 1024 * 1024,
            dtype=torch.int8,
            device=self.weights[0].get(W.mla_k_nope_w).device,
        )
        self.prefill_wrapper_ragged = (
            flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
                g_workspace_buffer, "NHD", backend="fa2"
            )
        )

    def support(self, attention_inputs: PyAttentionInputs):
        return self.use_mla and attention_inputs.is_prefill

    def prepare(self, attention_inputs: PyAttentionInputs):
        return rtp_llm_ops.FlashInferMlaAttnParams().fill_mla_params(
            attention_inputs.prefix_lengths,
            attention_inputs.sequence_lengths,
            attention_inputs.input_lengths,
            attention_inputs.kv_cache_block_id_host,
            self.token_per_block,
        )

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        fmha_params: Any,
        layer_id: int,
    ) -> torch.Tensor:

        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)
        self.k_nope_proj = LinearFactory.create_linear_from_weights(
            self.weights[layer_id], W.mla_k_nope_w, W.mla_k_nope_s, None, self.config
        )

        self.v_proj = LinearFactory.create_linear_from_weights(
            self.weights[layer_id], W.mla_v_w, W.mla_v_s, None, self.config
        )

        k_nope = self.k_nope_proj(compressed_kv)
        value_states = self.v_proj(compressed_kv)

        k_nope = k_nope.view(-1, self.num_heads, self.qk_nope_head_dim)
        value_states = value_states.view(-1, self.num_heads, self.qk_nope_head_dim)

        k = k_pe.new_empty(
            k_pe.size(0), self.num_heads, self.qk_rope_head_dim + self.qk_nope_head_dim
        )
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe

        self.prefill_wrapper_ragged.plan(
            fmha_params.qo_indptr,
            fmha_params.page_indptr,
            self.num_heads,
            self.num_heads,
            self.qk_rope_head_dim + self.qk_nope_head_dim,
            self.qk_nope_head_dim,
            sm_scale=(1.0 / (self.qk_rope_head_dim + self.qk_nope_head_dim) ** 0.5)
            * self.softmax_extra_scale,
            causal=True,
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
        )

        attn_output, _ = self.prefill_wrapper_ragged.run_return_lse(q, k, value_states)
        attn_output = attn_output.view(-1, self.num_heads, self.qk_nope_head_dim)
        return attn_output


class MlaFlashInferDecodeOp(object):
    def __init__(
        self,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        token_per_block: int,
        softmax_extra_scale: float,
        use_mla: bool,
        weights: List[Dict[str, torch.Tensor]] | None = None,
    ):
        super().__init__()
        if weights is None:
            raise Exception(f"MlaAbsorbAttention need weights but got none")
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5
        self.token_per_block = token_per_block
        self.softmax_extra_scale = softmax_extra_scale
        self.weights = weights
        self.use_mla = use_mla
        g_workspace_buffer = torch.empty(
            512 * 1024 * 1024,
            dtype=torch.int8,
            device=self.weights[0].get(W.mla_vc).device,
        )
        self.mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            g_workspace_buffer, backend="fa2"
        )

    def support(self, attention_inputs: PyAttentionInputs):
        return self.use_mla and not attention_inputs.is_prefill

    def prepare(self, attention_inputs: PyAttentionInputs):
        return rtp_llm_ops.FlashInferMlaAttnParams().fill_mla_params(
            attention_inputs.prefix_lengths,
            attention_inputs.sequence_lengths,
            attention_inputs.input_lengths,
            attention_inputs.kv_cache_block_id_host,
            self.token_per_block,
        )

    def forward(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        fmha_params: Any,
        layer_id: int,
    ) -> torch.Tensor:

        k_weight = self.weights[layer_id].get(W.mla_kc, None)
        v_weight = self.weights[layer_id].get(W.mla_vc, None)

        compressed_kv = kv_cache.k_cache_base

        self.mla_wrapper.plan(
            fmha_params.qo_indptr,
            fmha_params.page_indptr,
            fmha_params.page_indice,
            fmha_params.kvlen,
            self.num_heads,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.token_per_block,
            False,  # causal
            self.scale,
            q_nope.dtype,
            compressed_kv.dtype,
        )
        q_nope = q_nope.view(-1, self.num_heads, self.qk_nope_head_dim)
        q_pe = q_pe.view(-1, self.num_heads, self.qk_rope_head_dim)

        q_nope = torch.bmm(q_nope.transpose(0, 1), k_weight)
        q_nope = q_nope.transpose(0, 1)

        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        profiler_args = ()

        num_heads = q_nope.shape[1]
        page_size = self.mla_wrapper._page_size
        sm_scale = self.mla_wrapper._sm_scale * self.softmax_extra_scale

        attn_output = torch.empty_like(q_nope)
        self.mla_wrapper._cached_module.run.default(
            self.mla_wrapper._float_workspace_buffer,
            self.mla_wrapper._int_workspace_buffer,
            self.mla_wrapper._plan_info,
            q_nope,
            q_pe,
            compressed_kv,
            k_pe,
            self.mla_wrapper._kv_indices_buf,
            attn_output,
            None,
            1,
            num_heads,
            page_size,
            sm_scale,
            *profiler_args,
        )
        attn_output = attn_output.view(-1, self.num_heads, self.kv_lora_rank)
        attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), v_weight)
        attn_bmm_output = attn_bmm_output.transpose(0, 1)
        return attn_bmm_output