import importlib.util
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import pkg_resources

import rtp_llm.models_py.modules.utils as utils


# 确保是从site-packages加载的Python包，避免同名的cpp flashinfer包冲突
def load_flashinfer_python():
    try:
        dist = pkg_resources.get_distribution("flashinfer-python")
        flashinfer_path = dist.location
        logging.info(f"Found flashinfer-python at: {flashinfer_path}")
        spec = importlib.util.spec_from_file_location(
            "flashinfer", os.path.join(flashinfer_path, "flashinfer", "__init__.py")
        )
        if spec and spec.origin and "site-packages" in spec.origin:
            flashinfer_module = importlib.util.module_from_spec(spec)
            sys.modules["flashinfer"] = flashinfer_module
            spec.loader.exec_module(flashinfer_module)
            logging.info(f"load flashinfer-python succeed! spec: {spec}")
            return flashinfer_module
        else:
            logging.warning(f"can't load flashinfer-python, spec: {spec}")
    except Exception as e:
        logging.warning(f"Failed to load flashinfer-python: {e}")
    return None


flashinfer_python = load_flashinfer_python()

# import flashinfer
import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.linear_factory import LinearFactory

# from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.ops import KVCache, PyAttentionInputs, rtp_llm_ops
from rtp_llm.utils.model_weight import W


def check_attention_inputs(attention_inputs: PyAttentionInputs) -> None:
    device = attention_inputs.input_lengths.device
    dtype = torch.int32

    default_tensors = {
        "prefix_lengths": torch.zeros(0, dtype=dtype, device=device),
        "sequence_lengths": torch.zeros(0, dtype=dtype, device=device),
        "kv_cache_block_id_host": torch.zeros(0, dtype=dtype, device=device),
    }

    for attr_name, default_tensor in default_tensors.items():
        if getattr(attention_inputs, attr_name) is None:
            setattr(attention_inputs, attr_name, default_tensor)


class MlaFlashInferPrefillOp(object):
    def __init__(
        self,
        config: GptInitModelParameters,  # for LinearFactory
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
            flashinfer_python.prefill.BatchPrefillWithRaggedKVCacheWrapper(
                g_workspace_buffer, "NHD", backend="fa2"
            )
        )

    def support(self, attention_inputs: PyAttentionInputs):
        return self.use_mla and attention_inputs.is_prefill

    def prepare(self, attention_inputs: PyAttentionInputs):
        check_attention_inputs(attention_inputs)
        return rtp_llm_ops.fill_mla_params(
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
        self.mla_wrapper = flashinfer_python.mla.BatchMLAPagedAttentionWrapper(
            g_workspace_buffer, backend="fa2"
        )

    def support(self, attention_inputs: PyAttentionInputs):
        return self.use_mla and not attention_inputs.is_prefill

    def prepare(self, attention_inputs: PyAttentionInputs):
        check_attention_inputs(attention_inputs)
        return rtp_llm_ops.fill_mla_params(
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


class TrtV2PrefillAttentionOp(object):
    def __init__(
        self,
        config: GptInitModelParameters,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        use_mla: bool,
        weights: List[Dict[str, torch.Tensor]] | None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5
        self.config = config
        self.weights = weights
        self.use_mla = use_mla
        from libth_transformer.rtp_llm_ops import TRTAttnOp

        self.fmha_impl = TRTAttnOp(self.config)

    def support(self, attention_inputs: PyAttentionInputs):
        return (
            self.use_mla
            and attention_inputs.is_prefill
            and self.fmha_impl.support(attention_inputs)
        )

    def prepare(self, attention_inputs: PyAttentionInputs):
        return self.fmha_impl.prepare(attention_inputs)

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

        pad_len = self.qk_rope_head_dim
        value_states = F.pad(value_states, (0, pad_len))

        fmha_input = torch.stack([q, k, value_states], dim=1)
        fmha_input = fmha_input.reshape(q.shape[0], -1)
        kv_cache: Optional[KVCache] = None
        attn_output = self.fmha_impl.forward(fmha_input, kv_cache, fmha_params)
        attn_output = attn_output.view(
            q.shape[0], self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim
        )
        attn_output, _ = torch.split(
            attn_output,
            [
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
            ],
            dim=-1,
        )
        return attn_output


"""
class TrtV2PrefillAttention(nn.Module):
    def __init__(
        self,
        config: GptInitModelParameters,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        k_nope_weight: torch.Tensor | None,
        v_weight: torch.Tensor | None,
    ):
        super().__init__()
        if k_nope_weight is None or v_weight is None:
            raise Exception(
                f"MlaAbsorbAttention need v_weight and k_weight but got none"
            )
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5
        self.v_weight = v_weight
        self.k_nope_weight = k_nope_weight
        self.config = config
    def forward(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        attention_inputs: PyAttentionInputs,
    ) -> torch.Tensor:
        q_nope = q_nope.view(-1, self.num_heads, self.qk_nope_head_dim)
        q_pe = q_pe.view(-1, self.num_heads, self.qk_rope_head_dim)
        q = torch.cat([q_nope, q_pe], dim=-1)
        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)
        k_nope = F.linear(compressed_kv, self.k_nope_weight.transpose(0, 1), None)
        k_nope = k_nope.view(-1, self.num_heads, self.qk_nope_head_dim)
        k = k_pe.new_empty(
            k_pe.size(0), self.num_heads, self.qk_rope_head_dim + self.qk_nope_head_dim
        )
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe
        value_states = F.linear(compressed_kv, self.v_weight.transpose(0, 1), None)
        value_states = value_states.view(-1, self.num_heads, self.qk_nope_head_dim)
        pad_len = self.qk_rope_head_dim
        value_states = F.pad(value_states, (0, pad_len))
        from libth_transformer.rtp_llm_ops import TRTAttnOp
        self.fmha_impl = TRTAttnOp(self.config)
        self.support_: bool = self.fmha_impl.support(attention_inputs)
        if self.support_:
            self.fmha_params = self.fmha_impl.prepare(attention_inputs)
        fmha_input = torch.stack([q, k, value_states], dim=1)
        fmha_input = fmha_input.reshape(q.shape[0], -1)
        attn_output = self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)
        attn_output = attn_output.view(
            q.shape[0], self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim
        )
        attn_output, _ = torch.split(
            attn_output,
            [
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
            ],
            dim=-1,
        )
        return attn_output
"""
