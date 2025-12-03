from typing import Any, Dict, List, Optional

import torch

# import flashinfer
import torch.nn.functional as F
from flashinfer import (
    BatchMLAPagedAttentionWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from flashinfer.jit import gen_batch_mla_module, gen_batch_prefill_module
from flashinfer.utils import is_sm90a_supported

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.linear_factory import LinearFactory

# from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs, rtp_llm_ops
from rtp_llm.utils.model_weight import W

g_workspace_buffer = None


def warmup_flashinfer_python():
    modules = []
    for backend in ["fa2", "fa3"]:
        if backend == "fa3" and not is_sm90a_supported(torch.device("cuda")):
            continue
        modules.append(
            gen_batch_prefill_module(
                backend,
                torch.bfloat16,
                torch.bfloat16,
                torch.bfloat16,
                torch.int32,
                192,
                128,
                0,
                False,
                False,
                False,
            )
        )

    for backend in ["fa2", "fa3"]:
        if backend == "fa3" and not is_sm90a_supported(torch.device("cuda")):
            continue
        modules.append(
            gen_batch_mla_module(
                backend,
                torch.bfloat16,
                torch.bfloat16,
                torch.bfloat16,
                torch.int32,
                512,
                64,
                False,
            )
        )


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
        use_trt_fmha: bool = False,
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
        self.use_trt_fmha = use_trt_fmha
        global g_workspace_buffer
        if g_workspace_buffer is None:
            g_workspace_buffer = torch.empty(
                512 * 1024 * 1024,
                dtype=torch.int8,
                device=self.weights[0].get(W.mla_k_nope_w).device,
            )
        if use_trt_fmha:
            from rtp_llm.ops.compute_ops import TRTAttnOp

            self.prefill_wrapper = TRTAttnOp(self.config)
            return

        self.prefill_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            g_workspace_buffer, "NHD", backend="auto"
        )

    def support(self, attention_inputs: PyAttentionInputs):
        return self.use_mla and attention_inputs.is_prefill

    def prepare(self, attention_inputs: PyAttentionInputs):
        check_attention_inputs(attention_inputs)
        mla_params = rtp_llm_ops.fill_mla_params(
            attention_inputs.prefix_lengths,
            attention_inputs.sequence_lengths,
            attention_inputs.input_lengths,
            attention_inputs.kv_cache_block_id_host,
            self.token_per_block,
        )
        self.prefill_wrapper.plan(
            mla_params.qo_indptr,
            mla_params.prefill_page_indptr,
            self.num_heads,
            self.num_heads,
            self.qk_rope_head_dim + self.qk_nope_head_dim,
            self.qk_nope_head_dim,
            sm_scale=(1.0 / (self.qk_rope_head_dim + self.qk_nope_head_dim) ** 0.5)
            * self.softmax_extra_scale,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        # for reuse cache indexed batched
        self.reuse_cache_page_indice = mla_params.reuse_cache_page_indice
        self.qo_indptr = mla_params.qo_indptr
        self.batch_reuse_info_vec = mla_params.batch_reuse_info_vec
        if self.use_trt_fmha:
            return self.prefill_wrapper.prepare(attention_inputs)
        return mla_params

    def _reuse_kv_cache_indexed_batched(
        self,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """使用索引操作的优化版本 - 根据kv_len和q_len的差值确定concat位置"""

        # 获取参数
        reuse_cache_page_indice = self.reuse_cache_page_indice  # [5, 3]
        num_blocks = reuse_cache_page_indice.size(0)  # 2

        if num_blocks == 0:
            return compressed_kv, k_pe

        compressed_kv_dim = compressed_kv.size(1)
        qo_indptr = self.qo_indptr  # [0, 17, 29, 47, 63]

        # 准备结果tensor
        batch_reuse_info = self.batch_reuse_info_vec.cpu().tolist()
        qo_indptr_list = qo_indptr.cpu().tolist()
        total_reuse_len = sum(info[1] for info in batch_reuse_info)
        if total_reuse_len == 0:
            return compressed_kv, k_pe

        # 创建最终的tensor
        final_compressed_kv = torch.empty(
            (compressed_kv.size(0) + total_reuse_len, compressed_kv.size(1)),
            dtype=compressed_kv.dtype,
            device=compressed_kv.device,
        )
        final_k_pe = torch.empty(
            (k_pe.size(0) + total_reuse_len, k_pe.size(1)),
            dtype=k_pe.dtype,
            device=k_pe.device,
        )

        # 按batch处理，将reuse cache和compressed_kv按正确位置concat
        compressed_kv_offset = 0
        final_offset = 0

        for (
            batch_idx,
            reuse_len,
            block_start_idx,
            blocks_needed,
        ) in batch_reuse_info:
            batch_q_len = qo_indptr_list[batch_idx + 1] - qo_indptr_list[batch_idx]

            if reuse_len > 0:
                # 获取这个batch需要的reuse blocks
                batch_cache_indices = reuse_cache_page_indice[
                    block_start_idx : block_start_idx + blocks_needed
                ]

                # 从kv_cache中获取对应的blocks
                batch_cache_blocks = kv_cache.k_cache_base[batch_cache_indices]
                batch_cache_blocks = batch_cache_blocks.view(
                    -1, batch_cache_blocks.size(-1)
                )

                # 将reuse cache放到最终tensor的前面部分
                final_compressed_kv[final_offset : final_offset + reuse_len] = (
                    batch_cache_blocks[:, :compressed_kv_dim]
                )
                final_k_pe[final_offset : final_offset + reuse_len] = (
                    batch_cache_blocks[:, compressed_kv_dim:]
                )
                final_offset += reuse_len

            # 将当前batch的compressed_kv放到对应位置
            batch_compressed_kv_start = compressed_kv_offset
            batch_compressed_kv_end = compressed_kv_offset + batch_q_len

            final_compressed_kv[final_offset : final_offset + batch_q_len] = (
                compressed_kv[batch_compressed_kv_start:batch_compressed_kv_end]
            )
            final_k_pe[final_offset : final_offset + batch_q_len] = k_pe[
                batch_compressed_kv_start:batch_compressed_kv_end
            ]

            final_offset += batch_q_len
            compressed_kv_offset += batch_q_len

        return final_compressed_kv, final_k_pe

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        fmha_params: Any,
        layer_id: int,
    ) -> torch.Tensor:

        # trt fmha not support reuse cache yet due to stack
        if not self.use_trt_fmha:
            compressed_kv, k_pe = self._reuse_kv_cache_indexed_batched(
                compressed_kv, k_pe, kv_cache
            )

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

        if self.use_trt_fmha:
            pad_len = self.qk_rope_head_dim
            value_states = F.pad(value_states, (0, pad_len))
            # trt fmha not support reuse cache yet due to stack
            fmha_input = torch.stack([q, k, value_states], dim=1)
            fmha_input = fmha_input.reshape(q.shape[0], -1)
            kv_cache: Optional[KVCache] = None
            attn_output = self.prefill_wrapper.forward(
                fmha_input, kv_cache, fmha_params
            )
            attn_output = attn_output.view(
                q.shape[0],
                self.num_heads,
                self.qk_nope_head_dim + self.qk_rope_head_dim,
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

        attn_output = self.prefill_wrapper.run(q, k, value_states)
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
        global g_workspace_buffer
        if g_workspace_buffer is None:
            g_workspace_buffer = torch.empty(
                512 * 1024 * 1024,
                dtype=torch.int8,
                device=self.weights[0].get(W.mla_vc).device,
            )
        self.mla_wrapper = BatchMLAPagedAttentionWrapper(
            g_workspace_buffer, backend="auto"
        )

    def support(self, attention_inputs: PyAttentionInputs):
        return self.use_mla

    def prepare(self, attention_inputs: PyAttentionInputs):
        check_attention_inputs(attention_inputs)
        fmha_params = rtp_llm_ops.fill_mla_params(
            attention_inputs.prefix_lengths,
            attention_inputs.sequence_lengths,
            attention_inputs.input_lengths,
            attention_inputs.kv_cache_block_id_host,
            self.token_per_block,
        )
        self.mla_wrapper.plan(
            fmha_params.qo_indptr,
            fmha_params.decode_page_indptr,
            fmha_params.page_indice,
            fmha_params.kvlen,
            self.num_heads,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.token_per_block,
            False,  # causal
            self.scale,
            torch.bfloat16,
            torch.bfloat16,
        )
        return fmha_params

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


"""
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
        from rtp_llm.ops.compute_ops import TRTAttnOp

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
        from rtp_llm.ops.compute_ops import TRTAttnOp
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
