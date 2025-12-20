from typing import Any, Dict, List, Optional

import torch

# import flashinfer
import torch.nn.functional as F
import triton
import triton.language as tl
from flashinfer import (
    BatchMLAPagedAttentionWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)

from flashinfer.jit import gen_batch_mla_module, gen_batch_prefill_module
from flashinfer.utils import is_sm90a_supported

from rtp_llm.models_py.utils.arch import is_cuda
from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs, rtp_llm_ops
from rtp_llm.ops import AttentionConfigs
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


# adapted from sglang/python/sglang/srt/layers/attention/utils.py
@triton.jit
def concat_and_cast_mha_k_kernel(
    k_ptr,
    k_nope_ptr,
    k_rope_ptr,
    head_cnt: tl.constexpr,
    k_stride0: tl.constexpr,
    k_stride1: tl.constexpr,
    nope_stride0: tl.constexpr,
    nope_stride1: tl.constexpr,
    rope_stride0: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    head_range = tl.arange(0, head_cnt)

    k_head_ptr = k_ptr + pid_loc * k_stride0 + head_range[:, None] * k_stride1

    nope_offs = tl.arange(0, nope_dim)

    src_nope_ptr = (
        k_nope_ptr
        + pid_loc * nope_stride0
        + head_range[:, None] * nope_stride1
        + nope_offs[None, :]
    )
    dst_nope_ptr = k_head_ptr + nope_offs[None, :]

    src_nope = tl.load(src_nope_ptr)
    tl.store(dst_nope_ptr, src_nope)

    rope_offs = tl.arange(0, rope_dim)
    src_rope_ptr = k_rope_ptr + pid_loc * rope_stride0 + rope_offs[None, :]
    dst_rope_ptr = k_head_ptr + nope_dim + rope_offs[None, :]
    src_rope = tl.load(src_rope_ptr)
    tl.store(dst_rope_ptr, src_rope)


def concat_and_cast_mha_k_triton(
    k: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
):
    # The source data type will be implicitly converted to the target data type.
    assert (
        len(k.shape) == 3 and len(k_nope.shape) == 3 and len(k_rope.shape) == 3
    ), f"shape should be 3d, but got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    assert (
        k.shape[0] == k_nope.shape[0] and k.shape[0] == k_rope.shape[0]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    assert (
        k.shape[1] == k_nope.shape[1] and 1 == k_rope.shape[1]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    assert (
        k.shape[-1] == k_nope.shape[-1] + k_rope.shape[-1]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"

    nope_dim = k_nope.shape[-1]
    rope_dim = k_rope.shape[-1]
    grid = (k.shape[0],)

    concat_and_cast_mha_k_kernel[grid](
        k,
        k_nope,
        k_rope,
        k.shape[1],
        k.stride(0),
        k.stride(1),
        k_nope.stride(0),
        k_nope.stride(1),
        k_rope.stride(0),
        nope_dim,
        rope_dim,
    )


class MlaFlashInferPrefillOp(object):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        page_size: int,
        softmax_extra_scale: float,
        use_mla: bool,
        weights: List[Dict[str, torch.Tensor]] | None,
        use_trt_fmha: bool = False,
        quant_config: Optional[object] = None,
    ):
        super().__init__()
        if weights is None:
            raise Exception(f"MlaAbsorbAttention need weights but got none")
        self.attn_configs = attn_configs
        self.quant_config = quant_config
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

            self.prefill_wrapper = TRTAttnOp(attn_configs)
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
        self.plan(mla_params)
        # for reuse cache indexed batched
        self.reuse_cache_page_indice = mla_params.reuse_cache_page_indice
        self.qo_indptr = mla_params.qo_indptr
        self.batch_reuse_info_vec = mla_params.batch_reuse_info_vec
        if self.use_trt_fmha:
            return self.prefill_wrapper.prepare(attention_inputs)
        return mla_params

    def plan(self, mla_params: Any):
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

    def _reuse_kv_cache_indexed_batched(
        self,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """使用融合 CUDA kernel 的优化版本"""

        reuse_cache_page_indice = self.reuse_cache_page_indice
        num_blocks = 0
        if reuse_cache_page_indice is not None:
            num_blocks = reuse_cache_page_indice.size(0)

        if num_blocks == 0:
            return compressed_kv, k_pe

        compressed_kv_dim = compressed_kv.size(1)
        qo_indptr = self.qo_indptr
        batch_reuse_info = self.batch_reuse_info_vec

        # 计算总长度
        total_reuse_len = num_blocks * self.token_per_block
        if total_reuse_len == 0:
            return compressed_kv, k_pe

        total_final_len = compressed_kv.size(0) + total_reuse_len

        # 创建输出 tensor
        final_compressed_kv = torch.empty(
            (total_final_len, compressed_kv_dim),
            dtype=compressed_kv.dtype,
            device=compressed_kv.device,
        )
        final_k_pe = torch.empty(
            (total_final_len, k_pe.size(1)),
            dtype=k_pe.dtype,
            device=k_pe.device,
        )

        # 调用融合 kernel
        k_pe = k_pe.contiguous()
        rtp_llm_ops.reuse_kv_cache_indexed_batched(
            final_compressed_kv,
            final_k_pe,
            compressed_kv,
            k_pe,
            kv_cache.kv_cache_base,
            reuse_cache_page_indice,
            batch_reuse_info,
            qo_indptr,
            self.token_per_block,
        )
        return final_compressed_kv, final_k_pe

    def _concat_and_cast_mha_k(self, k_nope, k_pe):
        # Temporary for DeepSeek V3/R1 only, but can generalize if needed
        k_shape = (
            k_nope.shape[0],
            self.num_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
        )
        if (
            is_cuda()
            and (self.num_heads == 128)
            and (self.qk_nope_head_dim == 128)
            and (self.qk_rope_head_dim == 64)
        ):
            k = k_nope.new_empty(*k_shape)
            rtp_llm_ops.mla_k_merge(k, k_nope, k_pe)
        elif is_cuda():
            attn_dtype = k_nope.dtype
            k = k_nope.new_empty(*k_shape, dtype=attn_dtype)
            concat_and_cast_mha_k_triton(k, k_nope, k_pe)
        else:
            k = k_nope.new_empty(*k_shape)
            k[..., : self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim :] = k_pe
        return k

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
            self.weights[layer_id], W.mla_k_nope_w, W.mla_k_nope_s, None, 
            self.quant_config
        )

        self.v_proj = LinearFactory.create_linear_from_weights(
            self.weights[layer_id], W.mla_v_w, W.mla_v_s, None,
            self.quant_config
        )

        k_nope = self.k_nope_proj(compressed_kv)
        value_states = self.v_proj(compressed_kv)

        k_nope = k_nope.view(-1, self.num_heads, self.qk_nope_head_dim)
        value_states = value_states.view(-1, self.num_heads, self.qk_nope_head_dim)
        k = self._concat_and_cast_mha_k(k_nope, k_pe)

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
        self.plan(fmha_params)
        return fmha_params

    def plan(self, fmha_params: Any):
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

        compressed_kv = kv_cache.kv_cache_base

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
        attn_configs: AttentionConfigs,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        use_mla: bool,
        weights: List[Dict[str, torch.Tensor]] | None,
        fmha_config,
        quant_config: Optional[object] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5
        self.attn_configs = attn_configs
        self.quant_config = quant_config
        self.weights = weights
        self.use_mla = use_mla   
        # Get FMHAConfig - will check in support() method
        self.fmha_config = fmha_config
        
        from rtp_llm.ops.compute_ops import TRTAttnOp

        self.fmha_impl = TRTAttnOp(attn_configs)

    def support(self, attention_inputs: PyAttentionInputs):
        # Check if TRT FMHA is enabled
        if not self.fmha_config.enable_paged_trt_fmha:
            return False
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
            self.weights[layer_id], W.mla_k_nope_w, W.mla_k_nope_s, None, 
            self.quant_config
        )

        self.v_proj = LinearFactory.create_linear_from_weights(
            self.weights[layer_id], W.mla_v_w, W.mla_v_s, None,
            self.quant_config
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
        attn_configs: AttentionConfigs,
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
        self.attn_configs = attn_configs
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

        self.fmha_impl = TRTAttnOp(self.attn_configs)
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
