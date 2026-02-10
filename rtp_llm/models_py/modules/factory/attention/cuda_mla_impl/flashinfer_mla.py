import logging
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

from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
from rtp_llm.models_py.utils.arch import is_cuda
from rtp_llm.ops import AttentionConfigs, KvCacheDataType
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs, rtp_llm_ops
from rtp_llm.utils.model_weight import W

g_workspace_buffer = None
warm_up_done = False


def warmup_flashinfer_python():
    global warm_up_done
    if warm_up_done:
        return
    warm_up_done = True
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
        "prefix_lengths": torch.empty(0, dtype=dtype, device=device),
        "sequence_lengths": torch.empty(0, dtype=dtype, device=device),
        "kv_cache_block_id_host": torch.empty(0, dtype=dtype, device=device),
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
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        page_size: int,
        softmax_extra_scale: float,
        use_mla: bool,
        weights: List[Dict[str, torch.Tensor]] | None,
        quant_config: Optional[object] = None,
        kv_cache_dtype: KvCacheDataType = KvCacheDataType.BASE,
    ):
        super().__init__()

        if weights is None:
            raise Exception(f"MlaAbsorbAttention need weights but got none")
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
        self.kv_cache_type = kv_cache_dtype
        global g_workspace_buffer
        if g_workspace_buffer is None:
            g_workspace_buffer = torch.empty(
                512 * 1024 * 1024,
                dtype=torch.int8,
                device=self.weights[0].get(W.mla_k_nope_w).device,
            )

        self.prefill_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            g_workspace_buffer,
            "NHD",
            backend="auto",
            use_cuda_graph=False,
        )

    def plan(self, mla_params: Any):
        self.prefill_wrapper.plan(
            mla_params.qo_indptr_d,
            mla_params.prefill_ragged_kv_len_indptr_d,
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
        self.reuse_cache_page_indice = mla_params.reuse_cache_page_indice_d
        self.qo_indptr = mla_params.qo_indptr_d
        self.batch_reuse_info_vec = mla_params.batch_reuse_info_vec_d
        self.total_kv_lens = mla_params.prefill_page_indptr_d[-1].item()
        self.block_table = mla_params.page_indice_d.unsqueeze(0)
        self.workspace_starts = torch.zeros(
            1, dtype=torch.int32, device=self.block_table.device
        )
        self.seq_lens = mla_params.prefill_page_indptr_d[-1:]

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

        if self.kv_cache_type == KvCacheDataType.FP8:
            final_compressed_kv = torch.empty(
                [self.total_kv_lens, self.kv_lora_rank],
                dtype=torch.bfloat16,
                device=compressed_kv.device,
            )
            final_k_pe = torch.empty(
                [self.total_kv_lens, self.qk_rope_head_dim],
                dtype=torch.bfloat16,
                device=compressed_kv.device,
            )
            rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache(
                kv_cache.kv_cache_base.view(torch.uint8),
                final_compressed_kv,
                final_k_pe,
                self.block_table,
                self.seq_lens,
                self.workspace_starts,
                batch_size=1,  # ragged
            )
            return final_compressed_kv, final_k_pe

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
        layer_id: int,
    ) -> torch.Tensor:

        compressed_kv, k_pe = self._reuse_kv_cache_indexed_batched(
            compressed_kv, k_pe, kv_cache
        )

        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)
        self.k_nope_proj = LinearFactory.create_linear_from_weights(
            self.weights[layer_id],
            W.mla_k_nope_w,
            W.mla_k_nope_s,
            None,
            self.quant_config,
        )

        self.v_proj = LinearFactory.create_linear_from_weights(
            self.weights[layer_id], W.mla_v_w, W.mla_v_s, None, self.quant_config
        )

        k_nope = self.k_nope_proj(compressed_kv)
        value_states = self.v_proj(compressed_kv)

        k_nope = k_nope.view(-1, self.num_heads, self.qk_nope_head_dim)
        value_states = value_states.view(-1, self.num_heads, self.qk_nope_head_dim)
        k = self._concat_and_cast_mha_k(k_nope, k_pe)

        # TODO: add TRT prefill support
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
        is_sparse: bool,
        weights: List[Dict[str, torch.Tensor]] | None = None,
        max_bs: int = 0,
        max_context_len: int = 0,
        num_tokens: int = 0,
        is_cuda_graph: bool = False,
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
        self.is_sparse = is_sparse
        self.use_cuda_graph = is_cuda_graph
        global g_workspace_buffer
        self.kv_indices_d = torch.empty(
            ((max_context_len + self.token_per_block - 1) // self.token_per_block)
            * max_bs,
            dtype=torch.int32,
            device="cuda",
        )
        self.qo_indptr_h = torch.arange(0, max_bs + 1, dtype=torch.int32, device="cpu")
        self.kv_indptr_h = torch.zeros((max_bs + 1,), dtype=torch.int32, device="cpu")
        self.kv_len_arr_h = torch.ones((max_bs,), dtype=torch.int32, device="cpu")

        if g_workspace_buffer is None:
            g_workspace_buffer = torch.empty(
                512 * 1024 * 1024,
                dtype=torch.int8,
                device=self.weights[0].get(W.mla_vc).device,
            )

        self.mla_wrapper = BatchMLAPagedAttentionWrapper(
            g_workspace_buffer,
            backend="auto",
            use_cuda_graph=is_cuda_graph,
            qo_indptr=self.qo_indptr_h,
            kv_indptr=self.kv_indptr_h,
            kv_indices=self.kv_indices_d,
            kv_len_arr=self.kv_len_arr_h,
        )

    def plan(self, fmha_params: Any):
        if self.use_cuda_graph and self.kv_indices_d.size(
            0
        ) < fmha_params.page_indice_d.size(0):
            logging.error(
                f"kv_indices_d.size(0): {self.kv_indices_d.size(0)}, fmha_params.page_indice_d.size(0): {fmha_params.page_indice_d.size(0)}"
            )
            raise ValueError(
                f"kv_indices_d.size(0) < fmha_params.page_indice_d.size(0)"
            )

        self.mla_wrapper.plan(
            fmha_params.qo_indptr_h,
            fmha_params.decode_page_indptr_h,
            fmha_params.page_indice_d,
            fmha_params.kvlen_h,
            self.num_heads,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.token_per_block,
            True,  # causal
            self.scale * self.softmax_extra_scale,
            torch.bfloat16,
            torch.bfloat16,
        )

    def forward(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
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

        attn_output = torch.empty_like(q_nope)
        self.mla_wrapper.run(q_nope, q_pe, compressed_kv, k_pe, attn_output)

        attn_output = attn_output.view(-1, self.num_heads, self.kv_lora_rank)
        attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), v_weight)
        attn_bmm_output = attn_bmm_output.transpose(0, 1)

        return attn_bmm_output
