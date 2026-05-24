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
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, rtp_llm_ops
from rtp_llm.utils.model_weight import W

try:
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata

    _HAS_FLASH_MLA = True
except ImportError:
    _HAS_FLASH_MLA = False

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
        "kv_cache_kernel_block_id_host": torch.empty(0, dtype=dtype, device=device),
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
    _triton_compat_warned = False  # Class variable to track warning status

    def __init__(
        self,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        v_head_dim: int,
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
        self.v_head_dim = v_head_dim
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
                device=self.weights[0].get(W.mla_kv_b_w).device,
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
            self.v_head_dim,
            sm_scale=(1.0 / (self.qk_rope_head_dim + self.qk_nope_head_dim) ** 0.5)
            * self.softmax_extra_scale,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        self.reuse_cache_page_indice = mla_params.reuse_cache_page_indice_d
        self.qo_indptr = mla_params.qo_indptr_d
        self.batch_reuse_info_vec = mla_params.batch_reuse_info_vec_d
        self.total_kv_lens = mla_params.prefill_ragged_kv_len_indptr_d[-1].item()
        self.block_table = mla_params.page_indice_d.unsqueeze(0)
        self.workspace_starts = torch.zeros(
            1, dtype=torch.int32, device=self.block_table.device
        )
        self.seq_lens = mla_params.prefill_ragged_kv_len_indptr_d[-1:]

    def _reuse_kv_cache_indexed_batched(
        self,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
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
        elif is_cuda() and self._is_triton_compatible():
            # Triton kernel requires dimensions to be power of 2
            attn_dtype = k_nope.dtype
            k = k_nope.new_empty(*k_shape, dtype=attn_dtype)
            concat_and_cast_mha_k_triton(k, k_nope, k_pe)
        else:
            # Fallback to PyTorch native operations for non-power-of-2 dimensions
            k = k_nope.new_empty(*k_shape)
            k[..., : self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim :] = k_pe
        return k

    def _is_triton_compatible(self):
        """Check if dimensions are compatible with Triton kernel (must be power of 2)."""

        def is_power_of_2(n):
            return n > 0 and (n & (n - 1)) == 0

        compatible = (
            is_power_of_2(self.qk_nope_head_dim)
            and is_power_of_2(self.qk_rope_head_dim)
            and is_power_of_2(self.num_heads)
        )
        return compatible

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
    ) -> torch.Tensor:

        compressed_kv, k_pe = self._reuse_kv_cache_indexed_batched(
            compressed_kv, k_pe, kv_cache
        )

        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)
        self.kv_b_proj = LinearFactory.create_linear_from_weights(
            self.weights[layer_id],
            W.mla_kv_b_w,
            W.mla_kv_b_s,
            None,
            self.quant_config,
        )

        kv = self.kv_b_proj(compressed_kv)
        expected_out = self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)
        if kv.shape[-1] != expected_out:
            w = self.weights[layer_id].get(W.mla_kv_b_w)
            s = self.weights[layer_id].get(W.mla_kv_b_s)
            logging.error(
                f"[MLA-PREFILL-DEBUG] kv_b_proj output mismatch: "
                f"kv={list(kv.shape)} expected_out={expected_out} "
                f"num_heads={self.num_heads} nope={self.qk_nope_head_dim} "
                f"v_dim={self.v_head_dim} w_shape={list(w.shape) if w is not None else None} "
                f"w_dtype={w.dtype if w is not None else None} "
                f"s_shape={list(s.shape) if s is not None else None} "
                f"ckv={list(compressed_kv.shape)} kpe={list(k_pe.shape)} "
                f"all_keys={list(self.weights[layer_id].keys())}"
            )
        kv = kv.view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[:, :, : self.qk_nope_head_dim]
        value_states = kv[:, :, self.qk_nope_head_dim :]
        k = self._concat_and_cast_mha_k(k_nope, k_pe)

        # TODO: add TRT prefill support
        attn_output = self.prefill_wrapper.run(q, k, value_states)
        attn_output = attn_output.view(-1, self.num_heads, self.v_head_dim)
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
        kv_cache_dtype: KvCacheDataType = KvCacheDataType.BASE,
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
        self._fp8_kv = kv_cache_dtype == KvCacheDataType.FP8
        self._fmha_params = None
        self._sched_meta = None
        self._fp8_prefill_sched_meta = None
        self._fp8_prefill_indices = None
        self._fp8_prefill_topk_length = None
        self._fp8_prefill_qo_indptr_h = None
        self._fp8_prefill_q_lens_h = None
        self._fp8_prefill_max_q_len = 0
        self._fp8_prefill_total_q = 0
        global g_workspace_buffer

        if self._fp8_kv:
            if not _HAS_FLASH_MLA:
                raise ImportError("flash_mla is required for FP8 kv_cache MLA decode")
            self._fp8_max_bs = max_bs
            self._fp8_max_context_len = max_context_len
            align = self._FP8_SPARSE_TOPK_ALIGN
            padded_topk_max = ((max_context_len + align - 1) // align) * align
            padded_topk_max = max(padded_topk_max, align)
            self._fp8_indices_buf = torch.full(
                (max_bs, 1, padded_topk_max),
                -1,
                dtype=torch.int32,
                device="cuda",
            )
            self._fp8_topk_len_buf = torch.zeros(
                max_bs, dtype=torch.int32, device="cuda"
            )
            self._fp8_position_buf = torch.arange(
                padded_topk_max, dtype=torch.int32, device="cuda"
            )
            self._fp8_block_ids_buf = (
                self._fp8_position_buf // self.token_per_block
            ).to(torch.long)
            self._fp8_offsets_buf = self._fp8_position_buf % self.token_per_block
            self._fp8_plan_B = 0
        else:
            self.kv_indices_d = torch.empty(
                ((max_context_len + self.token_per_block - 1) // self.token_per_block)
                * max_bs,
                dtype=torch.int32,
                device="cuda",
            )
            self.qo_indptr_h = torch.arange(
                0, max_bs + 1, dtype=torch.int32, device="cpu"
            )
            self.kv_indptr_h = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device="cpu"
            )
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
        self._fmha_params = fmha_params

        if self._fp8_kv:
            self._reset_fp8_sched_meta()
            block_table = self._build_block_table(fmha_params)
            B = int(fmha_params.kvlen_h.size(0))
            cache_seqlens = fmha_params.kvlen_h[:B].to(
                dtype=torch.int32, device=block_table.device
            )
            total_q = (
                int(fmha_params.qo_indptr_h[B].item())
                if B > 0 and fmha_params.qo_indptr_h.numel() > B
                else B
            )
            self._fp8_plan_B = B
            self._clear_fp8_prefill_absorb_plan()
            if total_q > B:
                self._build_fp8_prefill_absorb_plan(
                    fmha_params, block_table, cache_seqlens, total_q
                )
                return
            self._fp8_indices_buf[:B].fill_(-1)
            self._fp8_topk_len_buf[:B].copy_(cache_seqlens)
            for i in range(B):
                seqlen = int(cache_seqlens[i].item())
                if seqlen == 0:
                    continue
                positions = torch.arange(
                    seqlen, device=block_table.device, dtype=torch.long
                )
                block_ids = positions // self.token_per_block
                offsets = (positions % self.token_per_block).to(torch.int32)
                phys_blocks = block_table[i, block_ids]
                self._fp8_indices_buf[i, 0, :seqlen] = (
                    phys_blocks * self.token_per_block + offsets
                )
            return

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

    def _reset_fp8_sched_meta(self):
        if self._sched_meta is None:
            self._sched_meta, _ = get_mla_metadata()
        self._sched_meta.tile_scheduler_metadata = None
        self._sched_meta.num_splits = None

    def _clear_fp8_prefill_absorb_plan(self) -> None:
        self._fp8_prefill_sched_meta = None
        self._fp8_prefill_indices = None
        self._fp8_prefill_topk_length = None
        self._fp8_prefill_qo_indptr_h = None
        self._fp8_prefill_q_lens_h = None
        self._fp8_prefill_max_q_len = 0
        self._fp8_prefill_total_q = 0

    def _build_fp8_prefill_absorb_plan(
        self,
        fmha_params: Any,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        total_q: int,
    ) -> None:
        B = int(cache_seqlens.size(0))
        if B == 0 or total_q == 0:
            return

        qo_indptr_h = fmha_params.qo_indptr_h[: B + 1].to(
            dtype=torch.int32, device="cpu"
        )
        kvlen_h = fmha_params.kvlen_h[:B].to(dtype=torch.int32, device="cpu")
        q_lens_h = qo_indptr_h[1:] - qo_indptr_h[:B]
        max_q_len = int(q_lens_h.max().item())
        max_kv_len = int(kvlen_h.max().item())
        if max_q_len <= 0 or max_kv_len <= 0:
            return
        align = self._FP8_SPARSE_TOPK_ALIGN
        padded_topk = ((max_kv_len + align - 1) // align) * align

        indices = torch.full(
            (B, max_q_len, padded_topk),
            -1,
            dtype=torch.int32,
            device=block_table.device,
        )
        for i in range(B):
            q_len = int(q_lens_h[i].item())
            kv_len = int(kvlen_h[i].item())
            if q_len <= 0 or kv_len <= 0:
                continue
            prefix_len = max(0, kv_len - q_len)
            positions = torch.arange(
                kv_len, dtype=torch.long, device=block_table.device
            )
            block_ids = positions // self.token_per_block
            offsets = (positions % self.token_per_block).to(torch.int32)
            dense_indices = (
                block_table[i, block_ids].to(torch.int32) * self.token_per_block
                + offsets
            )
            for q_pos in range(q_len):
                causal_len = min(prefix_len + q_pos + 1, kv_len)
                indices[i, q_pos, :causal_len] = dense_indices[:causal_len]

        self._fp8_prefill_sched_meta, _ = get_mla_metadata()
        self._fp8_prefill_indices = indices
        self._fp8_prefill_topk_length = kvlen_h.to(
            dtype=torch.int32, device=block_table.device
        )
        self._fp8_prefill_qo_indptr_h = qo_indptr_h
        self._fp8_prefill_q_lens_h = q_lens_h
        self._fp8_prefill_max_q_len = max_q_len
        self._fp8_prefill_total_q = total_q

    def plan_cuda_graph(self, attn_inputs: PyAttentionInputs) -> bool:
        if not self._fp8_kv:
            return False

        sequence_lengths = attn_inputs.sequence_lengths_plus_1_d
        if sequence_lengths is None or sequence_lengths.numel() == 0:
            sequence_lengths = attn_inputs.sequence_lengths + 1
        self._plan_fp8_from_device(
            sequence_lengths,
            attn_inputs.kv_cache_kernel_block_id_device,
        )
        return True

    def _plan_fp8_from_device(
        self,
        sequence_lengths: torch.Tensor,
        block_table: torch.Tensor,
    ) -> None:
        self._reset_fp8_sched_meta()
        self._clear_fp8_prefill_absorb_plan()
        B = int(sequence_lengths.shape[0])
        self._fp8_plan_B = B
        if B == 0:
            return

        if B > self._fp8_indices_buf.shape[0]:
            raise ValueError(
                f"FP8 MLA decode batch size {B} exceeds reserved "
                f"{self._fp8_indices_buf.shape[0]}"
            )
        if block_table is None or block_table.dim() != 2:
            raise ValueError("FP8 MLA CUDA graph requires a 2-D block table")

        device = self._fp8_indices_buf.device
        block_table = block_table[:B].to(device=device, dtype=torch.int32)
        cache_seqlens = sequence_lengths[:B].to(device=device, dtype=torch.int32)
        topk_cap = min(
            self._fp8_indices_buf.shape[-1],
            int(block_table.shape[1]) * self.token_per_block,
        )

        self._fp8_indices_buf[:B].fill_(-1)
        self._fp8_topk_len_buf[:B].copy_(cache_seqlens)
        if topk_cap <= 0:
            return

        positions = self._fp8_position_buf[:topk_cap]
        block_ids = self._fp8_block_ids_buf[:topk_cap].unsqueeze(0).expand(B, -1)
        offsets = self._fp8_offsets_buf[:topk_cap].unsqueeze(0)

        phys_blocks = torch.gather(block_table, 1, block_ids)
        dense_indices = phys_blocks * self.token_per_block + offsets
        valid = positions.unsqueeze(0) < cache_seqlens.unsqueeze(1)
        self._fp8_indices_buf[:B, 0, :topk_cap].copy_(
            torch.where(valid, dense_indices, -1)
        )

    def _build_block_table(self, fmha_params: Any) -> torch.Tensor:
        indptr_h = fmha_params.decode_page_indptr_h
        batch_size = int(indptr_h.size(0)) - 1
        lengths = indptr_h[1 : batch_size + 1] - indptr_h[:batch_size]
        max_blocks = int(lengths.max().item()) if batch_size > 0 else 0
        page_indices_d = fmha_params.page_indice_d
        page_indices_h = page_indices_d.cpu()
        block_table = torch.zeros((batch_size, max_blocks), dtype=torch.int32)
        for i in range(batch_size):
            start = int(indptr_h[i].item())
            n = int(lengths[i].item())
            if n > 0:
                block_table[i, :n] = page_indices_h[start : start + n]
        return block_table.to(device=page_indices_d.device)

    def forward(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
    ) -> torch.Tensor:

        k_weight = self.weights[layer_id].get(W.mla_kc, None)
        v_weight = self.weights[layer_id].get(W.mla_vc, None)

        q_nope = q_nope.view(-1, self.num_heads, self.qk_nope_head_dim)
        q_pe = q_pe.view(-1, self.num_heads, self.qk_rope_head_dim)

        q_nope = torch.bmm(q_nope.transpose(0, 1), k_weight)
        q_nope = q_nope.transpose(0, 1)

        if self._fp8_kv:
            return self._forward_fp8(q_nope, q_pe, kv_cache, v_weight, layer_id)

        compressed_kv = kv_cache.kv_cache_base.view(
            -1, self.token_per_block, self.kv_lora_rank + self.qk_rope_head_dim
        )

        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        attn_output = torch.empty_like(q_nope)
        self.mla_wrapper.run(q_nope, q_pe, compressed_kv, k_pe, attn_output)

        attn_output = attn_output.view(-1, self.num_heads, self.kv_lora_rank)
        attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), v_weight)
        attn_bmm_output = attn_bmm_output.transpose(0, 1)

        return attn_bmm_output

    _FP8_SPARSE_TOPK_ALIGN = 64

    def _build_dense_indices(
        self, block_table: torch.Tensor, cache_seqlens: torch.Tensor, B: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build dense indices for sparse decode kernel (all tokens per request).

        Returns (indices [B, 1, padded_topk], topk_length [B]).
        Pads topk dimension to a multiple of _FP8_SPARSE_TOPK_ALIGN.
        """
        max_topk = int(cache_seqlens.max().item())
        align = self._FP8_SPARSE_TOPK_ALIGN
        padded_topk = ((max_topk + align - 1) // align) * align
        padded_topk = max(padded_topk, align)
        indices = torch.full(
            (B, 1, padded_topk), -1, dtype=torch.int32, device=block_table.device
        )
        for i in range(B):
            seqlen = int(cache_seqlens[i].item())
            if seqlen == 0:
                continue
            positions = torch.arange(
                seqlen, device=block_table.device, dtype=torch.long
            )
            block_ids = positions // self.token_per_block
            offsets = (positions % self.token_per_block).to(torch.int32)
            phys_blocks = block_table[i, block_ids]
            indices[i, 0, :seqlen] = phys_blocks * self.token_per_block + offsets
        topk_length = cache_seqlens.to(dtype=torch.int32, device=block_table.device)
        return indices, topk_length

    def _forward_fp8(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        v_weight: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        if self._fp8_prefill_indices is not None:
            return self._forward_fp8_prefill_absorb(
                q_nope, q_pe, kv_cache, v_weight, layer_id
            )

        B = q_nope.size(0)
        q_absorbed = torch.cat([q_nope, q_pe], dim=-1)
        q_4d = q_absorbed.unsqueeze(1)

        kv_cache_u8 = kv_cache.kv_cache_base.view(torch.uint8)
        fp8_per_token = (
            self.kv_lora_rank + self.kv_lora_rank // 128 * 4 + self.qk_rope_head_dim * 2
        )
        kv_cache_paged = kv_cache_u8.view(-1, self.token_per_block, 1, fp8_per_token)

        if layer_id == 0:
            self._sched_meta.tile_scheduler_metadata = None
            self._sched_meta.num_splits = None

        indices = self._fp8_indices_buf[:B]
        topk_length = self._fp8_topk_len_buf[:B]

        try:
            attn_out, _ = flash_mla_with_kvcache(
                q=q_4d,
                k_cache=kv_cache_paged,
                block_table=None,
                cache_seqlens=None,
                head_dim_v=self.kv_lora_rank,
                tile_scheduler_metadata=self._sched_meta,
                num_splits=None,
                softmax_scale=self.scale * self.softmax_extra_scale,
                causal=False,
                is_fp8_kvcache=True,
                indices=indices,
                topk_length=topk_length,
            )
        except Exception as e:
            logging.error(
                f"[MLA-FP8-DECODE] flash_mla_with_kvcache FAILED layer={layer_id}: {e}"
            )
            raise

        attn_output = attn_out.view(B, self.num_heads, self.kv_lora_rank)
        attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), v_weight)
        attn_bmm_output = attn_bmm_output.transpose(0, 1)

        return attn_bmm_output

    def _pack_fp8_prefill_q(self, q_absorbed: torch.Tensor) -> torch.Tensor:
        assert self._fp8_prefill_qo_indptr_h is not None
        assert self._fp8_prefill_q_lens_h is not None
        B = int(self._fp8_prefill_q_lens_h.size(0))
        q_padded = q_absorbed.new_zeros(
            (
                B,
                self._fp8_prefill_max_q_len,
                self.num_heads,
                self.kv_lora_rank + self.qk_rope_head_dim,
            )
        )
        for i in range(B):
            q_len = int(self._fp8_prefill_q_lens_h[i].item())
            if q_len <= 0:
                continue
            start = int(self._fp8_prefill_qo_indptr_h[i].item())
            q_padded[i, :q_len].copy_(q_absorbed[start : start + q_len])
        return q_padded

    def _unpack_fp8_prefill_out(self, attn_out: torch.Tensor) -> torch.Tensor:
        assert self._fp8_prefill_qo_indptr_h is not None
        assert self._fp8_prefill_q_lens_h is not None
        B = int(self._fp8_prefill_q_lens_h.size(0))
        attn_output = attn_out.new_empty(
            (self._fp8_prefill_total_q, self.num_heads, self.kv_lora_rank)
        )
        for i in range(B):
            q_len = int(self._fp8_prefill_q_lens_h[i].item())
            if q_len <= 0:
                continue
            start = int(self._fp8_prefill_qo_indptr_h[i].item())
            attn_output[start : start + q_len].copy_(attn_out[i, :q_len])
        return attn_output

    def _forward_fp8_prefill_absorb(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        v_weight: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        assert self._fp8_prefill_sched_meta is not None
        assert self._fp8_prefill_indices is not None

        q_absorbed = torch.cat([q_nope, q_pe], dim=-1)
        q_4d = self._pack_fp8_prefill_q(q_absorbed)

        kv_cache_u8 = kv_cache.kv_cache_base.view(torch.uint8)
        fp8_per_token = (
            self.kv_lora_rank + self.kv_lora_rank // 128 * 4 + self.qk_rope_head_dim * 2
        )
        kv_cache_paged = kv_cache_u8.view(-1, self.token_per_block, 1, fp8_per_token)

        if layer_id == 0:
            self._fp8_prefill_sched_meta.tile_scheduler_metadata = None
            self._fp8_prefill_sched_meta.num_splits = None

        try:
            attn_out, _ = flash_mla_with_kvcache(
                q=q_4d,
                k_cache=kv_cache_paged,
                block_table=None,
                cache_seqlens=None,
                head_dim_v=self.kv_lora_rank,
                tile_scheduler_metadata=self._fp8_prefill_sched_meta,
                num_splits=None,
                softmax_scale=self.scale * self.softmax_extra_scale,
                causal=False,
                is_fp8_kvcache=True,
                indices=self._fp8_prefill_indices,
                topk_length=self._fp8_prefill_topk_length,
            )
        except Exception as e:
            logging.error(
                f"[MLA-FP8-PREFILL-ABSORB] flash_mla_with_kvcache FAILED layer={layer_id}: {e}"
            )
            raise

        attn_output = self._unpack_fp8_prefill_out(
            attn_out.view(
                q_4d.shape[0],
                q_4d.shape[1],
                self.num_heads,
                self.kv_lora_rank,
            )
        )
        attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), v_weight)
        attn_bmm_output = attn_bmm_output.transpose(0, 1)

        return attn_bmm_output
