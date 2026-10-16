import logging
import os
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
from rtp_llm.utils.warmup import model_warm_up_enabled

g_workspace_buffer = None
warm_up_done = False


def warmup_flashinfer_python():
    global warm_up_done
    if not model_warm_up_enabled():
        return
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
        "kv_cache_block_id": torch.empty(0, dtype=dtype, device=device),
        "kv_cache_kernel_block_id": torch.empty(0, dtype=dtype, device=device),
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


def _use_trtllm_gen_prefill() -> bool:
    """Env gate for the trtllm-gen ragged prefill (default off, A/B form)."""
    return os.environ.get("XING4_MLA_TRTLLM_GEN_PREFILL", "0") == "1"


def _use_trtllm_gen_decode() -> bool:
    """Env gate for the trtllm-gen MLA decode (default off, A/B form)."""
    return os.environ.get("XING4_MLA_TRTLLM_GEN_DECODE", "0") == "1"


class MlaFlashInferPrefillOp(object):
    _triton_compat_warned = False  # Class variable to track warning status
    _trtllm_gen_workspace = None  # class-level lazy workspace (uint8, 128 MiB)
    _trtllm_lens_cache = None  # (key, (bs, max_q, max_kv), qo_indptr, kv_indptr
    _trtllm_run = None  # cached JIT module entry (wrapper bypass)
    _trtllm_sm_count = 0
    _trtllm_pdl = False
    _trtllm_log2e = 1.0
    _kv_b_proj_cache = None  # {(weights_id, layer_id): linear} — forward rebuilt per call

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
            # Find first layer that has MLA weights (hybrid models may have non-MLA layers)
            device = None
            for w in self.weights:
                kv_b = w.get(W.mla_kv_b_w)
                if kv_b is not None:
                    device = kv_b.device
                    break
            assert device is not None, "No MLA weights found in any layer"
            g_workspace_buffer = torch.empty(
                512 * 1024 * 1024,
                dtype=torch.int8,
                device=device,
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
        if _use_trtllm_gen_prefill():
            # trtllm-gen ragged branch captures: cum kv indptr, per-seq lens,
            # and the host scalars the kernel takes. The two .item() maxes
            # are cached by indptr identity: every layer's op plans with the
            # SAME mla_params tensors within a forward, so the syncs happen
            # once per forward, not once per layer (40x).
            self._trtllm_kv_indptr = mla_params.prefill_ragged_kv_len_indptr_d
            self._trtllm_seq_lens = self._trtllm_kv_indptr.diff()
            cache_key = (
                self.qo_indptr.data_ptr(),
                self._trtllm_kv_indptr.data_ptr(),
            )
            cached = MlaFlashInferPrefillOp._trtllm_lens_cache
            if cached is not None and cached[0] == cache_key:
                (
                    self._trtllm_batch_size,
                    self._trtllm_max_q_len,
                    self._trtllm_max_kv_len,
                ) = cached[1]
            else:
                self._trtllm_batch_size = int(self.qo_indptr.numel()) - 1
                self._trtllm_max_q_len = int(
                    torch.diff(self.qo_indptr).max().item()
                )
                self._trtllm_max_kv_len = int(self._trtllm_seq_lens.max().item())
                # Hold refs to the key tensors so data_ptr stays unique.
                MlaFlashInferPrefillOp._trtllm_lens_cache = (
                    cache_key,
                    (
                        self._trtllm_batch_size,
                        self._trtllm_max_q_len,
                        self._trtllm_max_kv_len,
                    ),
                    self.qo_indptr,
                    self._trtllm_kv_indptr,
                )

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

    def _kv_b_proj_for(self, layer_id: int):
        """Per-layer kv_b projection, constructed once and reused.

        ``forward`` used to rebuild the LinearFactory object every call
        (weight lookup + linear construction + quant resolution) x40 layers
        x every prefill — a pure host-side cost on the host-bound path.
        """
        cache = MlaFlashInferPrefillOp._kv_b_proj_cache
        if cache is None:
            cache = {}
            MlaFlashInferPrefillOp._kv_b_proj_cache = cache
        # Key on the weights-dict identity as well: the target model and the
        # MTP draft each build their own op with their own weights, and a
        # layer_id-only key would hand the draft the target's cached linear.
        key = (id(self.weights), layer_id)
        linear = cache.get(key)
        if linear is None:
            linear = LinearFactory.create_linear_from_weights(
                self.weights[layer_id],
                W.mla_kv_b_w,
                W.mla_kv_b_s,
                None,
                self.quant_config,
            )
            cache[key] = linear
        return linear

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

        kv = self._kv_b_proj_for(layer_id)(compressed_kv)
        kv = kv.view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[:, :, : self.qk_nope_head_dim]
        value_states = kv[:, :, self.qk_nope_head_dim :]
        k = self._concat_and_cast_mha_k(k_nope, k_pe)

        if _use_trtllm_gen_prefill():
            return self._trtllm_ragged_run(q, k, value_states)
        # TODO: add TRT prefill support
        attn_output = self.prefill_wrapper.run(q, k, value_states)
        attn_output = attn_output.view(-1, self.num_heads, self.v_head_dim)
        return attn_output

    def _trtllm_ragged_run(
        self, q: torch.Tensor, k: torch.Tensor, value_states: torch.Tensor
    ) -> torch.Tensor:
        """trtllm-gen ragged prefill (XING4_MLA_TRTLLM_GEN_PREFILL=1).

        Drop-in replacement for ``prefill_wrapper.run`` on the same
        decompressed MHA-form q/k/v the wrapper path builds. The bmm1 scale
        carries the same sm_scale the flashinfer plan uses (sqrt folded);
        bf16 inputs take unit quant scales (bmm2/o_sf = 1.0).
        """
        import flashinfer

        if MlaFlashInferPrefillOp._trtllm_gen_workspace is None:
            # Zero-init: the trtllm counter workspace carved from this
            # buffer must start zeroed (decode multi-block semaphores
            # device-assert on garbage).
            MlaFlashInferPrefillOp._trtllm_gen_workspace = torch.zeros(
                128 * 1024 * 1024, dtype=torch.uint8, device=q.device
            )
        ws = MlaFlashInferPrefillOp._trtllm_gen_workspace
        # Bypass the flashinfer python wrapper (its per-call device queries,
        # api-logging decorator and arg processing cost ~390us x40 layers on
        # the host-bound prefill path): cache the JIT module entry, sm count
        # and PDL flag once, and call the op directly with the same argument
        # order the wrapper assembles (bmm1 carries the wrapper's log2e
        # exp2-trick scaling).
        if MlaFlashInferPrefillOp._trtllm_run is None:
            from flashinfer.prefill import get_trtllm_gen_fmha_module
            from flashinfer.utils import (
                device_support_pdl,
                get_device_sm_count,
            )
            import math

            MlaFlashInferPrefillOp._trtllm_run = (
                get_trtllm_gen_fmha_module().trtllm_ragged_attention
            )
            MlaFlashInferPrefillOp._trtllm_sm_count = get_device_sm_count(
                q.device
            )
            MlaFlashInferPrefillOp._trtllm_pdl = device_support_pdl(q.device)
            MlaFlashInferPrefillOp._trtllm_log2e = math.log2(math.e)
        out = torch.empty(
            (q.shape[0], self.num_heads, self.v_head_dim),
            dtype=torch.bfloat16,
            device=q.device,
        )
        MlaFlashInferPrefillOp._trtllm_run(
            out,
            q,
            k,
            value_states,
            ws,
            self._trtllm_seq_lens,
            self._trtllm_max_q_len,
            self._trtllm_max_kv_len,
            self.scale * self.softmax_extra_scale
            * MlaFlashInferPrefillOp._trtllm_log2e,
            1.0,
            1.0,
            self._trtllm_batch_size,
            -1,
            self.qo_indptr,
            self._trtllm_kv_indptr,
            MlaFlashInferPrefillOp._trtllm_sm_count,
            MlaFlashInferPrefillOp._trtllm_pdl,
            True,  # is_causal
            ws.numel() * ws.element_size(),  # workspace_size
            None,  # attention_sinks
            None,  # skip_softmax_threshold_scale_factor
            None,  # lse
        )
        return out.view(-1, self.num_heads, self.v_head_dim)


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
        self.max_context_len = max_context_len
        self._t_kvlen_d = None
        self._t_page_indice_d = None
        self._t_decode_page_indptr_d = None
        self._t_bs = 0
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
            # Find first layer that has MLA weights (hybrid models may have non-MLA layers)
            device = None
            for w in self.weights:
                vc = w.get(W.mla_vc)
                if vc is not None:
                    device = vc.device
                    break
            assert device is not None, "No MLA weights found in any layer"
            g_workspace_buffer = torch.empty(
                512 * 1024 * 1024,
                dtype=torch.int8,
                device=device,
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
        if _use_trtllm_gen_decode():
            # trtllm-gen decode branch captures (refreshed every plan, so
            # graph replays read the live page table through the scatter
            # captured alongside the kernel):
            self._t_kvlen_d = fmha_params.kvlen_d
            self._t_page_indice_d = fmha_params.page_indice_d
            self._t_decode_page_indptr_d = fmha_params.decode_page_indptr_d
            self._t_bs = int(fmha_params.kvlen_h.size(0))
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
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
    ) -> torch.Tensor:

        k_weight = self.weights[layer_id].get(W.mla_kc, None)
        v_weight = self.weights[layer_id].get(W.mla_vc, None)

        compressed_kv = kv_cache.kv_cache_base.view(
            -1, self.token_per_block, self.kv_lora_rank + self.qk_rope_head_dim
        )

        q_nope = q_nope.view(-1, self.num_heads, self.qk_nope_head_dim)
        q_pe = q_pe.view(-1, self.num_heads, self.qk_rope_head_dim)

        q_nope = torch.bmm(q_nope.transpose(0, 1), k_weight)
        q_nope = q_nope.transpose(0, 1)

        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        if (
            _use_trtllm_gen_decode()
            # Plain decode only: the MTP verify capture flows bs*(1+gen)
            # rows with plan metadata that disagrees (kvlen count vs row
            # grouping); the trtllm branch scopes to tokens == requests,
            # which is exactly the 6.94ms/step attention it targets.
            and q_nope.shape[0] == int(self._t_kvlen_d.size(0))
            # Large-batch regime only: the kernel halves the attention time
            # (3.69 vs 6.94 ms/step at bs64) but the aten scatter machinery
            # costs ~1.6ms of extra launches, which loses below ~bs32.
            and q_nope.shape[0] >= 32
        ):
            attn_output = self._trtllm_decode_run(q_nope, q_pe, kv_cache)
        else:
            attn_output = torch.empty_like(q_nope)
            self.mla_wrapper.run(q_nope, q_pe, compressed_kv, k_pe, attn_output)

        attn_output = attn_output.view(-1, self.num_heads, self.kv_lora_rank)
        attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), v_weight)
        attn_bmm_output = attn_bmm_output.transpose(0, 1)

        return attn_bmm_output

    def _trtllm_decode_run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
    ) -> torch.Tensor:
        """trtllm-gen MLA decode (XING4_MLA_TRTLLM_GEN_DECODE=1).

        Microbench-resolved contract (fusion_workflow/p2_2 plan): query is
        the pre-absorbed latent [bs, 1, H, 576] (nope part = q_nope @ w_kc,
        already computed by the caller); kv_cache is the SAME paged latent
        view forward already builds ([pages, block, 576]); output is the
        latent [bs, 1, H, 512] the caller's w_vc bmm consumes. The padded
        block table is scattered from the flat page table + per-request
        indptr every step so graph replays read the live page ids (the C1
        refresh pattern).
        """
        import flashinfer

        # Live read: fill_params resizes the SAME kvlen tensor each plan, so
        # size(0) is the current request count (a plan-time python int went
        # stale by one cycle during the MTP verify capture: 124 rows with a
        # recorded bs of 32 instead of the live 31).
        bs = int(self._t_kvlen_d.size(0))
        flat_pages = self._t_page_indice_d
        indptr = self._t_decode_page_indptr_d
        total_pages = int(flat_pages.numel())
        # The kernel walks block_table columns up to max_seq_len/block_size
        # regardless of each request's live seq_len (the max_seq_len we
        # pass is the construction bound), so the table must be padded to
        # that width — a total_pages-wide table device-asserts on the
        # out-of-bounds column reads.
        table_width = (
            self.max_context_len + self.token_per_block - 1
        ) // self.token_per_block
        padded = torch.zeros(
            (bs, table_width), dtype=torch.int32, device=flat_pages.device
        )
        # Graph-safe scatter: searchsorted (fixed output shape, no host
        # sync) maps each flat page to its request; repeat_interleave would
        # sync on the device counts and abort stream capture.
        col_flat = torch.arange(
            total_pages, device=flat_pages.device, dtype=torch.int64
        )
        row_ids = torch.searchsorted(
            indptr[1:].to(torch.int64), col_flat, right=True
        )
        col_ids = col_flat - indptr.to(torch.int64)[row_ids]
        # The flat page table's numel can lag the indptr coverage by one
        # plan cycle (both are resized shared buffers): clamp stale tail
        # entries into the last row's unused table region with value 0 —
        # deterministic shapes stay graph-safe and live entries untouched.
        row_c = row_ids.clamp(max=bs - 1)
        col_c = col_ids.clamp(min=0, max=table_width - 1)
        vals = torch.where(
            row_ids < bs, flat_pages, torch.zeros_like(flat_pages)
        )
        padded[row_c, col_c] = vals

        full_kv = kv_cache.kv_cache_base.view(
            -1, self.token_per_block, self.kv_lora_rank + self.qk_rope_head_dim
        )
        # MTP verify rows flow through this op as bs x (1 + gen) tokens; the
        # API takes them natively as q_len_per_request.
        query = torch.cat([q_nope, q_pe], dim=-1).reshape(
            bs, 1, self.num_heads, -1
        )
        if MlaFlashInferPrefillOp._trtllm_gen_workspace is None:
            MlaFlashInferPrefillOp._trtllm_gen_workspace = torch.zeros(
                128 * 1024 * 1024, dtype=torch.uint8, device=query.device
            )
        out = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
            query,
            full_kv,
            MlaFlashInferPrefillOp._trtllm_gen_workspace,
            self.qk_nope_head_dim,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            padded,
            self._t_kvlen_d,
            self.max_context_len,
            bmm1_scale=self.scale * self.softmax_extra_scale,
            bmm2_scale=1.0,
        )
        return out.reshape(-1, self.num_heads, self.kv_lora_rank)

