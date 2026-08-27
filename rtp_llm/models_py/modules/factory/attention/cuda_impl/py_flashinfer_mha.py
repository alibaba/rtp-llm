from typing import Any, Optional

import torch
from flashinfer.cascade import merge_state_in_place
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
    MhaRotaryEmbeddingOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op import (
    KVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import (
    check_attention_inputs,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.utils.arch import is_sm10x, is_sm90
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, ParallelismConfig, RopeStyle
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    LayerKVCache,
    ParamsBase,
    PyAttentionInputs,
    fill_mla_params,
    rtp_llm_ops,
)

# Constants
DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_MB = 128

# FP8 KV cache uses a unit quantization scale: K/V are cast
# directly to float8_e4m3fn and FA3 FP8 kernels run with scale_q/k/v = 1.0.
FP8_UNIT_SCALE = 1.0
_g_fp8_unit_scale_tensors: dict[torch.device, torch.Tensor] = {}


def _get_fp8_unit_scale_tensor(device: torch.device) -> torch.Tensor:
    scale = _g_fp8_unit_scale_tensors.get(device)
    if scale is None:
        scale = torch.tensor([FP8_UNIT_SCALE], dtype=torch.float32, device=device)
        _g_fp8_unit_scale_tensors[device] = scale
    return scale


def quantize_to_fp8_if_needed(
    tensor: torch.Tensor, target_dtype: torch.dtype
) -> torch.Tensor:
    """Return a matching tensor or quantize it to the supported FP8 dtype."""
    if tensor.dtype == target_dtype:
        return tensor
    if target_dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"unsupported dtype conversion from {tensor.dtype} to {target_dtype}; "
            "only quantization to torch.float8_e4m3fn is supported"
        )

    # per_tensor_quant_fp8 requires contiguous input; packed-QKV split
    # views are strided, so materialize them first.
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    output = torch.empty_like(tensor, dtype=target_dtype)
    rtp_llm_ops.per_tensor_quant_fp8(
        tensor, output, _get_fp8_unit_scale_tensor(tensor.device), True
    )
    return output


# Global workspace buffer pool
_g_py_flashinfer_workspace_pool: list[torch.Tensor] = []
_g_py_flashinfer_pool_lock = __import__("threading").Lock()


def get_py_flashinfer_workspace_buffer(device: str = "cuda") -> torch.Tensor:
    """Get a PyFlashInfer workspace buffer from the pool.

    This function manages workspace buffers to support multiple concurrent instances.
    """
    with _g_py_flashinfer_pool_lock:
        if _g_py_flashinfer_workspace_pool:
            return _g_py_flashinfer_workspace_pool.pop()
    return torch.zeros(
        DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_MB * 1024 * 1024,
        dtype=torch.uint8,
        device=device,
    )


def release_py_flashinfer_workspace_buffer(buffer: torch.Tensor) -> None:
    """Release a PyFlashInfer workspace buffer back to the pool."""
    with _g_py_flashinfer_pool_lock:
        _g_py_flashinfer_workspace_pool.append(buffer)


def _host_i32(t):
    """Lift a possibly CUDA-resident tensor to host for the host fill path.

    The C++ fillParams reads raw host pointers; hybrid-attention models
    (qwen3-next) and the MTP device-state fast path can hand in CUDA-resident
    lengths/block tables even when input_lengths stays on the host.
    """
    return t.cpu() if t is not None and t.numel() > 0 and t.is_cuda else t


def _device_or(device_tensor, host_tensor):
    """Prefer the *_device mirror; fall back to the base field.

    Unit tests (and some callers) construct PyAttentionInputs with only the
    base fields populated (possibly already CUDA-resident), so the device
    mirror may be missing.
    """
    if device_tensor is not None and device_tensor.numel() >= 0:
        return device_tensor
    return host_tensor


def attn_kv_dtype(attn_configs: AttentionConfigs) -> torch.dtype:
    # Use one dtype source for both plan() and forward().
    if attn_configs.kv_cache_dtype == KvCacheDataType.FP8:
        return torch.float8_e4m3fn
    return attn_configs.dtype


def attn_q_dtype(attn_configs: AttentionConfigs) -> torch.dtype:
    # FA3 FP8 (Hopper wgmma) requires Q/KV in the same FP8 dtype and is
    # SM90-only with head_dim 64/128/256;
    # Otherwise keep Q in fp16/bf16 (FA2 KV-dequant path)
    # Q uses the same unit-scale FP8 contract as the KV cache on this path.
    if (
        attn_configs.kv_cache_dtype == KvCacheDataType.FP8
        and is_sm90()
        and attn_configs.size_per_head in (64, 128, 256)
    ):
        return torch.float8_e4m3fn
    return attn_configs.dtype


class PyFlashinferPrefillPagedAttnOp(object):
    """FlashInfer Prefill Attention Op with Paged KV Cache support"""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        backend: str = "auto",
    ) -> None:
        self.g_workspace_buffer = get_py_flashinfer_workspace_buffer()
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        self.head_dim_vo = attn_configs.size_per_head
        self.page_size = attn_configs.kernel_tokens_per_block
        self.dtype = attn_configs.dtype
        self.kv_dtype = attn_kv_dtype(attn_configs)
        self.q_dtype = attn_q_dtype(attn_configs)
        self.max_seq_len = attn_configs.max_seq_len
        self.is_causal = attn_configs.is_causal
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.enable_cuda_graph = attn_inputs.is_cuda_graph
        self.prefill_cuda_graph_copy_params = None
        # Pre-allocated buffers for CUDA graph copy path (avoid per-forward allocation)
        self._aligned_q_buf = None
        # reserve buffer for q cast
        self._aligned_q_cast_buf = None
        self._compact_out_buf = None
        # Use Paged KV Cache wrapper
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.g_workspace_buffer,
            "HND",
            backend=backend,
        )

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def set_params(self, params: rtp_llm_ops.FlashInferMlaAttnParams):
        """Set the params object to be used by this op."""
        self.fmha_params = params

    def prepare(
        self,
        attn_inputs: PyAttentionInputs,
        forbid_realloc: bool = False,
    ) -> ParamsBase:
        """
        Prepare the prefill wrapper with paged KV cache parameters.

        forbid_realloc: True only when called from prepare_cuda_graph (replay); forbids buffer realloc.
        """
        check_attention_inputs(attn_inputs)
        block_id_host = attn_inputs.kv_cache_kernel_block_id
        if block_id_host is None or block_id_host.numel() == 0:
            block_id_host = attn_inputs.kv_cache_kernel_block_id_device
        graph_copy_params = (
            self.prefill_cuda_graph_copy_params
            if self.prefill_cuda_graph_copy_params is not None
            else attn_inputs.prefill_cuda_graph_copy_params
        )
        planned_batch_size = (
            graph_copy_params.max_batch_size if graph_copy_params is not None else -1
        )
        # Keep the same fill path for capture and replay: the host fill sizes
        # buffers exactly while the device fill sizes for the worst case, so
        # switching paths between capture and replay forces a (forbidden)
        # reallocation during graph replay.
        if attn_inputs.input_lengths.is_cuda:
            self.fmha_params.fill_params_mha_device(
                _device_or(
                    attn_inputs.prefix_lengths_device, attn_inputs.prefix_lengths
                ),
                attn_inputs.sequence_lengths,
                _device_or(attn_inputs.input_lengths_device, attn_inputs.input_lengths),
                _device_or(
                    attn_inputs.kv_cache_kernel_block_id_device,
                    attn_inputs.kv_cache_kernel_block_id,
                ),
                self.page_size,
                forbid_realloc,
                planned_batch_size,
            )
        else:
            self.fmha_params.fill_params(
                _host_i32(attn_inputs.prefix_lengths),
                _host_i32(attn_inputs.sequence_lengths),
                _host_i32(attn_inputs.input_lengths),
                _host_i32(block_id_host),
                self.page_size,
                forbid_realloc,
                planned_batch_size,
            )
        # Store CUDA graph copy parameters
        # Define qo_indptr early for CUDA graph initialization
        if attn_inputs.prefill_cuda_graph_copy_params is not None:
            # For CUDA graph mode, create a buffer that will be filled later
            self.input_lengths = attn_inputs.input_lengths
            self.cu_seq_lens = attn_inputs.cu_seqlens_device
            qo_indptr = attn_inputs.cu_seqlens_device.clone()
        else:
            qo_indptr = attn_inputs.cu_seqlens_device[
                : attn_inputs.input_lengths.size(0) + 1
            ]

        if self.enable_cuda_graph and self.prefill_wrapper._qo_indptr_buf is None:
            self.prefill_wrapper._use_cuda_graph = True
            self.prefill_wrapper._qo_indptr_buf = qo_indptr
            self.prefill_wrapper._paged_kv_indptr_buf = (
                self.fmha_params.decode_page_indptr_d
            )
            self.prefill_wrapper._paged_kv_last_page_len_buf = (
                self.fmha_params.paged_kv_last_page_len_d
            )
            self.prefill_wrapper._paged_kv_indices_buf = self.fmha_params.page_indice_d
            self.prefill_wrapper._fixed_batch_size = (
                len(attn_inputs.cu_seqlens_device) - 1
            )
            if attn_inputs.prefill_cuda_graph_copy_params is not None:
                self.prefill_cuda_graph_copy_params = (
                    attn_inputs.prefill_cuda_graph_copy_params
                )
                # input_lengths and cu_seq_lens were already set above
                self.qo_indptr = qo_indptr
                # Fill with cumulative sequence: [0, max_seq_len, 2*max_seq_len, ...]
                self.qo_indptr.copy_(
                    torch.arange(
                        self.qo_indptr.size(0),
                        device=self.qo_indptr.device,
                        dtype=self.qo_indptr.dtype,
                    )
                    * self.prefill_cuda_graph_copy_params.max_seq_len
                )

        # Update buffers for subsequent calls if in CUDA graph mode
        if self.prefill_cuda_graph_copy_params is not None:
            assert attn_inputs.prefill_cuda_graph_copy_params is not None
            assert self.input_lengths is not None
            assert self.cu_seq_lens is not None
            self.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size[0] = (
                attn_inputs.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size
            )
            self.input_lengths[: attn_inputs.input_lengths.size(0)] = (
                attn_inputs.input_lengths
            )
            self.cu_seq_lens[: attn_inputs.cu_seqlens_device.size(0)] = (
                attn_inputs.cu_seqlens_device
            )
            qo_indptr = self.qo_indptr

        self.prefill_wrapper.plan(
            qo_indptr,
            self.fmha_params.decode_page_indptr_d,
            self.fmha_params.page_indice_d,
            self.fmha_params.paged_kv_last_page_len_d,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.page_size,
            causal=self.is_causal,
            q_data_type=self.q_dtype,
            kv_data_type=self.kv_dtype,
            o_data_type=self.dtype,
        )
        return self.fmha_params

    @staticmethod
    def support(attn_inputs: PyAttentionInputs) -> bool:
        return True

    def forward(
        self, q: torch.Tensor, kv_cache: Optional[LayerKVCache]
    ) -> torch.Tensor:
        """
        Forward pass with paged KV cache

        Args:
            q: Query tensor [total_tokens, num_heads, head_dim]
            kv_cache: Paged KV cache [num_pages, 2, page_size, kv_heads, head_dim]
            params: Parameters (not used currently)

        Returns:
            output: [total_tokens, num_heads, head_dim]
        """
        from rtp_llm.ops.compute_ops import (
            cuda_graph_copy_large2small,
            cuda_graph_copy_small2large,
        )

        assert kv_cache is not None, "kv_cache is required for paged attention"
        assert (
            q.dim() == 3
        ), f"Expected q to be 3D tensor [total_tokens, num_heads, head_dim], got {q.dim()}D"

        paged_kv_cache = kv_cache.kv_cache_base
        if paged_kv_cache.dim() == 2:
            paged_kv_cache = common.reshape_paged_kv_cache(
                paged_kv_cache, self.local_kv_head_num, self.page_size, self.head_dim_qk
            )
        # CUDA graph copy logic for prefill
        if self.prefill_cuda_graph_copy_params:
            assert (
                self.input_lengths is not None
            ), "input_lengths is required for CUDA graph copy"
            assert (
                self.cu_seq_lens is not None
            ), "cu_seq_lens is required for CUDA graph copy"

            # Reshape from 3D [token_num, head_num, head_size] to 2D [token_num, hidden_size]
            token_num, head_num, head_size = q.shape
            hidden_size = head_num * head_size

            # Pre-allocate buffers on first use (avoid per-forward GPU allocation)
            total_len = (
                self.prefill_cuda_graph_copy_params.max_seq_len
                * self.prefill_cuda_graph_copy_params.max_batch_size
            )
            if self._aligned_q_buf is None or self._aligned_q_buf.shape != (
                total_len,
                hidden_size,
            ):
                self._aligned_q_buf = torch.zeros(
                    (total_len, hidden_size), dtype=q.dtype, device=q.device
                )
            if self._compact_out_buf is None or self._compact_out_buf.shape != (
                token_num,
                hidden_size,
            ):
                self._compact_out_buf = torch.zeros(
                    (token_num, hidden_size), dtype=q.dtype, device=q.device
                )

            q_2d = q.view(token_num, hidden_size).contiguous()
            self._aligned_q_buf.zero_()

            # Copy small to large (compact -> aligned)
            cuda_graph_copy_small2large(
                q_2d,
                self._aligned_q_buf,
                self.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size,
                self.prefill_cuda_graph_copy_params.max_batch_size,
                self.prefill_cuda_graph_copy_params.max_seq_len,
                self.input_lengths,
                hidden_size,
                self.cu_seq_lens,
            )

            # Reshape back to 3D for FlashInfer
            q_aligned = self._aligned_q_buf.view(total_len, head_num, head_size)
            if q_aligned.dtype != self.q_dtype:
                if (
                    self._aligned_q_cast_buf is None
                    or self._aligned_q_cast_buf.shape != q_aligned.shape
                    or self._aligned_q_cast_buf.dtype != self.q_dtype
                    or self._aligned_q_cast_buf.device != q_aligned.device
                ):
                    self._aligned_q_cast_buf = torch.empty(
                        q_aligned.shape,
                        dtype=self.q_dtype,
                        device=q_aligned.device,
                    )
                rtp_llm_ops.per_tensor_quant_fp8(
                    q_aligned,
                    self._aligned_q_cast_buf,
                    _get_fp8_unit_scale_tensor(q_aligned.device),
                    True,
                )
                q_aligned = self._aligned_q_cast_buf

            # Paged FP8 defaults to unit scales and the output dtype from plan().
            result = self.prefill_wrapper.run(q_aligned, paged_kv_cache)

            # Reshape result to 2D for copy back (ensure contiguous)
            result_2d = result.view(total_len, hidden_size).contiguous()
            self._compact_out_buf.zero_()

            # Copy large to small (aligned -> compact)
            cuda_graph_copy_large2small(
                result_2d,
                self._compact_out_buf,
                self.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size,
                self.prefill_cuda_graph_copy_params.max_batch_size,
                self.prefill_cuda_graph_copy_params.max_seq_len,
                self.input_lengths,
                hidden_size,
                self.cu_seq_lens,
            )

            # Reshape back to 3D
            result = self._compact_out_buf.view(token_num, head_num, head_size)
        else:
            # No CUDA graph copy, direct execution
            # Paged FP8 defaults to unit scales and the output dtype from plan().
            result = self.prefill_wrapper.run(
                quantize_to_fp8_if_needed(q, self.q_dtype), paged_kv_cache
            )

        return result


class PyFlashinferPrefillAttnOp(object):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        backend: str = "auto",
    ) -> None:
        self.g_workspace_buffer = get_py_flashinfer_workspace_buffer()
        # attn_configs.head_num and kv_head_num are already divided by tp_size in ModelConfig::getAttentionConfigs
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        self.page_size = attn_configs.kernel_tokens_per_block
        # TODO: maybe use v_head_dim
        self.head_dim_vo = attn_configs.size_per_head
        self.prefill_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self.g_workspace_buffer,
            backend=backend,
        )
        self.dtype = attn_configs.dtype
        self.q_dtype = attn_q_dtype(attn_configs)
        self.kv_dtype = attn_kv_dtype(attn_configs)
        self.is_causal = attn_configs.is_causal
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def set_params(self, params: rtp_llm_ops.FlashInferMlaAttnParams):
        """Set the params object to be used by this op."""
        self.fmha_params = params

    def prepare(self, attn_inputs: PyAttentionInputs) -> ParamsBase:
        """
        Prepare the prefill wrapper

        Args:
            attn_inputs: Attention inputs containing sequence information
        """
        batch_size = attn_inputs.input_lengths.size(0)
        cu_seqlens = attn_inputs.cu_seqlens_device[: batch_size + 1]

        # Encoder-only models (BERT) have no paged kv cache; fill_params
        # pybind requires a Tensor, so substitute an empty int32 tensor.
        kv_block_id = attn_inputs.kv_cache_kernel_block_id
        if kv_block_id is None or kv_block_id.numel() == 0:
            kv_block_id = attn_inputs.kv_cache_kernel_block_id_device
        if kv_block_id is None:
            kv_block_id = torch.empty(0, dtype=torch.int32)

        self.fmha_params.fill_params(
            _host_i32(attn_inputs.prefix_lengths),
            _host_i32(attn_inputs.sequence_lengths),
            _host_i32(attn_inputs.input_lengths),
            _host_i32(kv_block_id),
            self.page_size,
        )

        self.prefill_wrapper.plan(
            cu_seqlens,
            cu_seqlens,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.head_dim_vo,
            causal=self.is_causal,
            q_data_type=self.q_dtype,
            kv_data_type=self.kv_dtype,
            o_data_type=self.dtype,
        )
        return self.fmha_params

    @staticmethod
    def support(attn_inputs: PyAttentionInputs) -> bool:
        return (
            attn_inputs.prefix_lengths.numel() <= 0
            or attn_inputs.prefix_lengths.sum().item() == 0
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        q = quantize_to_fp8_if_needed(q, self.q_dtype)
        k = quantize_to_fp8_if_needed(k, self.kv_dtype)
        v = quantize_to_fp8_if_needed(v, self.kv_dtype)
        if q.dtype == torch.float8_e4m3fn:
            # FlashInfer's FA3 FP8 need scale_q, scale_k, scale_v and an output matching the planned dtype.
            out = torch.empty(
                q.shape[:-1] + v.shape[-1:], dtype=self.dtype, device=q.device
            )
            return self.prefill_wrapper.run(
                q,
                k,
                v,
                FP8_UNIT_SCALE,
                FP8_UNIT_SCALE,
                FP8_UNIT_SCALE,
                out=out,
            )
        return self.prefill_wrapper.run(q, k, v)


class PyFlashinferHybridPrefillAttnOp(object):
    """FlashInfer hybrid prefill op.

    It evaluates attention over the newest ragged KV segment and the existing
    prefix paged KV cache sequentially on the current CUDA stream, then merges
    the two attention states with FlashInfer's LSE merge.
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        backend: str = "auto",
    ) -> None:
        self.g_workspace_buffer = get_py_flashinfer_workspace_buffer()
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        self.head_dim_vo = attn_configs.size_per_head
        self.page_size = attn_configs.kernel_tokens_per_block
        self.dtype = attn_configs.dtype
        self.kv_dtype = attn_kv_dtype(attn_configs)
        self.q_dtype = attn_q_dtype(attn_configs)
        self.is_causal = attn_configs.is_causal
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        # The serial ragged/write/paged flow can share one workspace buffer.
        self.ragged_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self.g_workspace_buffer,
            backend=backend,
        )
        self.prefix_paged_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.g_workspace_buffer,
            "HND",
            backend=backend,
        )

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def set_params(self, params: rtp_llm_ops.FlashInferMlaAttnParams):
        """Set the params object to be used by this op."""
        self.fmha_params = params

    def prepare(
        self,
        attn_inputs: PyAttentionInputs,
        forbid_realloc: bool = False,
    ) -> ParamsBase:
        """Prepare ragged-new and paged-prefix FlashInfer wrappers."""
        block_table = attn_inputs.kv_cache_kernel_block_id
        if block_table is None or block_table.numel() == 0:
            block_table = attn_inputs.kv_cache_kernel_block_id_device
        assert (
            block_table is not None and block_table.numel() > 0
        ), "hybrid prefill requires a non-empty kv_cache_kernel_block_id"
        self.fmha_params.fill_params(
            _host_i32(attn_inputs.prefix_lengths),
            _host_i32(attn_inputs.sequence_lengths),
            _host_i32(attn_inputs.input_lengths),
            _host_i32(block_table),
            self.page_size,
            forbid_realloc,
        )

        batch_size = attn_inputs.input_lengths.size(0)
        qo_indptr = attn_inputs.cu_seqlens_device[: batch_size + 1]

        self.ragged_wrapper.plan(
            qo_indptr,
            qo_indptr,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.head_dim_vo,
            causal=self.is_causal,
            q_data_type=self.q_dtype,
            kv_data_type=self.kv_dtype,
            o_data_type=self.dtype,
        )

        # batch_reuse_info_vec_h columns are defined in FlashInferMlaParams.cc.
        prefix_len_col = 1
        page_start_col = 2
        reuse_info = self.fmha_params.batch_reuse_info_vec_h
        prefix_lengths = reuse_info[:, prefix_len_col]
        if (prefix_lengths <= 0).any().item():
            raise ValueError(
                "hybrid prefill requires a non-empty prefix cache per batch item"
            )

        prefix_paged_kv_indptr = torch.empty(
            batch_size + 1, dtype=torch.int32, device="cpu"
        )
        prefix_paged_kv_indptr[:-1].copy_(reuse_info[:, page_start_col])
        prefix_paged_kv_indptr[-1] = self.fmha_params.reuse_cache_page_indice_h.numel()
        prefix_paged_kv_last_page_len = (prefix_lengths - 1) % self.page_size + 1

        self.prefix_paged_wrapper.plan(
            qo_indptr,
            prefix_paged_kv_indptr,
            self.fmha_params.reuse_cache_page_indice_d,
            prefix_paged_kv_last_page_len,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.page_size,
            causal=False,
            q_data_type=self.q_dtype,
            kv_data_type=self.kv_dtype,
            o_data_type=self.dtype,
        )
        return self.fmha_params

    @staticmethod
    def support(attn_inputs: PyAttentionInputs) -> bool:
        block_table = attn_inputs.kv_cache_kernel_block_id
        if block_table is None or block_table.numel() == 0:
            block_table = attn_inputs.kv_cache_kernel_block_id_device
        prefix_lengths = attn_inputs.prefix_lengths
        return (
            block_table is not None
            and block_table.numel() > 0
            and prefix_lengths is not None
            and prefix_lengths.numel() > 0
            and prefix_lengths.min().item() > 0
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        kv_cache_write_op: Optional[KVCacheWriteOp] = None,
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required for hybrid prefill"
        paged_kv_cache = kv_cache.kv_cache_base
        if paged_kv_cache.dim() == 2:
            paged_kv_cache = common.reshape_paged_kv_cache(
                paged_kv_cache, self.local_kv_head_num, self.page_size, self.head_dim_qk
            )

        q = quantize_to_fp8_if_needed(q, self.q_dtype)
        k = quantize_to_fp8_if_needed(k, self.kv_dtype)
        v = quantize_to_fp8_if_needed(v, self.kv_dtype)
        if q.dtype == torch.float8_e4m3fn:
            # Positional scale_q/scale_k/scale_v; see FP8_UNIT_SCALE.
            out = torch.empty(
                q.shape[:-1] + v.shape[-1:], dtype=self.dtype, device=q.device
            )
            new_out, new_lse = self.ragged_wrapper.run(
                q,
                k,
                v,
                FP8_UNIT_SCALE,
                FP8_UNIT_SCALE,
                FP8_UNIT_SCALE,
                out=out,
                return_lse=True,
            )
        else:
            new_out, new_lse = self.ragged_wrapper.run(q, k, v, return_lse=True)

        if kv_cache_write_op is not None:
            kv_cache_write_op.forward(k, v, kv_cache)

        # Paged FP8 defaults to unit scales and the output dtype from plan().
        prefix_out, prefix_lse = self.prefix_paged_wrapper.run(
            q, paged_kv_cache, return_lse=True
        )
        merge_state_in_place(new_out, new_lse, prefix_out, prefix_lse)
        return new_out


class PyFlashinferPrefillImplBase(FMHAImplBase):
    """Base class for FlashInfer prefill implementations (Ragged and Paged)."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        """Initialize prefill implementation with common setup.

        Args:
            attn_configs: Attention configuration
            attn_inputs: Attention inputs
        """
        # Store configs and inputs
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs

        self.fmha_impl = self._create_fmha_impl(attn_configs, attn_inputs)
        self.rope_impl = self._create_rope_impl(attn_configs)
        # Create KV cache write op
        self.kv_cache_write_op = KVCacheWriteOp(
            num_kv_heads=attn_configs.kv_head_num,
            head_size=attn_configs.size_per_head,
            token_per_block=attn_configs.kernel_tokens_per_block,
        )
        self.create_params(attn_inputs)
        self.fmha_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        self.fmha_impl.prepare(attn_inputs, forbid_realloc=True)

    def create_params(self, attn_inputs: PyAttentionInputs):
        """Create FlashInfer MLA attention parameters.

        Similar to MLA implementation, this creates and initializes the params
        that will be used for both FMHA and RoPE operations.
        """
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.rope_params = self.fmha_params
        # Pass the shared params to all ops
        self.fmha_impl.set_params(self.fmha_params)
        if self.rope_impl is not None:
            self.rope_impl.set_params(self.rope_params)
        # KV cache write always needs params (even without RoPE)
        self.kv_cache_write_op.set_params(self.rope_params)

    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        """Create FMHA implementation. To be overridden by subclasses."""
        raise NotImplementedError("Subclass must implement _create_fmha_impl")

    def _create_rope_impl(self, attn_configs: AttentionConfigs) -> Any:
        """Create RoPE implementation. To be overridden by subclasses."""
        raise NotImplementedError("Subclass must implement _create_rope_impl")

    def _split_qkv(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split QKV tensor into query, key, value.

        Args:
            qkv: QKV tensor [total_tokens, (num_heads + 2*num_kv_heads) * head_dim]

        Returns:
            Tuple of (query, key, value) tensors
        """
        qkv = qkv.reshape(qkv.shape[0], -1)
        num_heads = self.attn_configs.head_num
        num_kv_heads = self.attn_configs.kv_head_num
        head_dim = self.attn_configs.size_per_head

        q, k, v = torch.split(
            qkv,
            [
                head_dim * num_heads,
                head_dim * num_kv_heads,
                head_dim * num_kv_heads,
            ],
            dim=-1,
        )

        query = q.reshape(q.shape[0], num_heads, head_dim)
        key = k.reshape(k.shape[0], num_kv_heads, head_dim)
        value = v.reshape(v.shape[0], num_kv_heads, head_dim)

        return query, key, value

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Common forward implementation for all prefill implementations."""
        if self.need_rope_kv_cache and self.rope_impl is not None:
            query, key, value = self.rope_impl.forward(qkv)
        else:
            query, key, value = self._split_qkv(qkv)

        # Cast K/V once so the KV cache write and the attention op share
        # the same tensors.
        kv_dtype = attn_kv_dtype(self.attn_configs)
        key = quantize_to_fp8_if_needed(key, kv_dtype)
        value = quantize_to_fp8_if_needed(value, kv_dtype)

        if self.need_rope_kv_cache:
            self.kv_cache_write_op.forward(key, value, kv_cache)

        fmha_inputs = self._prepare_fmha_input(query, key, value)

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(*fmha_inputs, kv_cache)

    def _prepare_fmha_input(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Positional args for fmha_impl.forward; kv_cache is appended by forward().

        Default: only query (paged layout, KV is read from the cache).
        """
        return (query,)


class PyFlashinferPagedPrefillImpl(PyFlashinferPrefillImplBase):
    """FlashInfer prefill implementation with paged KV cache layout using MhaRotaryEmbeddingOp."""

    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        """Create paged FMHA implementation."""
        return PyFlashinferPrefillPagedAttnOp(attn_configs, attn_inputs)

    def _create_rope_impl(self, attn_configs: AttentionConfigs) -> Any:
        """Create RoPE implementation for paged layout."""
        if attn_configs.rope_config.style == RopeStyle.No:
            return None
        return MhaRotaryEmbeddingOp(attn_configs)

    def _prepare_fmha_input(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """For paged layout, only pass query (KV is already in cache)."""
        return (query,)

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        """Check if paged prefill implementation is supported.

        Returns True if:
        1. Not running on SM10x datacenter Blackwell, where TRTLLMGen is preferred.
           SM12x consumer Blackwell keeps this FlashInfer paged fallback because
           TRTLLMGen/XQA do not have sm_120a support in this build.
        2. The underlying paged FMHA op supports the inputs
        3. MhaRotaryEmbeddingOp supports the inputs
        """
        return (
            not is_sm10x()
            and PyFlashinferPrefillPagedAttnOp.support(attn_inputs)
            and attn_configs.rope_config.style != RopeStyle.Mrope
        )

    def support_cuda_graph(self) -> bool:
        return True


class PyFlashinferHybridPrefillImpl(PyFlashinferPrefillImplBase):
    """FlashInfer hybrid prefill implementation.

    The current qkv chunk first attends to the new ragged KV, appends that KV to
    the cache, and then attends to the existing prefix through its prefix-only
    page table. The two attention states are merged via LSE.
    """

    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        """Create hybrid FMHA implementation."""
        return PyFlashinferHybridPrefillAttnOp(attn_configs, attn_inputs)

    def _create_rope_impl(self, attn_configs: AttentionConfigs) -> Any:
        """Create RoPE implementation for hybrid layout."""
        if attn_configs.rope_config.style == RopeStyle.No:
            return None
        return MhaRotaryEmbeddingOp(attn_configs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Run ragged attention, append KV, then run paged prefix attention."""
        # Single-stream flow: RoPE -> ragged attention -> KV write -> paged attention.
        # Hybrid always needs the new K/V for its ragged half.
        if self.need_rope_kv_cache and self.rope_impl is not None:
            query, key, value = self.rope_impl.forward(qkv)
        else:
            query, key, value = self._split_qkv(qkv)

        query = quantize_to_fp8_if_needed(query, attn_q_dtype(self.attn_configs))
        kv_dtype = attn_kv_dtype(self.attn_configs)
        key = quantize_to_fp8_if_needed(key, kv_dtype)
        value = quantize_to_fp8_if_needed(value, kv_dtype)

        # Write new K/V after ragged attention and before paged attention.
        kv_cache_write_op = self.kv_cache_write_op if self.need_rope_kv_cache else None
        result = self.fmha_impl.forward(
            query,
            key,
            value,
            kv_cache,
            kv_cache_write_op,
        )
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return result

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        """Check if hybrid prefill implementation is supported."""
        return (
            not attn_inputs.is_cuda_graph
            and not is_sm10x()
            and PyFlashinferHybridPrefillAttnOp.support(attn_inputs)
            and attn_configs.rope_config.style != RopeStyle.Mrope
        )

    def support_cuda_graph(self) -> bool:
        return False


class PyFlashinferPrefillImpl(PyFlashinferPrefillImplBase):
    """FlashInfer prefill implementation with ragged KV cache layout using MhaRotaryEmbeddingOp."""

    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        """Create ragged FMHA implementation."""
        return PyFlashinferPrefillAttnOp(attn_configs)

    def _create_rope_impl(self, attn_configs: AttentionConfigs) -> Any:
        """Create RoPE implementation for ragged layout."""
        if attn_configs.rope_config.style == RopeStyle.No:
            return None
        return MhaRotaryEmbeddingOp(attn_configs)

    def _prepare_fmha_input(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """For ragged layout, pass Q/K/V directly to the ragged wrapper."""
        return query, key, value

    def support_cuda_graph(self) -> bool:
        return False

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        """Check if ragged prefill implementation is supported.

        Returns True if:
        1. The underlying ragged FMHA op supports the inputs
           (requires prefix_lengths to be empty or zero)
        2. MhaRotaryEmbeddingOp supports the inputs
        3. Mrope is not used

        Note: Unlike the paged variant, ragged prefill is kept enabled on
        Blackwell: TRT-LLM Gen prefill requires a paged kv cache and
        therefore does not cover BERT-style encoder-only inputs that lack
        one. Without this fallback, sm_120 has no usable prefill impl for
        such cases.
        """
        return (
            PyFlashinferPrefillAttnOp.support(attn_inputs)
            and attn_configs.rope_config.style != RopeStyle.Mrope
        )


def determine_use_tensor_core_from_configs(attn_configs: AttentionConfigs) -> bool:
    """Determine whether to use tensor cores based on attention configs."""
    # Use tensor cores for larger head dimensions and when kv_head_num matches requirements
    return attn_configs.head_num // attn_configs.kv_head_num >= 4


class PyFlashinferDecodeAttnOp(object):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
    ) -> None:
        self.g_workspace_buffer = get_py_flashinfer_workspace_buffer()
        # attn_configs already has head_num and kv_head_num divided by tp_size
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        self.head_dim_vo = attn_configs.size_per_head
        self.seq_size_per_block = attn_configs.kernel_tokens_per_block
        self.use_tensor_core = determine_use_tensor_core_from_configs(attn_configs)
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.g_workspace_buffer,
            "HND",
            use_tensor_cores=self.use_tensor_core,
        )
        self.dtype = attn_configs.dtype
        self.kv_dtype = attn_kv_dtype(attn_configs)
        # CUDA-core decode dequantizes FP8 KV; tensor-core decode uses the
        # batch-prefill path and therefore shares attn_q_dtype().
        self.q_dtype = (
            attn_q_dtype(attn_configs) if self.use_tensor_core else self.dtype
        )
        self.enable_cuda_graph = attn_inputs.is_cuda_graph
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def set_params(self, params: rtp_llm_ops.FlashInferMlaAttnParams) -> None:
        """Set the params object to be used by this op."""
        self.fmha_params = params

    def _requires_tensor_core_cuda_graph_replan(self) -> bool:
        # Tensor-core decode refreshes replay-time plan metadata.
        return self.use_tensor_core

    def _plan_decode_wrapper(self, attn_inputs: PyAttentionInputs) -> None:
        if self._requires_tensor_core_cuda_graph_replan():
            page_indptr = self.fmha_params.decode_page_indptr_h
            page_indice = self.fmha_params.page_indice_h
            last_page_len = self.fmha_params.paged_kv_last_page_len_h
            plan_kwargs = {"non_blocking": True}
        else:
            page_indptr = self.fmha_params.decode_page_indptr_d
            page_indice = self.fmha_params.page_indice_d
            last_page_len = self.fmha_params.paged_kv_last_page_len_d
            plan_kwargs = {}

        self.decode_wrapper.plan(
            page_indptr,
            page_indice,
            last_page_len,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.seq_size_per_block,
            q_data_type=self.q_dtype,
            kv_data_type=self.kv_dtype,
            o_data_type=self.dtype,
            **plan_kwargs,
        )

    def prepare(
        self,
        attn_inputs: PyAttentionInputs,
        forbid_realloc: bool = False,
    ) -> ParamsBase:
        """
        Prepare the decode wrapper with paged KV cache parameters.

        forbid_realloc: True only when called from prepare_cuda_graph (replay); forbids buffer realloc.
        """
        # Tensor-core decode plans from the HOST mirrors
        # (decode_page_indptr_h etc. in _plan_decode_wrapper); the device fill
        # only populates the device buffers and leaves the host mirrors at
        # their stale capacity sizes (MIN_CACHE_BATCH_SIZE), which corrupts
        # plan's batch size. Route tensor-core through the host fill.
        if attn_inputs.input_lengths.is_cuda and not self.use_tensor_core:
            self.fmha_params.fill_params_mha_device(
                _device_or(
                    attn_inputs.prefix_lengths_device, attn_inputs.prefix_lengths
                ),
                attn_inputs.sequence_lengths,
                _device_or(attn_inputs.input_lengths_device, attn_inputs.input_lengths),
                _device_or(
                    attn_inputs.kv_cache_kernel_block_id_device,
                    attn_inputs.kv_cache_kernel_block_id,
                ),
                self.seq_size_per_block,
                forbid_realloc=forbid_realloc,
            )
        else:
            block_id_host = attn_inputs.kv_cache_kernel_block_id
            if block_id_host is None or block_id_host.numel() == 0:
                block_id_host = attn_inputs.kv_cache_kernel_block_id_device
            self.fmha_params.fill_params(
                _host_i32(attn_inputs.prefix_lengths),
                _host_i32(attn_inputs.sequence_lengths),
                _host_i32(attn_inputs.input_lengths),
                _host_i32(block_id_host),
                self.seq_size_per_block,
                forbid_realloc=forbid_realloc,
            )

        if self.enable_cuda_graph and self.decode_wrapper._fixed_batch_size == 0:
            batch_size = attn_inputs.input_lengths.size(0)
            self.decode_wrapper._use_cuda_graph = True
            # Both decode backends read these buffers during run(); replay only
            # updates fmha_params in-place, so the wrapper must hold these views.
            self.decode_wrapper._paged_kv_indptr_buf = (
                self.fmha_params.decode_page_indptr_d
            )
            self.decode_wrapper._paged_kv_last_page_len_buf = (
                self.fmha_params.paged_kv_last_page_len_d
            )
            self.decode_wrapper._paged_kv_indices_buf = self.fmha_params.page_indice_d
            self.decode_wrapper._fixed_batch_size = batch_size
            if self.use_tensor_core:
                self.decode_wrapper._qo_indptr_buf = torch.arange(
                    batch_size + 1,
                    dtype=torch.int32,
                    device=self.g_workspace_buffer.device,
                )

        self._plan_decode_wrapper(attn_inputs)
        return self.fmha_params

    def prepare_for_cuda_graph_replay(self, attn_inputs: PyAttentionInputs) -> None:
        """Refresh FlashInfer runtime buffers before replaying the captured graph."""
        if not attn_inputs.sequence_lengths.is_cuda:
            # Host pipeline: refresh host metadata. Tensor-core decode must
            # re-plan from these mirrors because its plan uses host metadata.
            block_id_host = attn_inputs.kv_cache_kernel_block_id
            if block_id_host is None or block_id_host.numel() == 0:
                block_id_host = attn_inputs.kv_cache_kernel_block_id_device
            self.fmha_params.fill_params(
                _host_i32(attn_inputs.prefix_lengths),
                _host_i32(attn_inputs.sequence_lengths),
                _host_i32(attn_inputs.input_lengths),
                _host_i32(block_id_host),
                self.seq_size_per_block,
                forbid_realloc=True,
            )
            if self._requires_tensor_core_cuda_graph_replan():
                self._plan_decode_wrapper(attn_inputs)
            return

        # Device pipeline: update the device-resident buffers in place.
        seq_plus_1 = attn_inputs.sequence_lengths_plus_1_device
        if seq_plus_1 is None or not seq_plus_1.is_cuda:
            seq_plus_1 = (attn_inputs.sequence_lengths.to(torch.int32) + 1).cuda()
        block_id = _device_or(
            attn_inputs.kv_cache_kernel_block_id_device,
            attn_inputs.kv_cache_kernel_block_id,
        )
        if block_id is not None and not block_id.is_cuda:
            block_id = block_id.cuda()
        self.fmha_params.fill_decode_cuda_graph_params(
            seq_plus_1,
            block_id,
            self.seq_size_per_block,
        )

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def forward(
        self, q: torch.Tensor, kv_cache: Optional[LayerKVCache], params: ParamsBase
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required"
        q = quantize_to_fp8_if_needed(
            q.reshape(q.shape[0], self.local_head_num, self.head_dim_qk),
            self.q_dtype,
        )
        paged_kv_cache = kv_cache.kv_cache_base
        if paged_kv_cache is not None and paged_kv_cache.dim() == 2:
            paged_kv_cache = common.reshape_paged_kv_cache(
                paged_kv_cache,
                self.local_kv_head_num,
                self.seq_size_per_block,
                self.head_dim_qk,
            )
        # Decode FP8 defaults to unit scales and the output dtype from plan().
        return self.decode_wrapper.run(q, paged_kv_cache)


class PyFlashinferDecodeImpl(FMHAImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = PyFlashinferDecodeAttnOp(attn_configs, attn_inputs)
        self.rope_impl = FusedRopeKVCacheDecodeOp(attn_configs)
        self.attn_configs = attn_configs

        # Store input info
        self.attn_inputs = attn_inputs

        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.fmha_impl.set_params(self.fmha_params)
        self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs) -> None:
        """Prepare FlashInfer/RoPE buffers and metadata for CUDA graph replay."""
        self.fmha_impl.prepare_for_cuda_graph_replay(attn_inputs)
        # Update rope params for correct position encoding during cuda graph replay
        new_rope_params = self.rope_impl.prepare(attn_inputs)
        common.copy_kv_cache_offset(
            self.rope_params.kv_cache_offset, new_rope_params.kv_cache_offset
        )

    def support_cuda_graph(self) -> bool:
        return True

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return not attn_configs.use_mla

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            qkv = self.rope_impl.forward(qkv, kv_cache, self.rope_params)

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(qkv, kv_cache, self.fmha_params)
