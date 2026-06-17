from typing import Any, Optional

import torch
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_impl.utils import (
    force_py_flashinfer,
    is_sm_100,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import (
    check_attention_inputs,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import (
    AttentionConfigs,
    FMHAType,
    KvCacheDataType,
    ParallelismConfig,
    RopeStyle,
)
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOpQKVOut,
    FusedRopeKVCachePrefillOpQNoTransposeOut,
    LayerKVCache,
    ParamsBase,
    PyAttentionInputs,
    fill_mla_params,
    get_scalar_type,
    rtp_llm_ops,
)

# Constants
# FlashInfer's batch-prefill plan allocates a few internal scratch tensors
# (e.g. ``batch_prefill_tmp_v``) out of this workspace. The size scales with
# max_batch_tokens * num_qo_heads * head_dim, so 128MB can overflow for wide
# GQA models. Allow overriding via env and default to a roomier 512MB.
DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_MB = int(
    __import__("os").environ.get("PY_FLASHINFER_WORKSPACE_SIZE_MB", "512")
)

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
        self.datatype = attn_configs.dtype
        # KV cache may be quantized (fp8/int8) independent of the activation
        # dtype. The reuse-cache path reads the paged KV cache directly, so the
        # plan() below must declare the real KV dtype — otherwise flashinfer
        # raises "dtype of k ... does not match kv_data_type". Mirror the
        # decode path (PyFlashinferDecodeAttnOp).
        self.kv_cache_dtype = attn_configs.kv_cache_dtype
        if self.kv_cache_dtype == KvCacheDataType.INT8:
            self.kv_datatype = torch.int8
        elif self.kv_cache_dtype == KvCacheDataType.FP8:
            self.kv_datatype = torch.float8_e4m3fn
        else:
            self.kv_datatype = self.datatype
        self.max_seq_len = attn_configs.max_seq_len
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.enable_cuda_graph = attn_inputs.is_cuda_graph
        self.prefill_cuda_graph_copy_params = None
        # Pre-allocated buffers for CUDA graph copy path (avoid per-forward allocation)
        self._aligned_q_buf = None
        self._compact_out_buf = None
        # Use Paged KV Cache wrapper
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.g_workspace_buffer,
            "HND",
            backend=backend,
        )

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def set_params(self, params: Any):
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
        # Fill FlashInfer paged-KV plus batch/position metadata on device.
        self.fmha_params.fill_params_mha_device(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id_device,
            self.page_size,
            forbid_realloc,
        )
        # Store CUDA graph copy parameters
        # Define qo_indptr early for CUDA graph initialization
        if attn_inputs.prefill_cuda_graph_copy_params is not None:
            # For CUDA graph mode, create a buffer that will be filled later
            self.input_lengths = attn_inputs.input_lengths
            self.cu_seq_lens = attn_inputs.cu_seqlens
            qo_indptr = attn_inputs.cu_seqlens.clone()
        else:
            qo_indptr = attn_inputs.cu_seqlens[: attn_inputs.input_lengths.size(0) + 1]

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
            self.prefill_wrapper._fixed_batch_size = len(attn_inputs.cu_seqlens) - 1
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
            self.cu_seq_lens[: attn_inputs.cu_seqlens.size(0)] = attn_inputs.cu_seqlens
            # Build qo_indptr matching the padded Q layout produced by small2large copy.
            # Each batch's Q tokens sit at [i*max_seq_len, i*max_seq_len + input_len_i)
            # in the padded buffer, so qo_indptr[i] = i*max_seq_len, but we set
            # qo_indptr[i+1] = i*max_seq_len + input_len_i to tell FlashInfer the
            # exact number of real tokens per batch (avoiding padding token processing
            # which causes numerical differences).
            batch_size = attn_inputs.input_lengths.size(0)
            max_sl = self.prefill_cuda_graph_copy_params.max_seq_len
            offsets = (
                torch.arange(
                    batch_size, device=self.qo_indptr.device, dtype=self.qo_indptr.dtype
                )
                * max_sl
            )
            self.qo_indptr[0] = 0
            self.qo_indptr[1 : batch_size + 1] = offsets + attn_inputs.input_lengths.to(
                self.qo_indptr.device
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
            causal=True,
            q_data_type=self.datatype,
            kv_data_type=self.kv_datatype,
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
            result = self.prefill_wrapper.run(q, paged_kv_cache)

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
        self.datatype = attn_configs.dtype
        self.params = None

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def set_params(self, params: ParamsBase):
        """Set the params object to be used by this op."""
        self.params = params

    def prepare(self, attn_inputs: PyAttentionInputs) -> ParamsBase:
        """
        Prepare the prefill wrapper

        Args:
            attn_inputs: Attention inputs containing sequence information
        """
        batch_size = attn_inputs.input_lengths.size(0)
        cu_seqlens = attn_inputs.cu_seqlens[: batch_size + 1]

        # Ragged prefill only needs the device metadata planner when a paged KV
        # cache is present. During warmup / context-only prefill (no prefix) the
        # block-id tensor is None, so skip the paged-metadata fill (the C++ binding
        # requires a real torch.Tensor) and just plan the ragged layout below.
        if attn_inputs.kv_cache_kernel_block_id_device is not None:
            self.params.fill_params_mha_device(
                attn_inputs.prefix_lengths,
                attn_inputs.sequence_lengths,
                attn_inputs.input_lengths,
                attn_inputs.kv_cache_kernel_block_id_device,
                self.page_size,
            )

        self.prefill_wrapper.plan(
            cu_seqlens,
            cu_seqlens,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.head_dim_vo,
            causal=True,
            q_data_type=get_scalar_type(attn_inputs.dtype),
        )
        return self.params if self.params is not None else ParamsBase()

    @staticmethod
    def support(attn_inputs: PyAttentionInputs) -> bool:
        return (
            attn_inputs.prefix_lengths.numel() <= 0
            or attn_inputs.prefix_lengths.sum().item() == 0
        )

    ## 1. pure prefill attn: qkv contains q and k,v
    ## 2. paged attn: qkv is only q, and kv is in kv_cache
    def forward(
        self, qkv: torch.Tensor, kv_cache: Optional[LayerKVCache]
    ) -> torch.Tensor:
        qkv = qkv.reshape(qkv.shape[0], -1)
        q, k, v = torch.split(
            qkv,
            [
                self.head_dim_qk * self.local_head_num,
                self.head_dim_qk * self.local_kv_head_num,
                self.head_dim_vo * self.local_kv_head_num,
            ],
            dim=-1,
        )
        q = q.reshape(q.shape[0], self.local_head_num, self.head_dim_qk)
        k = k.reshape(k.shape[0], self.local_kv_head_num, self.head_dim_qk)
        v = v.reshape(v.shape[0], self.local_kv_head_num, self.head_dim_vo)
        return self.prefill_wrapper.run(q, k, v)


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
        # Fused RoPE + paged-KV write (rtp_kernel). SM103 is supported since the
        # 260608 rtp_kernel wheel, so this replaces the previous FlashInfer
        # MhaRotaryEmbeddingOp + KVCacheWriteOp workaround. The fused op applies
        # RoPE, appends K/V to the paged cache (FlashInfer-HND layout, read back
        # by the Python FlashInfer FMHA wrapper) and returns the FMHA query input
        # in a single kernel launch.
        self.rope_kvcache_impl = (
            self._create_rope_kvcache_impl(attn_configs)
            if self.need_rope_kv_cache
            else None
        )
        self.create_params(attn_inputs)
        self.fmha_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        self.fmha_impl.prepare(attn_inputs, forbid_realloc=True)
        if self.rope_kvcache_impl is not None:
            self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)

    def create_params(self, attn_inputs: PyAttentionInputs):
        """Create the FlashInfer FMHA params and the fused-RoPE params.

        The Python FlashInfer FMHA keeps its own paged-KV metadata params, while
        the fused RoPE op owns a separate ``FusedRopeAttnParams`` (kv-cache block
        array, cu_seqlens, prefix lengths, ...). They are independent.
        """
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.fmha_impl.set_params(self.fmha_params)
        self.rope_params = (
            self.rope_kvcache_impl.prepare(attn_inputs)
            if self.rope_kvcache_impl is not None
            else None
        )

    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        """Create FMHA implementation. To be overridden by subclasses."""
        raise NotImplementedError("Subclass must implement _create_fmha_impl")

    def _create_rope_kvcache_impl(self, attn_configs: AttentionConfigs) -> Any:
        """Create the fused RoPE+KV-write op. To be overridden by subclasses."""
        raise NotImplementedError("Subclass must implement _create_rope_kvcache_impl")

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Common forward implementation for all prefill implementations."""
        # Fused RoPE + paged-KV write; returns the FMHA query input (qkv for the
        # ragged layout, q for the paged layout).
        if self.need_rope_kv_cache and self.rope_kvcache_impl is not None:
            qkv = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(qkv, kv_cache)


class PyFlashinferPagedPrefillImpl(PyFlashinferPrefillImplBase):
    """FlashInfer paged prefill implementation; RoPE+KV-write via the fused rtp_kernel op."""

    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        """Create paged FMHA implementation."""
        return PyFlashinferPrefillPagedAttnOp(attn_configs, attn_inputs)

    def _create_rope_kvcache_impl(self, attn_configs: AttentionConfigs) -> Any:
        """Fused RoPE+KV-write returning ragged q [tokens, heads, dim] for paged FMHA."""
        return FusedRopeKVCachePrefillOpQNoTransposeOut(attn_configs)

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        """Check if paged prefill implementation is supported.

        Returns True if:
        1. Not running on SM 10.0 (Blackwell) architecture
        2. The underlying paged FMHA op supports the inputs
        3. MhaRotaryEmbeddingOp supports the inputs
        """
        return (
            (not is_sm_100() or force_py_flashinfer())
            and PyFlashinferPrefillPagedAttnOp.support(attn_inputs)
            and attn_configs.rope_config.style != RopeStyle.Mrope
        )

    def support_cuda_graph(self) -> bool:
        return True


class PyFlashinferPrefillImpl(PyFlashinferPrefillImplBase):
    """FlashInfer ragged prefill implementation; RoPE+KV-write via the fused rtp_kernel op."""

    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        """Create ragged FMHA implementation."""
        return PyFlashinferPrefillAttnOp(attn_configs)

    def _create_rope_kvcache_impl(self, attn_configs: AttentionConfigs) -> Any:
        """Fused RoPE+KV-write returning the full qkv [tokens, (h+2kv)*d] for ragged FMHA."""
        return FusedRopeKVCachePrefillOpQKVOut(attn_configs)

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        """Check if ragged prefill implementation is supported.

        Returns True if:
        1. Not running on SM 10.0 (Blackwell) architecture
        2. The underlying ragged FMHA op supports the inputs
           (requires prefix_lengths to be empty or zero)
        3. MhaRotaryEmbeddingOp supports the inputs
        """
        return (
            (not is_sm_100() or force_py_flashinfer())
            and PyFlashinferPrefillAttnOp.support(attn_inputs)
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
        self.kv_cache_dtype = attn_configs.kv_cache_dtype
        # Decode owns its FlashInfer paged-KV metadata params (the decode Impl
        # does not share a params object across ops the way prefill does); it is
        # filled and the wrapper planned in prepare() below.
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def prepare(self, attn_inputs: PyAttentionInputs):
        # from rtp_llm.models_py.utils.debug import set_trace_on_tty
        # set_trace_on_tty()
        # Convert kv_cache_dtype to torch dtype
        if self.kv_cache_dtype == KvCacheDataType.INT8:
            kv_datatype = torch.int8
        elif self.kv_cache_dtype == KvCacheDataType.FP8:
            kv_datatype = torch.float8_e4m3fn
        else:  # BASE
            kv_datatype = get_scalar_type(attn_inputs.dtype)

        # Steady-state decode drops the host metadata loop and H2D copy.
        self.fmha_params.fill_params_mha_device(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id_device,
            self.seq_size_per_block,
        )
        # Get torch.dtype from attention configs
        self.decode_wrapper.plan(
            self.fmha_params.decode_page_indptr_d,
            self.fmha_params.page_indice_d,
            self.fmha_params.paged_kv_last_page_len_d,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.seq_size_per_block,
            q_data_type=get_scalar_type(attn_inputs.dtype),
            kv_data_type=kv_datatype,
        )
        return self.fmha_params

    def prepare_for_cuda_graph_replay(self, attn_inputs: PyAttentionInputs) -> None:
        """Update CUDA graph replay buffers without calling plan().

        Replay refreshes decode metadata in-place via fill_decode_cuda_graph_params,
        falling back to the device MHA planner when the decode-only input is absent.
        """
        fill_decode = getattr(self.fmha_params, "fill_decode_cuda_graph_params", None)
        if (
            callable(fill_decode)
            and attn_inputs.sequence_lengths_plus_1_d is not None
            and attn_inputs.sequence_lengths_plus_1_d.numel() > 0
        ):
            fill_decode(
                attn_inputs.sequence_lengths_plus_1_d,
                attn_inputs.kv_cache_kernel_block_id_device,
                self.seq_size_per_block,
            )
            return

        # CUDA graph replay fallback when sequence_lengths_plus_1_d is absent.
        self.fmha_params.fill_params_mha_device(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id_device,
            self.seq_size_per_block,
            forbid_realloc=True,
        )

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def forward(
        self, q: torch.Tensor, kv_cache: Optional[LayerKVCache], params: ParamsBase
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required"
        q = q.reshape(q.shape[0], self.local_head_num, self.head_dim_qk)
        paged_kv_cache = kv_cache.kv_cache_base
        if paged_kv_cache is not None and paged_kv_cache.dim() == 2:
            paged_kv_cache = common.reshape_paged_kv_cache(
                paged_kv_cache,
                self.local_kv_head_num,
                self.seq_size_per_block,
                self.head_dim_qk,
            )
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
        self.fmha_impl = PyFlashinferDecodeAttnOp(attn_configs)
        self.attn_configs = attn_configs

        # Store input info
        self.attn_inputs = attn_inputs

        # fmha_impl.prepare() fills the shared FlashInfer paged-KV params
        # (positions_d, batch_indice_d, the page table, last_page_len, ...) and
        # plans the decode wrapper. All of these are produced by the SM103-safe
        # C++ fill_params_mha_device.
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)

        # Fused decode RoPE + paged-KV write (rtp_kernel). The fused C++ decode
        # kernel (decode_fused_rope_kvcache) gained an SM103 image in the 260608
        # rtp_kernel wheel, so it replaces the previous FlashInfer
        # MhaRotaryEmbeddingOp + KVCacheWriteOp workaround. It applies RoPE,
        # appends K/V to the paged cache (FlashInfer-HND layout, read back by the
        # Python FlashInfer decode wrapper) and returns the rotated query
        # [tokens, heads, dim] in a single launch. It owns its own
        # FusedRopeAttnParams (block-array offset + position ids), independent of
        # the FlashInfer FMHA params filled above.
        self.rope_kvcache_impl: Optional[FusedRopeKVCacheDecodeOp] = None
        self.rope_params = None
        if self.need_rope_kv_cache:
            self.rope_kvcache_impl = FusedRopeKVCacheDecodeOp(attn_configs)
            self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)

        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

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
        # Fused RoPE + paged-KV write; returns the rotated query for decode FMHA.
        if self.need_rope_kv_cache and self.rope_kvcache_impl is not None:
            assert kv_cache is not None, "decode requires a paged kv_cache"
            qkv = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward (decode attention reads K/V from the paged cache)
        return self.fmha_impl.forward(qkv, kv_cache, self.fmha_params)
