from typing import Any, Optional

import torch
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
from rtp_llm.models_py.modules.factory.attention.cuda_impl.utils import is_sm_100
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
    LayerKVCache,
    ParamsBase,
    PyAttentionInputs,
    fill_mla_params,
    get_scalar_type,
    rtp_llm_ops,
)

# Constants
DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_MB = 128

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
        self.kv_cache_dtype = attn_configs.kv_cache_dtype
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

    def _get_kv_dtype(self, attn_inputs: PyAttentionInputs) -> torch.dtype:
        if self.kv_cache_dtype == KvCacheDataType.INT8:
            return torch.int8
        elif self.kv_cache_dtype == KvCacheDataType.FP8:
            return torch.float8_e4m3fn
        return get_scalar_type(attn_inputs.dtype)

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
        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id_host,
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
            # Allocate DEDICATED max-sized CUDA-graph buffers instead of
            # aliasing to fmha_params.{decode_page_indptr_d, page_indice_d,
            # paged_kv_last_page_len_d}. Those fmha_params tensors are
            # reshaped per call by FlashInferMlaParams::refreshBuffer
            # (set_sizes_contiguous), violating FlashInfer's CG contract
            # and causing captured graphs to bake stale plan_info scalars.
            fixed_batch_size = len(attn_inputs.cu_seqlens) - 1
            max_pages_per_seq = (
                self.max_seq_len + self.page_size - 1
            ) // self.page_size
            max_total_pages = max(fixed_batch_size * max_pages_per_seq, 1)
            dtype = qo_indptr.dtype
            # FlashInfer CG buffers must be GPU-resident so captured kernels can
            # dereference them.  qo_indptr may be a non-pinned CPU tensor (result
            # of cu_seqlens.clone() which does NOT preserve pin_memory), so we
            # move everything to CUDA explicitly.
            cg_device = torch.device("cuda")
            qo_indptr = qo_indptr.to(cg_device)
            self._paged_kv_indptr_buf = torch.zeros(
                fixed_batch_size + 1, dtype=dtype, device=cg_device
            )
            self._paged_kv_last_page_len_buf = torch.zeros(
                fixed_batch_size, dtype=dtype, device=cg_device
            )
            self._paged_kv_indices_buf = torch.zeros(
                max_total_pages, dtype=dtype, device=cg_device
            )
            self.prefill_wrapper._use_cuda_graph = True
            self.prefill_wrapper._qo_indptr_buf = qo_indptr
            self.prefill_wrapper._paged_kv_indptr_buf = self._paged_kv_indptr_buf
            self.prefill_wrapper._paged_kv_last_page_len_buf = (
                self._paged_kv_last_page_len_buf
            )
            self.prefill_wrapper._paged_kv_indices_buf = self._paged_kv_indices_buf
            self.prefill_wrapper._fixed_batch_size = fixed_batch_size
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
            batch_size = attn_inputs.input_lengths.size(0)
            max_batch_size = self.prefill_cuda_graph_copy_params.max_batch_size
            self.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size.copy_(
                attn_inputs.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size
            )
            self.input_lengths[:batch_size] = attn_inputs.input_lengths
            self.cu_seq_lens[: batch_size + 1] = attn_inputs.cu_seqlens
            if batch_size < max_batch_size:
                self.input_lengths[batch_size:max_batch_size] = 0
                last_cu_seqlen = attn_inputs.cu_seqlens[batch_size]
                self.cu_seq_lens[batch_size + 1 : max_batch_size + 1] = last_cu_seqlen
            # Build qo_indptr matching the padded Q layout produced by small2large copy.
            # Each batch's Q tokens sit at [i*max_seq_len, i*max_seq_len + input_len_i)
            # in the padded buffer, so qo_indptr[i] = i*max_seq_len, but we set
            # qo_indptr[i+1] = i*max_seq_len + input_len_i to tell FlashInfer the
            # exact number of real tokens per batch (avoiding padding token processing
            # which causes numerical differences).
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
            if batch_size < max_batch_size:
                self.qo_indptr[batch_size + 1 : max_batch_size + 1] = self.qo_indptr[
                    batch_size
                ]
            qo_indptr = self.qo_indptr

        kv_dtype = self._get_kv_dtype(attn_inputs)

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
            kv_data_type=kv_dtype,
            disable_split_kv=self.enable_cuda_graph,
        )
        # CG replay: pad stale entries in the fixed-size CG buffers.
        # plan() only copies source tensors up to actual_batch_size+1 entries into
        # the max-sized CG buffers; entries [actual_batch_size+1:max_batch_size+1]
        # remain stale from prior calls.  During replay (which processes max_batch_size
        # sequences), padding batches use those stale page ranges and read wrong KV data.
        if getattr(self, "_paged_kv_indptr_buf", None) is not None:
            actual_bs = attn_inputs.input_lengths.size(0)
            max_bs = self._paged_kv_indptr_buf.size(0) - 1
            if actual_bs < max_bs:
                last_indptr = self._paged_kv_indptr_buf[actual_bs].item()
                self._paged_kv_indptr_buf[actual_bs + 1 :].fill_(last_indptr)
                self._paged_kv_last_page_len_buf[actual_bs:].fill_(1)
            else:
                torch.clamp_min_(self._paged_kv_last_page_len_buf, 1)
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

    def prepare(
        self, attn_inputs: PyAttentionInputs, forbid_realloc: bool = False
    ) -> ParamsBase:
        """
        Prepare the prefill wrapper

        Args:
            attn_inputs: Attention inputs containing sequence information
            forbid_realloc: Unused for ragged prefill (CUDA graph replay compatibility).
        """
        batch_size = attn_inputs.input_lengths.size(0)
        cu_seqlens = attn_inputs.cu_seqlens[: batch_size + 1]

        self.params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id_host,
            self.page_size,
            forbid_realloc,
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
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            if self.rope_impl is not None:
                # Apply RoPE and get Q, K, V
                query, key, value = self.rope_impl.forward(qkv)
            else:
                # No RoPE, just split QKV
                query, key, value = self._split_qkv(qkv)

            # Write KV to cache
            self.kv_cache_write_op.forward(key, value, kv_cache)

            # Pass query to FMHA (for paged) or reconstruct qkv (for ragged)
            qkv = self._prepare_fmha_input(query, key, value)

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(qkv, kv_cache)

    def _prepare_fmha_input(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Prepare input for FMHA. To be overridden by subclasses if needed."""
        # Default: just return query (for paged layout)
        return query


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
    ) -> torch.Tensor:
        """For paged layout, only return query (KV is already in cache)."""
        return query

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        """Check if paged prefill implementation is supported.

        Returns True if:
        1. Not running on SM 10.0 (Blackwell) architecture
        2. The underlying paged FMHA op supports the inputs
        3. MhaRotaryEmbeddingOp supports the inputs
        """
        return not is_sm_100() and PyFlashinferPrefillPagedAttnOp.support(attn_inputs)

    def support_cuda_graph(self) -> bool:
        return True


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
    ) -> torch.Tensor:
        """For ragged layout, reconstruct full qkv tensor from q, k, v."""
        # query: [total_tokens, num_heads, head_dim]
        # key: [total_tokens, num_kv_heads, head_dim]
        # value: [total_tokens, num_kv_heads, head_dim]

        # Flatten to 2D and concatenate
        q_flat = query.reshape(
            query.shape[0], -1
        )  # [total_tokens, num_heads * head_dim]
        k_flat = key.reshape(
            key.shape[0], -1
        )  # [total_tokens, num_kv_heads * head_dim]
        v_flat = value.reshape(
            value.shape[0], -1
        )  # [total_tokens, num_kv_heads * head_dim]

        # Concatenate along feature dimension
        qkv = torch.cat(
            [q_flat, k_flat, v_flat], dim=-1
        )  # [total_tokens, (num_heads + 2*num_kv_heads) * head_dim]

        return qkv

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        """Check if ragged prefill implementation is supported.

        Returns True if:
        1. Not running on SM 10.0 (Blackwell) architecture
        2. The underlying ragged FMHA op supports the inputs
           (requires prefix_lengths to be empty or zero)
        3. MhaRotaryEmbeddingOp supports the inputs
        """
        return not is_sm_100() and PyFlashinferPrefillAttnOp.support(attn_inputs)


class PyFlashinferPrefillPagedTargetVerifyAttnOp(object):
    """Target verify using paged KV cache directly (no KV gather).

    Uses BatchPrefillWithPagedKVCacheWrapper which reads from the paged KV
    cache natively.  FlashInfer's causal mask with qo_len < kv_len treats Q
    as the suffix of KV (mask = tril(diagonal=kv_len-qo_len)), which is
    exactly what MTP verification needs.

    Compared to the ragged gather approach (~27 small CUDA kernels per layer
    for arange/searchsorted/index_select/where/copy), this launches only the
    single BatchPrefillWithPagedKVCacheKernel.
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
        self.max_seq_len = attn_configs.max_seq_len
        self.datatype = attn_configs.dtype
        self.kv_cache_dtype = attn_configs.kv_cache_dtype
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.enable_cuda_graph = getattr(attn_inputs, "is_cuda_graph", False)
        self._cuda_graph_initialized = False
        # Dedicated max-sized CUDA-graph buffers; allocated at first prepare()
        # when max_batch_size is known from the captured attn_inputs layout.
        # These MUST be separate from fmha_params.{decode_page_indptr_d,
        # page_indice_d, paged_kv_last_page_len_d} — those tensors are
        # reshaped in place per call by FlashInferMlaParams::refreshBuffer,
        # violating FlashInfer's CG contract that the buffers must be sized
        # for the maximum workload over the wrapper lifetime.  Aliasing
        # caused the captured graph's kernel launch to bake stale
        # plan_info scalars and concurrent replays read wrong KV ranges.
        self._qo_indptr_buf: Optional[torch.Tensor] = None
        self._paged_kv_indptr_buf: Optional[torch.Tensor] = None
        self._paged_kv_last_page_len_buf: Optional[torch.Tensor] = None
        self._paged_kv_indices_buf: Optional[torch.Tensor] = None
        self._capture_qo_indptr: Optional[torch.Tensor] = None
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.g_workspace_buffer,
            "HND",
            backend=backend,
        )

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def _get_kv_dtype(self, attn_inputs: PyAttentionInputs) -> torch.dtype:
        if self.kv_cache_dtype == KvCacheDataType.INT8:
            return torch.int8
        elif self.kv_cache_dtype == KvCacheDataType.FP8:
            return torch.float8_e4m3fn
        return get_scalar_type(attn_inputs.dtype)

    def set_params(self, params: Any) -> None:
        self.fmha_params = params

    def _get_qo_indptr(
        self, attn_inputs: PyAttentionInputs, batch_size: int
    ) -> torch.Tensor:
        """QO indptr for target verify: draft token cumulative lengths."""
        if (
            hasattr(attn_inputs, "decode_cu_seqlens_d")
            and attn_inputs.decode_cu_seqlens_d is not None
            and attn_inputs.decode_cu_seqlens_d.numel() > batch_size
        ):
            return attn_inputs.decode_cu_seqlens_d[: batch_size + 1]
        return self.fmha_params.qo_indptr_d[: batch_size + 1]

    def prepare(
        self, attn_inputs: PyAttentionInputs, forbid_realloc: bool = False
    ) -> ParamsBase:
        check_attention_inputs(attn_inputs)
        batch_size = attn_inputs.input_lengths.size(0)
        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id_host,
            self.page_size,
            forbid_realloc,
        )

        qo_indptr = self._get_qo_indptr(attn_inputs, batch_size)
        kv_dtype = self._get_kv_dtype(attn_inputs)

        if self.enable_cuda_graph and not self._cuda_graph_initialized:
            # Size CG buffers to the maximum workload: max_batch_size slots,
            # each holding up to ceil(max_seq_len / page_size) page IDs.
            # FlashInfer's plan() in CG mode copies the input indptr/indices
            # tensors into these dedicated buffers via .copy_() — that write
            # must target storage distinct from the source (fmha_params.*_d),
            # otherwise it degenerates to a self-copy and the captured graph
            # ends up reading from the dynamically-reshaped fmha_params
            # storage instead of the stable CG buffer.
            max_pages_per_seq = (
                self.max_seq_len + self.page_size - 1
            ) // self.page_size
            max_total_pages = max(batch_size * max_pages_per_seq, 1)
            device = qo_indptr.device
            dtype = qo_indptr.dtype
            self._qo_indptr_buf = torch.zeros(
                batch_size + 1, dtype=dtype, device=device
            )
            self._paged_kv_indptr_buf = torch.zeros(
                batch_size + 1, dtype=dtype, device=device
            )
            self._paged_kv_last_page_len_buf = torch.zeros(
                batch_size, dtype=dtype, device=device
            )
            self._paged_kv_indices_buf = torch.zeros(
                max_total_pages, dtype=dtype, device=device
            )
            self.prefill_wrapper._use_cuda_graph = True
            self.prefill_wrapper._qo_indptr_buf = self._qo_indptr_buf
            self.prefill_wrapper._paged_kv_indptr_buf = self._paged_kv_indptr_buf
            self.prefill_wrapper._paged_kv_last_page_len_buf = (
                self._paged_kv_last_page_len_buf
            )
            self.prefill_wrapper._paged_kv_indices_buf = self._paged_kv_indices_buf
            self.prefill_wrapper._fixed_batch_size = batch_size
            self._capture_qo_indptr = qo_indptr.clone()
            self._cuda_graph_initialized = True

        # In CG mode, always use the capture-time qo_indptr so that
        # plan_info (baked into the captured CUDA graph as kernel arguments)
        # stays identical across capture and all replays.  The C++ runner
        # modifies decode_cu_seqlens_d for padding batches (filling them
        # with the last active offset), which changes total_num_rows and
        # per-batch Q counts, producing a different plan_info grid that
        # mismatches the captured kernel launch configuration.
        if self._capture_qo_indptr is not None:
            qo_indptr = self._capture_qo_indptr

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
            kv_data_type=kv_dtype,
            disable_split_kv=self.enable_cuda_graph,
        )
        # CG replay: ensure padding batches have safe KV metadata.
        # plan() copies source tensors into CG buffers; padding batches
        # (input_lengths=0) get 0 KV pages from fill_params.  With the
        # fixed capture-time qo_indptr, the kernel still launches CTAs
        # for padding batches.  Give each padding batch 1 dummy KV page
        # (pointing to a real page) so the kernel reads valid memory
        # instead of computing negative kv_len from 0 pages.
        if self._cuda_graph_initialized and self._paged_kv_indptr_buf is not None:
            actual_bs = int((attn_inputs.input_lengths > 0).sum().item())
            max_bs = self._paged_kv_indptr_buf.size(0) - 1
            if actual_bs < max_bs and actual_bs > 0:
                last_real_indptr = int(self._paged_kv_indptr_buf[actual_bs].item())
                num_padding = max_bs - actual_bs
                self._paged_kv_indptr_buf[actual_bs + 1 :] = (
                    last_real_indptr
                    + torch.arange(
                        1,
                        num_padding + 1,
                        device=self._paged_kv_indptr_buf.device,
                        dtype=self._paged_kv_indptr_buf.dtype,
                    )
                )
                safe_page = int(self._paged_kv_indices_buf[0].item())
                end_idx = min(
                    last_real_indptr + num_padding,
                    self._paged_kv_indices_buf.size(0),
                )
                self._paged_kv_indices_buf[last_real_indptr:end_idx].fill_(safe_page)
                self._paged_kv_last_page_len_buf[actual_bs:].fill_(1)
            else:
                torch.clamp_min_(self._paged_kv_last_page_len_buf, 1)
        return self.fmha_params

    @staticmethod
    def support(attn_inputs: PyAttentionInputs) -> bool:
        return True

    def forward(
        self, q: torch.Tensor, kv_cache: Optional[LayerKVCache]
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required for target verify"
        assert q.dim() == 3, f"Expected q [total_tokens, H, D], got shape {q.shape}"
        paged_kv_cache = kv_cache.kv_cache_base
        if paged_kv_cache.dim() == 2:
            paged_kv_cache = common.reshape_paged_kv_cache(
                paged_kv_cache,
                self.local_kv_head_num,
                self.page_size,
                self.head_dim_qk,
            )
        return self.prefill_wrapper.run(q, paged_kv_cache)


class PyFlashinferTargetVerifyPrefillImpl(PyFlashinferPrefillImplBase):
    """FA2 paged batch prefill for MTP target verify.

    Uses BatchPrefillWithPagedKVCacheWrapper (FA2 backend) which reads from
    the paged KV cache directly, avoiding the ~27 small gather kernels per
    layer that the ragged approach requires.
    """

    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        return PyFlashinferPrefillPagedTargetVerifyAttnOp(
            attn_configs, attn_inputs, backend="auto"
        )

    def _create_rope_impl(self, attn_configs: AttentionConfigs) -> Any:
        if attn_configs.rope_config.style == RopeStyle.No:
            return None
        return MhaRotaryEmbeddingOp(attn_configs)

    def _prepare_fmha_input(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        return query

    def support_cuda_graph(self) -> bool:
        return True

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        if not attn_inputs.is_prefill:
            return False
        if not getattr(attn_inputs, "is_target_verify", False):
            return False
        if attn_configs.use_mla:
            return False
        return True


class PyFlashinferDecodeTargetVerifyAttnOp(object):
    """MTP target verify using BatchDecodeWithPagedKVCacheWrapper + tensor cores.

    Flattens B requests x Q verify tokens into B*Q individual decode requests,
    each with 1 query token sharing the same paged KV cache as its parent.
    This is ~4-5x faster than the BatchPrefill approach for typical MTP verify
    workloads (small Q, large prefix, high GQA ratio).
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
    ) -> None:
        self.g_workspace_buffer = get_py_flashinfer_workspace_buffer()
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        self.head_dim_vo = attn_configs.size_per_head
        self.page_size = attn_configs.kernel_tokens_per_block
        self.max_seq_len = attn_configs.max_seq_len
        self.datatype = attn_configs.dtype
        self.kv_cache_dtype = attn_configs.kv_cache_dtype
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.enable_cuda_graph = getattr(attn_inputs, "is_cuda_graph", False)
        self._cuda_graph_initialized = False
        self.use_tensor_core = determine_use_tensor_core_from_configs(attn_configs)
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.g_workspace_buffer,
            "HND",
            use_tensor_cores=self.use_tensor_core,
        )
        # Pre-allocated expanded page table buffers for the flattened batch
        self._flat_page_indptr: Optional[torch.Tensor] = None
        self._flat_page_indices: Optional[torch.Tensor] = None
        self._flat_last_page_len: Optional[torch.Tensor] = None

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def _get_kv_dtype(self, attn_inputs: PyAttentionInputs) -> torch.dtype:
        if self.kv_cache_dtype == KvCacheDataType.INT8:
            return torch.int8
        elif self.kv_cache_dtype == KvCacheDataType.FP8:
            return torch.float8_e4m3fn
        return get_scalar_type(attn_inputs.dtype)

    def set_params(self, params: Any) -> None:
        self.fmha_params = params

    def _get_q_lens(
        self, attn_inputs: PyAttentionInputs, batch_size: int
    ) -> torch.Tensor:
        """Get per-request query lengths for the target verify batch."""
        if (
            hasattr(attn_inputs, "decode_cu_seqlens_d")
            and attn_inputs.decode_cu_seqlens_d is not None
            and attn_inputs.decode_cu_seqlens_d.numel() > batch_size
        ):
            cu = attn_inputs.decode_cu_seqlens_d[: batch_size + 1]
            return cu[1:] - cu[:-1]
        qo_indptr = self.fmha_params.qo_indptr_d[: batch_size + 1]
        return qo_indptr[1:] - qo_indptr[:-1]

    def _expand_page_table(
        self,
        batch_size: int,
        q_lens: torch.Tensor,
        page_indptr: torch.Tensor,
        page_indices: torch.Tensor,
        last_page_len: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand the B-request page table to a B*Q flattened page table
        with per-token causal masking.

        Token j within request i should only attend to KV entries
        [0, ..., S_i - Q_i + j], i.e. S_i - Q_i + j + 1 visible entries.
        We set per-token page count and last_page_len accordingly.
        """
        flat_batch = int(q_lens.sum().item())
        device = page_indptr.device
        dtype = page_indptr.dtype
        page_size = self.page_size

        flat_num_pages = torch.empty(flat_batch, dtype=dtype, device=device)
        flat_lpl = torch.empty(flat_batch, dtype=dtype, device=device)

        flat_idx = 0
        for i in range(batch_size):
            S_i = int(sequence_lengths[i].item())
            Q_i = int(q_lens[i].item())
            prefix_len = S_i - Q_i
            for j in range(Q_i):
                visible = prefix_len + j + 1
                n_pages = (visible + page_size - 1) // page_size
                lpl = ((visible - 1) % page_size) + 1
                flat_num_pages[flat_idx] = n_pages
                flat_lpl[flat_idx] = lpl
                flat_idx += 1

        flat_indptr = torch.zeros(flat_batch + 1, dtype=dtype, device=device)
        flat_indptr[1:] = torch.cumsum(flat_num_pages, dim=0)

        total_flat_pages = int(flat_indptr[-1].item())
        flat_indices = torch.empty(total_flat_pages, dtype=dtype, device=device)

        flat_idx = 0
        offset = 0
        for i in range(batch_size):
            Q_i = int(q_lens[i].item())
            src_start = int(page_indptr[i].item())
            for j in range(Q_i):
                n_pages = int(flat_num_pages[flat_idx].item())
                flat_indices[offset : offset + n_pages] = page_indices[
                    src_start : src_start + n_pages
                ]
                offset += n_pages
                flat_idx += 1

        return flat_indptr, flat_indices, flat_lpl

    def prepare(
        self, attn_inputs: PyAttentionInputs, forbid_realloc: bool = False
    ) -> Any:
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import (
            check_attention_inputs,
        )

        check_attention_inputs(attn_inputs)
        batch_size = attn_inputs.input_lengths.size(0)
        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id_host,
            self.page_size,
            forbid_realloc,
        )

        q_lens = self._get_q_lens(attn_inputs, batch_size)
        flat_batch = int(q_lens.sum().item())
        kv_dtype = self._get_kv_dtype(attn_inputs)

        page_indptr = self.fmha_params.decode_page_indptr_d
        page_indices = self.fmha_params.page_indice_d
        last_page_len = self.fmha_params.paged_kv_last_page_len_d

        flat_indptr, flat_indices, flat_lpl = self._expand_page_table(
            batch_size,
            q_lens,
            page_indptr,
            page_indices,
            last_page_len,
            attn_inputs.sequence_lengths,
        )

        if self.enable_cuda_graph and not self._cuda_graph_initialized:
            max_pages_per_seq = (
                self.max_seq_len + self.page_size - 1
            ) // self.page_size
            max_total_flat_pages = max(flat_batch * max_pages_per_seq, 1)
            device = flat_indptr.device
            dtype = flat_indptr.dtype
            self._flat_page_indptr = torch.zeros(
                flat_batch + 1, dtype=dtype, device=device
            )
            self._flat_last_page_len = torch.zeros(
                flat_batch, dtype=dtype, device=device
            )
            self._flat_page_indices = torch.zeros(
                max_total_flat_pages, dtype=dtype, device=device
            )
            # qo_indptr for decode: each flat request has exactly 1 query token
            self._flat_qo_indptr = torch.arange(
                flat_batch + 1, dtype=dtype, device=device
            )
            self.decode_wrapper._use_cuda_graph = True
            self.decode_wrapper._qo_indptr_buf = self._flat_qo_indptr
            self.decode_wrapper._paged_kv_indptr_buf = self._flat_page_indptr
            self.decode_wrapper._paged_kv_indices_buf = self._flat_page_indices
            self.decode_wrapper._paged_kv_last_page_len_buf = self._flat_last_page_len
            self.decode_wrapper._fixed_batch_size = flat_batch
            self._cuda_graph_initialized = True

        if self._cuda_graph_initialized:
            n_indptr = min(flat_indptr.size(0), self._flat_page_indptr.size(0))
            self._flat_page_indptr[:n_indptr].copy_(flat_indptr[:n_indptr])
            n_idx = min(flat_indices.size(0), self._flat_page_indices.size(0))
            self._flat_page_indices[:n_idx].copy_(flat_indices[:n_idx])
            n_lpl = min(flat_lpl.size(0), self._flat_last_page_len.size(0))
            self._flat_last_page_len[:n_lpl].copy_(flat_lpl[:n_lpl])
            actual_flat = flat_indptr.size(0) - 1
            max_flat = self._flat_page_indptr.size(0) - 1
            if actual_flat < max_flat and actual_flat > 0:
                last_real_indptr = int(self._flat_page_indptr[actual_flat].item())
                num_padding = max_flat - actual_flat
                self._flat_page_indptr[actual_flat + 1 :] = (
                    last_real_indptr
                    + torch.arange(
                        1,
                        num_padding + 1,
                        device=self._flat_page_indptr.device,
                        dtype=self._flat_page_indptr.dtype,
                    )
                )
                safe_page = int(self._flat_page_indices[0].item())
                end_idx = min(
                    last_real_indptr + num_padding,
                    self._flat_page_indices.size(0),
                )
                self._flat_page_indices[last_real_indptr:end_idx].fill_(safe_page)
                self._flat_last_page_len[actual_flat:].fill_(1)
            else:
                torch.clamp_min_(self._flat_last_page_len, 1)
            flat_indptr = self._flat_page_indptr
            flat_indices = self._flat_page_indices
            flat_lpl = self._flat_last_page_len

        self.decode_wrapper.plan(
            flat_indptr,
            flat_indices,
            flat_lpl,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.page_size,
            q_data_type=self.datatype,
            kv_data_type=kv_dtype,
        )
        return self.fmha_params

    @staticmethod
    def support(attn_inputs: PyAttentionInputs) -> bool:
        return True

    def forward(
        self, q: torch.Tensor, kv_cache: Optional[LayerKVCache]
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required for target verify"
        assert q.dim() == 3, f"Expected q [total_tokens, H, D], got shape {q.shape}"
        paged_kv_cache = kv_cache.kv_cache_base
        if paged_kv_cache.dim() == 2:
            paged_kv_cache = common.reshape_paged_kv_cache(
                paged_kv_cache,
                self.local_kv_head_num,
                self.page_size,
                self.head_dim_qk,
            )
        return self.decode_wrapper.run(q, paged_kv_cache)


class PyFlashinferTargetVerifyDecodeImpl(PyFlashinferPrefillImplBase):
    """BatchDecode-based MTP target verify with tensor cores.

    Flattens B*Q verify tokens into individual decode requests sharing the
    same paged KV cache.  ~4-5x faster than the BatchPrefill approach for
    typical MTP verify workloads (small Q, large prefix, high GQA ratio).
    """

    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        return PyFlashinferDecodeTargetVerifyAttnOp(attn_configs, attn_inputs)

    def _create_rope_impl(self, attn_configs: AttentionConfigs) -> Any:
        if attn_configs.rope_config.style == RopeStyle.No:
            return None
        return MhaRotaryEmbeddingOp(attn_configs)

    def _prepare_fmha_input(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        return query

    def support_cuda_graph(self) -> bool:
        return True

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        if not attn_inputs.is_prefill:
            return False
        if not getattr(attn_inputs, "is_target_verify", False):
            return False
        if attn_configs.use_mla:
            return False
        if is_sm_100():
            return False
        return True


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
        flashinfer_decode_params = fill_mla_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id_host,
            self.seq_size_per_block,
        )
        # Get torch.dtype from attention configs
        self.decode_wrapper.plan(
            flashinfer_decode_params.decode_page_indptr_d,
            flashinfer_decode_params.page_indice_d,
            flashinfer_decode_params.paged_kv_last_page_len_d,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.seq_size_per_block,
            q_data_type=get_scalar_type(attn_inputs.dtype),
            kv_data_type=kv_datatype,
        )
        return flashinfer_decode_params

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
        self.rope_impl = FusedRopeKVCacheDecodeOp(attn_configs)
        self.attn_configs = attn_configs

        # Store input info
        self.attn_inputs = attn_inputs

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_impl.prepare(attn_inputs)
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
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            qkv = self.rope_impl.forward(qkv, kv_cache, self.rope_params)

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(qkv, kv_cache, self.fmha_params)
