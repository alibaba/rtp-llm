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
from rtp_llm.models_py.modules.factory.attention.cuda_impl.utils import is_sm10x
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
        if self.kv_cache_dtype == KvCacheDataType.FP8:
            self.kv_datatype = torch.float8_e4m3fn
        else:
            self.kv_datatype = self.datatype
        self.max_seq_len = attn_configs.max_seq_len
        self.is_causal = attn_configs.is_causal
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
        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id,
            self.page_size,
            forbid_realloc,
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
            causal=self.is_causal,
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
        kv_block_id_host = attn_inputs.kv_cache_kernel_block_id
        if kv_block_id_host is None:
            kv_block_id_host = torch.empty(0, dtype=torch.int32)

        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            kv_block_id_host,
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
            q_data_type=get_scalar_type(attn_inputs.dtype),
        )
        return self.fmha_params

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
        self.kv_cache_dtype = attn_configs.kv_cache_dtype
        self.enable_cuda_graph = attn_inputs.is_cuda_graph
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def set_params(self, params: rtp_llm_ops.FlashInferMlaAttnParams) -> None:
        """Set the params object to be used by this op."""
        self.fmha_params = params

    def _get_kv_data_type(self, attn_inputs: PyAttentionInputs) -> torch.dtype:
        if self.kv_cache_dtype == KvCacheDataType.FP8:
            return torch.float8_e4m3fn
        return get_scalar_type(attn_inputs.dtype)

    def _requires_cuda_graph_replan(self) -> bool:
        # FlashInfer plan_info is derived from runtime page tables/kv lengths.
        # CUDA graph replay keeps wrapper buffer addresses fixed, but their
        # contents and derived plan metadata must be refreshed before replay.
        return True

    def _plan_decode_wrapper(self, attn_inputs: PyAttentionInputs) -> None:
        if self.use_tensor_core:
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
            q_data_type=get_scalar_type(attn_inputs.dtype),
            kv_data_type=self._get_kv_data_type(attn_inputs),
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
        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id,
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
        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id,
            self.seq_size_per_block,
            forbid_realloc=True,
        )
        if self._requires_cuda_graph_replan():
            self._plan_decode_wrapper(attn_inputs)

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
