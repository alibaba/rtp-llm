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
    KVCache,
    ParamsBase,
    PyAttentionInputs,
    fill_mla_params,
    get_scalar_type,
    rtp_llm_ops,
)

# Constants
DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_MB = 512

# Global workspace buffer pool
_g_py_flashinfer_workspace_pool: list[torch.Tensor] = []
_g_py_flashinfer_pool_lock = __import__("threading").Lock()


def get_py_flashinfer_workspace_buffer(device: str = "cuda:0") -> torch.Tensor:
    """Get a PyFlashInfer workspace buffer from the pool.

    This function manages a pool of workspace buffers to support multiple
    concurrent instances while avoiding excessive memory allocation.

    Args:
        device: CUDA device to allocate buffer on (default: "cuda:0")

    Returns:
        Workspace buffer tensor of size DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_MB
    """
    with _g_py_flashinfer_pool_lock:
        if _g_py_flashinfer_workspace_pool:
            return _g_py_flashinfer_workspace_pool.pop()
        else:
            # No available buffer in pool, create a new one
            return torch.zeros(
                DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_MB * 1024 * 1024,
                dtype=torch.uint8,
                device=device,
            )


def release_py_flashinfer_workspace_buffer(buffer: torch.Tensor) -> None:
    """Release a PyFlashInfer workspace buffer back to the pool.

    Args:
        buffer: The workspace buffer to release
    """
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
        self.page_size = attn_configs.tokens_per_block
        self.datatype = attn_configs.dtype
        self.max_seq_len = attn_configs.max_seq_len
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.enable_cuda_graph = attn_inputs.is_cuda_graph
        # Use Paged KV Cache wrapper
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.g_workspace_buffer,
            "HND",
            backend=backend,
        )

    def __del__(self):
        """Release workspace buffer back to pool when object is destroyed."""
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def set_params(self, params: Any):
        """Set the params object to be used by this op."""
        self.fmha_params = params

    def prepare(
        self,
        attn_inputs: PyAttentionInputs,
    ) -> ParamsBase:
        """
        Prepare the prefill wrapper with paged KV cache parameters

        Args:
            attn_inputs: Attention inputs containing sequence information
            paged_kv_indptr: Page count boundaries [batch_size + 1]
            paged_kv_indices: Actual page IDs [total_pages]
            paged_kv_last_page_len: Valid length of last page [batch_size]
        """
        check_attention_inputs(attn_inputs)
        qo_indptr = attn_inputs.cu_seqlens[: attn_inputs.input_lengths.size(0) + 1]
        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
            self.page_size,
        )

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
            self.prefill_wrapper._fixed_batch_size = len(qo_indptr) - 1

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
            kv_data_type=self.datatype,
        )
        return self.fmha_params

    @staticmethod
    def support(attn_inputs: PyAttentionInputs) -> bool:
        return True

    def forward(self, q: torch.Tensor, kv_cache: Optional[KVCache]) -> torch.Tensor:
        """
        Forward pass with paged KV cache

        Args:
            q: Query tensor [total_tokens, num_heads, head_dim]
            kv_cache: Paged KV cache [num_pages, 2, page_size, kv_heads, head_dim]
            params: Parameters (not used currently)

        Returns:
            output: [total_tokens, num_heads, head_dim]
        """
        assert kv_cache is not None, "kv_cache is required for paged attention"
        assert (
            q.dim() == 3
        ), f"Expected q to be 3D tensor [total_tokens, num_heads, head_dim], got {q.dim()}D"

        result = self.prefill_wrapper.run(q, kv_cache.kv_cache_base)

        return result


class PyFlashinferPrefillAttnOp(object):
    def __init__(self, attn_configs: AttentionConfigs, backend: str = "auto") -> None:
        self.g_workspace_buffer = get_py_flashinfer_workspace_buffer()
        # attn_configs.head_num and kv_head_num are already divided by tp_size in ModelConfig::getAttentionConfigs
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        self.page_size = attn_configs.tokens_per_block
        # TODO: maybe use v_head_dim
        self.head_dim_vo = attn_configs.size_per_head
        self.prefill_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self.g_workspace_buffer,
            backend=backend,
        )
        self.datatype = attn_configs.dtype
        self.params = None

    def __del__(self):
        """Release workspace buffer back to pool when object is destroyed."""
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

        self.params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
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
    def forward(self, qkv: torch.Tensor, kv_cache: Optional[KVCache]) -> torch.Tensor:
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
        max_seq_len: int,
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
            token_per_block=attn_configs.tokens_per_block,
        )
        self.create_params(attn_inputs)
        self.fmha_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        self.fmha_impl.prepare(attn_inputs)

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
        kv_cache: Optional[KVCache],
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
    def support(
        attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs, max_seq_len: int
    ) -> bool:
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
    def support(
        attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs, max_seq_len: int
    ) -> bool:
        """Check if ragged prefill implementation is supported.

        Returns True if:
        1. Not running on SM 10.0 (Blackwell) architecture
        2. The underlying ragged FMHA op supports the inputs
           (requires prefix_lengths to be empty or zero)
        3. MhaRotaryEmbeddingOp supports the inputs
        """
        return not is_sm_100() and PyFlashinferPrefillAttnOp.support(attn_inputs)


def determine_use_tensor_core_from_configs(attn_configs: AttentionConfigs) -> bool:
    """Determine whether to use tensor cores based on attention configs."""
    # Use tensor cores for larger head dimensions and when kv_head_num matches requirements
    return attn_configs.head_num // attn_configs.kv_head_num >= 4


class PyFlashinferDecodeAttnOp(object):
    def __init__(self, attn_configs: AttentionConfigs) -> None:
        self.g_workspace_buffer = get_py_flashinfer_workspace_buffer()
        # attn_configs already has head_num and kv_head_num divided by tp_size
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        self.head_dim_vo = attn_configs.size_per_head
        self.seq_size_per_block = attn_configs.tokens_per_block
        self.use_tensor_core = determine_use_tensor_core_from_configs(attn_configs)
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.g_workspace_buffer,
            "HND",
            use_tensor_cores=self.use_tensor_core,
        )
        self.kv_cache_dtype = attn_configs.kv_cache_dtype

    def __del__(self):
        """Release workspace buffer back to pool when object is destroyed."""
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
            attn_inputs.kv_cache_block_id_host,
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
        self, q: torch.Tensor, kv_cache: Optional[KVCache], params: ParamsBase
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required"
        q = q.reshape(q.shape[0], self.local_head_num, self.head_dim_qk)
        return self.decode_wrapper.run(q, kv_cache.kv_cache_base)


class PyFlashinferDecodeImpl(FMHAImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        max_seq_len: int,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = PyFlashinferDecodeAttnOp(attn_configs)
        self.rope_impl = FusedRopeKVCacheDecodeOp(attn_configs, max_seq_len)
        self.attn_configs = attn_configs

        # Store input info
        self.attn_inputs = attn_inputs

        # Create params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        max_seq_len: int,
    ) -> bool:
        return not attn_configs.use_mla

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
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
