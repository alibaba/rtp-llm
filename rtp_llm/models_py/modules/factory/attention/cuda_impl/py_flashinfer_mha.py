from typing import Any, Optional

import flashinfer
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
DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_MB = 256

# Global workspace buffer pool
_g_py_flashinfer_workspace_pool: list[torch.Tensor] = []
_g_py_flashinfer_pool_lock = __import__("threading").Lock()
# Empty prefix tensor for decode paths where prefix_lengths is None.
_EMPTY_PREFIX = torch.empty(0, dtype=torch.int32)
_NVFP4_BLOCK_SIZE = 16
_NVFP4_GLOBAL_SCALE = 1.0
_NVFP4_ATTN_KV_DTYPE = torch.float8_e4m3fn
_NVFP4_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)
_NVFP4_UNSWIZZLE_CACHE: dict[
    tuple[str, int, int, int], tuple[torch.Tensor, torch.Tensor]
] = {}
_NVFP4_DEQUANT_SHARED_CACHE: dict[tuple[str, int, int], torch.Tensor] = {}
_NVFP4_DEQUANT_SHARED_LOCK = __import__("threading").Lock()
_NVFP4_ACTIVE_PAGE_DEBUG_LOGGED = False


def _reshape_nvfp4_paged_kv_cache(
    kv_cache_base: torch.Tensor,
    local_kv_head_num: int,
    page_size: int,
    head_dim: int,
) -> torch.Tensor:
    if kv_cache_base.dim() == 5:
        return kv_cache_base
    return kv_cache_base.view(
        kv_cache_base.shape[0],
        2,
        local_kv_head_num,
        page_size,
        head_dim // 2,
    )


def _reshape_nvfp4_scale_cache(
    kv_scale_base: Optional[torch.Tensor],
    local_kv_head_num: int,
    page_size: int,
    head_dim: int,
) -> torch.Tensor:
    if kv_scale_base is None:
        raise ValueError("kv_scale_base is required for NVFP4 KV cache")
    if kv_scale_base.dtype == torch.uint8:
        kv_scale_base = kv_scale_base.view(torch.float8_e4m3fn)
    return kv_scale_base.view(
        kv_scale_base.shape[0],
        2,
        local_kv_head_num,
        page_size,
        head_dim // _NVFP4_BLOCK_SIZE,
    )


def _build_nvfp4_unswizzle_indices(
    page_size: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_key = (device.type, device.index or -1, page_size, head_dim)
    cached = _NVFP4_UNSWIZZLE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    scale_dim = head_dim // _NVFP4_BLOCK_SIZE
    if scale_dim % 4 != 0:
        raise ValueError(
            f"NVFP4 scale_dim must be divisible by 4, got {scale_dim} for head_dim={head_dim}"
        )
    t_idx = torch.arange(page_size, device=device)
    s_idx = torch.arange(scale_dim, device=device)
    t_grid, s_grid = torch.meshgrid(t_idx, s_idx, indexing="ij")
    if is_sm_100():
        s_parts = scale_dim // 4
        row_idx = (t_grid // 4) * 4 + (s_grid // s_parts)
        col_idx = (s_grid % s_parts) * 4 + (t_grid % 4)
    else:
        # SM90 fallback path stores scales in linear [T, S] layout.
        row_idx = t_grid
        col_idx = s_grid
    _NVFP4_UNSWIZZLE_CACHE[cache_key] = (row_idx, col_idx)
    return row_idx, col_idx


def _dequantize_nvfp4_cache(
    kv_cache: LayerKVCache,
    local_kv_head_num: int,
    page_size: int,
    head_dim: int,
    out_dtype: torch.dtype,
    row_idx: torch.Tensor,
    col_idx: torch.Tensor,
    lut: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    chunk_pages: int = 1,
    page_indices: Optional[torch.Tensor] = None,
    allow_flashinfer_dequant: bool = True,
) -> torch.Tensor:
    total_pages = kv_cache.kv_cache_base.shape[0]
    if total_pages <= 0:
        return torch.empty(
            (0, 2, local_kv_head_num, page_size, head_dim),
            dtype=out_dtype,
            device=kv_cache.kv_cache_base.device,
        )

    packed_cache_full = _reshape_nvfp4_paged_kv_cache(
        kv_cache.kv_cache_base, local_kv_head_num, page_size, head_dim
    )
    scale_cache_full = _reshape_nvfp4_scale_cache(
        kv_cache.kv_scale_base, local_kv_head_num, page_size, head_dim
    )
    if page_indices is not None:
        if page_indices.numel() == 0:
            return torch.empty(
                (0, 2, local_kv_head_num, page_size, head_dim),
                dtype=out_dtype,
                device=kv_cache.kv_cache_base.device,
            )
        num_pages = int(page_indices.numel())
    else:
        num_pages = total_pages
    scale_dim = head_dim // _NVFP4_BLOCK_SIZE

    expected_shape = (num_pages, 2, local_kv_head_num, page_size, head_dim)
    if (
        out is None
        or out.shape != expected_shape
        or out.dtype != out_dtype
        or out.device != packed_cache_full.device
    ):
        dequant_cache = torch.empty(
            expected_shape,
            dtype=out_dtype,
            device=packed_cache_full.device,
        )
    else:
        dequant_cache = out
    chunk_pages = max(1, min(chunk_pages, num_pages))
    use_flashinfer_dequant = (
        allow_flashinfer_dequant and not torch.cuda.is_current_stream_capturing()
    )
    inv_global_scale = None
    if use_flashinfer_dequant:
        inv_global_scale = torch.full(
            (1,),
            1.0 / _NVFP4_GLOBAL_SCALE,
            dtype=torch.float32,
            device=packed_cache_full.device,
        )
    for start in range(0, num_pages, chunk_pages):
        end = min(start + chunk_pages, num_pages)
        if page_indices is not None:
            chunk_indices = page_indices[start:end]
            packed_chunk = packed_cache_full.index_select(0, chunk_indices)
            scale_chunk_swizzled = scale_cache_full.index_select(0, chunk_indices)
        else:
            packed_chunk = packed_cache_full[start:end]
            scale_chunk_swizzled = scale_cache_full[start:end]
        if is_sm_100():
            scale_chunk = scale_chunk_swizzled[:, :, :, row_idx, col_idx]
        else:
            # SM90 stores NVFP4 scales in linear [T, S], no unswizzle needed.
            scale_chunk = scale_chunk_swizzled
        if use_flashinfer_dequant:
            assert inv_global_scale is not None
            fp4_flat = packed_chunk.reshape(-1, head_dim // 2)
            if scale_chunk.dtype == torch.uint8:
                scale_flat = scale_chunk.reshape(-1, scale_dim)
            else:
                scale_flat = scale_chunk.view(torch.uint8).reshape(-1, scale_dim)
            values = flashinfer.e2m1_and_ufp8sf_scale_to_float(
                fp4_flat,
                scale_flat.reshape(-1),
                inv_global_scale,
                _NVFP4_BLOCK_SIZE,
                1,
                False,
            ).reshape(end - start, 2, local_kv_head_num, page_size, head_dim)
        else:
            scale_chunk_f32 = (
                torch.nan_to_num(
                    scale_chunk.to(torch.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                / _NVFP4_GLOBAL_SCALE
            )
            low = packed_chunk & 0x0F
            high = (packed_chunk & 0xF0) >> 4
            nibble = torch.stack((low, high), dim=-1).reshape(
                end - start,
                2,
                local_kv_head_num,
                page_size,
                head_dim,
            )
            values = lut[(nibble & 0x7).to(torch.long)]
            values = torch.where((nibble & 0x8).bool(), -values, values)
            values = (
                values.view(
                    end - start,
                    2,
                    local_kv_head_num,
                    page_size,
                    scale_dim,
                    _NVFP4_BLOCK_SIZE,
                )
                * scale_chunk_f32.unsqueeze(-1)
            ).reshape(end - start, 2, local_kv_head_num, page_size, head_dim)
        dequant_cache[start:end] = values.to(out_dtype)

    return dequant_cache


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
        self.is_target_verify = bool(attn_inputs.is_target_verify)
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.enable_cuda_graph = attn_inputs.is_cuda_graph
        self.prefill_cuda_graph_copy_params = None
        # Pre-allocated buffers for CUDA graph copy path (avoid per-forward allocation)
        self._aligned_q_buf = None
        self._compact_out_buf = None
        self._nvfp4_row_idx: Optional[torch.Tensor] = None
        self._nvfp4_col_idx: Optional[torch.Tensor] = None
        self._nvfp4_lut: Optional[torch.Tensor] = None
        self._nvfp4_compact_page_indices: Optional[torch.Tensor] = None
        self._nvfp4_saved_page_indices: Optional[torch.Tensor] = None
        self._nvfp4_physical_page_indices: Optional[torch.Tensor] = None
        self._nvfp4_cached_active_pages: Optional[int] = None
        self._nvfp4_mode_logged = False
        if self.kv_cache_dtype == KvCacheDataType.NVFP4:
            self._nvfp4_row_idx, self._nvfp4_col_idx = _build_nvfp4_unswizzle_indices(
                self.page_size, self.head_dim_qk, self.g_workspace_buffer.device
            )
            self._nvfp4_lut = _NVFP4_E2M1_LUT.to(self.g_workspace_buffer.device)
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

        kv_data_type = (
            _NVFP4_ATTN_KV_DTYPE
            if self.kv_cache_dtype == KvCacheDataType.NVFP4
            else self.datatype
        )
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
            kv_data_type=kv_data_type,
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
        restore_indices = False
        original_indices: Optional[torch.Tensor] = None
        page_indices: Optional[torch.Tensor] = None
        if self.kv_cache_dtype == KvCacheDataType.NVFP4:
            assert self._nvfp4_row_idx is not None and self._nvfp4_col_idx is not None
            assert self._nvfp4_lut is not None
            if not self._nvfp4_mode_logged:
                self._nvfp4_mode_logged = True
                copy_max_seq = (
                    int(self.prefill_cuda_graph_copy_params.max_seq_len)
                    if self.prefill_cuda_graph_copy_params is not None
                    else -1
                )
                __import__("logging").getLogger(__name__).info(
                    "NVFP4 prefill mode: is_target_verify=%s, copy_max_seq=%d, q_tokens=%d, kv_cache_base=%s stride=%s dtype=%s, kv_scale_base=%s stride=%s dtype=%s",
                    self.is_target_verify,
                    copy_max_seq,
                    int(q.shape[0]),
                    tuple(kv_cache.kv_cache_base.shape),
                    tuple(kv_cache.kv_cache_base.stride()),
                    str(kv_cache.kv_cache_base.dtype),
                    (
                        tuple(kv_cache.kv_scale_base.shape)
                        if kv_cache.kv_scale_base is not None
                        else None
                    ),
                    (
                        tuple(kv_cache.kv_scale_base.stride())
                        if kv_cache.kv_scale_base is not None
                        else None
                    ),
                    (
                        str(kv_cache.kv_scale_base.dtype)
                        if kv_cache.kv_scale_base is not None
                        else None
                    ),
                )
            use_compact_pages = (
                self.is_target_verify or self.prefill_cuda_graph_copy_params is not None
            )
            if not use_compact_pages:
                paged_kv_cache = _dequantize_nvfp4_cache(
                    kv_cache,
                    self.local_kv_head_num,
                    self.page_size,
                    self.head_dim_qk,
                    _NVFP4_ATTN_KV_DTYPE,
                    self._nvfp4_row_idx,
                    self._nvfp4_col_idx,
                    self._nvfp4_lut,
                    chunk_pages=128,
                    allow_flashinfer_dequant=False,
                )
            else:
                in_capture = torch.cuda.is_current_stream_capturing()
                page_count = int(self.fmha_params.page_indice_d.shape[0])
                active_pages = page_count
                if page_count > 0:
                    if (
                        self.prefill_cuda_graph_copy_params is not None
                        and not self.is_target_verify
                    ):
                        active_pages = min(
                            page_count,
                            max(
                                1,
                                (self.max_seq_len + self.page_size - 1)
                                // self.page_size,
                            ),
                        )
                        self._nvfp4_cached_active_pages = active_pages
                    elif in_capture and self._nvfp4_cached_active_pages is not None:
                        active_pages = min(self._nvfp4_cached_active_pages, page_count)
                    else:
                        candidate_pages: list[int] = []
                        if self.fmha_params.decode_page_indptr_d.numel() > 0:
                            if self.prefill_cuda_graph_copy_params is not None:
                                bs = int(
                                    self.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size[
                                        0
                                    ].item()
                                )
                            else:
                                bs = (
                                    int(self.fmha_params.decode_page_indptr_d.numel())
                                    - 1
                                )
                            indptr_pos = min(
                                max(bs, 0),
                                int(self.fmha_params.decode_page_indptr_d.numel()) - 1,
                            )
                            candidate_pages.append(
                                int(
                                    self.fmha_params.decode_page_indptr_d[
                                        indptr_pos
                                    ].item()
                                )
                            )
                        if candidate_pages:
                            active_pages = min(max(min(candidate_pages), 1), page_count)
                        else:
                            active_pages = min(page_count, 1)
                        self._nvfp4_cached_active_pages = active_pages
                active_pages = max(active_pages, 0)

                if (
                    self._nvfp4_compact_page_indices is None
                    or self._nvfp4_compact_page_indices.numel() < page_count
                    or self._nvfp4_compact_page_indices.device
                    != kv_cache.kv_cache_base.device
                ):
                    if in_capture:
                        raise RuntimeError(
                            "NVFP4 compact page index buffer must be initialized before CUDA graph capture"
                        )
                    self._nvfp4_compact_page_indices = torch.arange(
                        page_count,
                        dtype=self.fmha_params.page_indice_d.dtype,
                        device=kv_cache.kv_cache_base.device,
                    )
                if (
                    self._nvfp4_saved_page_indices is None
                    or self._nvfp4_saved_page_indices.numel() < page_count
                    or self._nvfp4_saved_page_indices.device
                    != kv_cache.kv_cache_base.device
                ):
                    if in_capture:
                        raise RuntimeError(
                            "NVFP4 saved page index buffer must be initialized before CUDA graph capture"
                        )
                    self._nvfp4_saved_page_indices = torch.empty(
                        page_count,
                        dtype=self.fmha_params.page_indice_d.dtype,
                        device=kv_cache.kv_cache_base.device,
                    )
                if (
                    self._nvfp4_physical_page_indices is None
                    or self._nvfp4_physical_page_indices.numel() < page_count
                    or self._nvfp4_physical_page_indices.device
                    != kv_cache.kv_cache_base.device
                ):
                    if in_capture:
                        raise RuntimeError(
                            "NVFP4 physical page index buffer must be initialized before CUDA graph capture"
                        )
                    self._nvfp4_physical_page_indices = torch.empty(
                        page_count,
                        dtype=torch.long,
                        device=kv_cache.kv_cache_base.device,
                    )

                if active_pages > 0:
                    original_indices = self._nvfp4_saved_page_indices[:active_pages]
                    original_indices.copy_(
                        self.fmha_params.page_indice_d[:active_pages]
                    )
                    page_indices = self._nvfp4_physical_page_indices[:active_pages]
                    page_indices.copy_(original_indices)
                    self.fmha_params.page_indice_d[:active_pages].copy_(
                        self._nvfp4_compact_page_indices[:active_pages]
                    )
                    restore_indices = True
                else:
                    page_indices = self._nvfp4_physical_page_indices[:0]

                dequant_numel = (
                    active_pages
                    * 2
                    * self.local_kv_head_num
                    * self.page_size
                    * self.head_dim_qk
                )
                shared_key = (
                    kv_cache.kv_cache_base.device.type,
                    kv_cache.kv_cache_base.device.index or -1,
                    __import__("threading").get_ident(),
                )
                with _NVFP4_DEQUANT_SHARED_LOCK:
                    shared_buffer = _NVFP4_DEQUANT_SHARED_CACHE.get(shared_key)
                    if (
                        shared_buffer is None
                        or shared_buffer.dtype != _NVFP4_ATTN_KV_DTYPE
                        or shared_buffer.device != kv_cache.kv_cache_base.device
                        or shared_buffer.numel() < dequant_numel
                    ):
                        if in_capture:
                            raise RuntimeError(
                                "NVFP4 shared dequant cache must be initialized before CUDA graph capture"
                            )
                        shared_buffer = torch.empty(
                            dequant_numel,
                            dtype=_NVFP4_ATTN_KV_DTYPE,
                            device=kv_cache.kv_cache_base.device,
                        )
                        _NVFP4_DEQUANT_SHARED_CACHE[shared_key] = shared_buffer
                if active_pages > 0:
                    dequant_cache = shared_buffer[:dequant_numel].view(
                        active_pages,
                        2,
                        self.local_kv_head_num,
                        self.page_size,
                        self.head_dim_qk,
                    )
                else:
                    dequant_cache = torch.empty(
                        (
                            0,
                            2,
                            self.local_kv_head_num,
                            self.page_size,
                            self.head_dim_qk,
                        ),
                        dtype=_NVFP4_ATTN_KV_DTYPE,
                        device=kv_cache.kv_cache_base.device,
                    )
                paged_kv_cache = _dequantize_nvfp4_cache(
                    kv_cache,
                    self.local_kv_head_num,
                    self.page_size,
                    self.head_dim_qk,
                    _NVFP4_ATTN_KV_DTYPE,
                    self._nvfp4_row_idx,
                    self._nvfp4_col_idx,
                    self._nvfp4_lut,
                    out=dequant_cache,
                    chunk_pages=128,
                    page_indices=page_indices,
                )
        elif paged_kv_cache.dim() == 2:
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

            try:
                result = self.prefill_wrapper.run(q_aligned, paged_kv_cache)
            finally:
                if (
                    restore_indices
                    and original_indices is not None
                    and page_indices is not None
                ):
                    active_pages = int(page_indices.numel())
                    self.fmha_params.page_indice_d[:active_pages].copy_(
                        original_indices
                    )

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
            return self._compact_out_buf.view(token_num, head_num, head_size)
        # No CUDA graph copy, direct execution
        try:
            return self.prefill_wrapper.run(q, paged_kv_cache)
        finally:
            if (
                restore_indices
                and original_indices is not None
                and page_indices is not None
            ):
                active_pages = int(page_indices.numel())
                self.fmha_params.page_indice_d[:active_pages].copy_(original_indices)


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
        check_attention_inputs(attn_inputs)
        batch_size = attn_inputs.input_lengths.size(0)
        cu_seqlens = attn_inputs.cu_seqlens[: batch_size + 1]

        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id_host,
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
        if attn_configs.kv_cache_dtype == KvCacheDataType.NVFP4:
            from rtp_llm.models_py.modules.factory.attention.cuda_impl.trtllm_gen import (
                NVFP4KVCacheWriteOp,
            )

            self.kv_cache_write_op = NVFP4KVCacheWriteOp(
                num_kv_heads=attn_configs.kv_head_num,
                head_size=attn_configs.size_per_head,
                token_per_block=attn_configs.kernel_tokens_per_block,
            )
        else:
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
        1. Not running on SM 10.0 (Blackwell) architecture, unless NVFP4 needs
           this path as a fallback because TRTLLM-GEN prefill is unavailable
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
        1. Not running on SM 10.0 (Blackwell) architecture, unless NVFP4 needs
           this path as a fallback because TRTLLM-GEN prefill is unavailable
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
        __import__("logging").getLogger(__name__).info(
            "Init PyFlashinferDecodeAttnOp: kv_cache_dtype=%s, seq_block=%d, kv_heads=%d, head_dim=%d",
            str(self.kv_cache_dtype),
            int(self.seq_size_per_block),
            int(self.local_kv_head_num),
            int(self.head_dim_qk),
        )
        self._nvfp4_row_idx: Optional[torch.Tensor] = None
        self._nvfp4_col_idx: Optional[torch.Tensor] = None
        self._nvfp4_lut: Optional[torch.Tensor] = None
        self._nvfp4_compact_page_indices: Optional[torch.Tensor] = None
        self._nvfp4_saved_page_indices: Optional[torch.Tensor] = None
        self._nvfp4_physical_page_indices: Optional[torch.Tensor] = None
        self._nvfp4_cached_active_pages: Optional[int] = None
        if self.kv_cache_dtype == KvCacheDataType.NVFP4:
            self._nvfp4_row_idx, self._nvfp4_col_idx = _build_nvfp4_unswizzle_indices(
                self.seq_size_per_block,
                self.head_dim_qk,
                self.g_workspace_buffer.device,
            )
            self._nvfp4_lut = _NVFP4_E2M1_LUT.to(self.g_workspace_buffer.device)

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def set_params(self, params: rtp_llm_ops.FlashInferMlaAttnParams) -> None:
        """Set the params object to be used by this op."""
        self.fmha_params = params

    def prepare(
        self,
        attn_inputs: PyAttentionInputs,
        forbid_realloc: bool = False,
    ) -> ParamsBase:
        """
        Prepare the decode wrapper with paged KV cache parameters.

        forbid_realloc: True only when called from prepare_cuda_graph (replay); forbids buffer realloc.
        """
        # Convert kv_cache_dtype to torch dtype
        if self.kv_cache_dtype == KvCacheDataType.INT8:
            kv_datatype = torch.int8
        elif self.kv_cache_dtype == KvCacheDataType.FP8:
            kv_datatype = torch.float8_e4m3fn
        elif self.kv_cache_dtype == KvCacheDataType.NVFP4:
            kv_datatype = _NVFP4_ATTN_KV_DTYPE
        else:  # BASE
            kv_datatype = get_scalar_type(attn_inputs.dtype)
        prefix_lengths = (
            attn_inputs.prefix_lengths
            if attn_inputs.prefix_lengths is not None
            else _EMPTY_PREFIX
        )
        self.fmha_params = fill_mla_params(
            prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id_host,
            self.seq_size_per_block,
            forbid_realloc=forbid_realloc,
        )

        if self.enable_cuda_graph and self.decode_wrapper._fixed_batch_size == 0:
            batch_size = attn_inputs.input_lengths.size(0)
            self.decode_wrapper._use_cuda_graph = True
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
        """Update buffer contents for CUDA graph replay without calling plan().

        During CUDA graph replay, we must NOT call plan() because it may launch
        GPU kernels on the current stream while the graph replays on the capture
        stream, causing a race condition. We only need to update the page table
        buffers in-place via fill_params — the pre-allocated buffers are already
        wired into the decode_wrapper from the initial prepare() call.
        """
        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id_host,
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
        if self.kv_cache_dtype == KvCacheDataType.NVFP4:
            assert self._nvfp4_row_idx is not None and self._nvfp4_col_idx is not None
            assert self._nvfp4_lut is not None
            in_capture = torch.cuda.is_current_stream_capturing()
            page_count = params.page_indice_d.shape[0]
            if page_count > 0:
                if in_capture and self._nvfp4_cached_active_pages is not None:
                    active_pages = min(self._nvfp4_cached_active_pages, page_count)
                else:
                    # Warmup path (non-capture): derive active page count once and cache for capture.
                    # Prefer tighter estimates than decode_page_indptr_d[-1], which may include
                    # preallocated tail pages in CUDA-graph setup.
                    candidate_pages: list[int] = []
                    if params.decode_page_indptr_d.numel() > 0:
                        bs = int(q.shape[0])
                        indptr_pos = min(
                            max(bs, 0),
                            int(params.decode_page_indptr_d.numel()) - 1,
                        )
                        candidate_pages.append(
                            int(params.decode_page_indptr_d[indptr_pos].item())
                        )
                    if hasattr(params, "kvlen_d"):
                        kvlen_d = params.kvlen_d
                        if kvlen_d is not None and kvlen_d.numel() > 0:
                            kv_bs = min(int(q.shape[0]), int(kvlen_d.numel()))
                            if kv_bs > 0:
                                kv_pages = (
                                    (kvlen_d[:kv_bs] + self.seq_size_per_block - 1)
                                    // self.seq_size_per_block
                                ).sum()
                                candidate_pages.append(int(kv_pages.item()))
                    if candidate_pages:
                        active_pages = min(max(min(candidate_pages), 1), page_count)
                    else:
                        active_pages = min(page_count, 1)
                    global _NVFP4_ACTIVE_PAGE_DEBUG_LOGGED
                    if not _NVFP4_ACTIVE_PAGE_DEBUG_LOGGED:
                        _NVFP4_ACTIVE_PAGE_DEBUG_LOGGED = True
                        __import__("logging").getLogger(__name__).info(
                            "NVFP4 active page estimate: q_bs=%d page_count=%d candidates=%s selected=%d indptr_numel=%d",
                            int(q.shape[0]),
                            int(page_count),
                            candidate_pages,
                            int(active_pages),
                            int(params.decode_page_indptr_d.numel()),
                        )
                    self._nvfp4_cached_active_pages = active_pages
                active_pages = max(active_pages, 1)
            else:
                active_pages = 0

            expected_shape = (
                active_pages,
                2,
                self.local_kv_head_num,
                self.seq_size_per_block,
                self.head_dim_qk,
            )
            dequant_numel = (
                active_pages
                * 2
                * self.local_kv_head_num
                * self.seq_size_per_block
                * self.head_dim_qk
            )
            shared_key = (
                kv_cache.kv_cache_base.device.type,
                kv_cache.kv_cache_base.device.index or -1,
                __import__("threading").get_ident(),
            )
            with _NVFP4_DEQUANT_SHARED_LOCK:
                shared_buffer = _NVFP4_DEQUANT_SHARED_CACHE.get(shared_key)
                if (
                    shared_buffer is None
                    or shared_buffer.dtype != _NVFP4_ATTN_KV_DTYPE
                    or shared_buffer.device != kv_cache.kv_cache_base.device
                    or shared_buffer.numel() < dequant_numel
                ):
                    if in_capture:
                        raise RuntimeError(
                            "NVFP4 shared dequant cache must be initialized before CUDA graph capture"
                        )
                    shared_buffer = torch.empty(
                        dequant_numel,
                        dtype=_NVFP4_ATTN_KV_DTYPE,
                        device=kv_cache.kv_cache_base.device,
                    )
                    _NVFP4_DEQUANT_SHARED_CACHE[shared_key] = shared_buffer
            dequant_cache = shared_buffer[:dequant_numel].view(expected_shape)
            if (
                self._nvfp4_compact_page_indices is None
                or self._nvfp4_compact_page_indices.numel() < page_count
                or self._nvfp4_compact_page_indices.device
                != kv_cache.kv_cache_base.device
            ):
                if in_capture:
                    raise RuntimeError(
                        "NVFP4 compact page index buffer must be initialized before CUDA graph capture"
                    )
                self._nvfp4_compact_page_indices = torch.arange(
                    page_count,
                    dtype=params.page_indice_d.dtype,
                    device=kv_cache.kv_cache_base.device,
                )
            if (
                self._nvfp4_saved_page_indices is None
                or self._nvfp4_saved_page_indices.numel() < page_count
                or self._nvfp4_saved_page_indices.device
                != kv_cache.kv_cache_base.device
            ):
                if in_capture:
                    raise RuntimeError(
                        "NVFP4 saved page index buffer must be initialized before CUDA graph capture"
                    )
                self._nvfp4_saved_page_indices = torch.empty(
                    page_count,
                    dtype=params.page_indice_d.dtype,
                    device=kv_cache.kv_cache_base.device,
                )
            if (
                self._nvfp4_physical_page_indices is None
                or self._nvfp4_physical_page_indices.numel() < page_count
                or self._nvfp4_physical_page_indices.device
                != kv_cache.kv_cache_base.device
            ):
                if in_capture:
                    raise RuntimeError(
                        "NVFP4 physical page index buffer must be initialized before CUDA graph capture"
                    )
                self._nvfp4_physical_page_indices = torch.empty(
                    page_count,
                    dtype=torch.long,
                    device=kv_cache.kv_cache_base.device,
                )

            original_indices = self._nvfp4_saved_page_indices[:active_pages]
            original_indices.copy_(params.page_indice_d[:active_pages])
            physical_indices = self._nvfp4_physical_page_indices[:active_pages]
            physical_indices.copy_(original_indices)
            params.page_indice_d[:active_pages].copy_(
                self._nvfp4_compact_page_indices[:active_pages]
            )
            paged_kv_cache = _dequantize_nvfp4_cache(
                kv_cache,
                self.local_kv_head_num,
                self.seq_size_per_block,
                self.head_dim_qk,
                _NVFP4_ATTN_KV_DTYPE,
                self._nvfp4_row_idx,
                self._nvfp4_col_idx,
                self._nvfp4_lut,
                out=dequant_cache,
                chunk_pages=128,
                page_indices=physical_indices,
            )
            output = self.decode_wrapper.run(q, paged_kv_cache)
            params.page_indice_d[:active_pages].copy_(original_indices)
            return output
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
        """Prepare for CUDA graph replay; only updates buffer contents, no plan()."""
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

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_impl.prepare(attn_inputs)
