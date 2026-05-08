from typing import NamedTuple, Optional, Tuple

import flashinfer
import flashinfer.page as page
import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
    MhaRotaryEmbeddingOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.utils import is_sm_100
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
    FusedRopeKVCachePrefillOpQOut,
    LayerKVCache,
    PyAttentionInputs,
    get_scalar_type,
    rtp_llm_ops,
)

# Empty int32 tensor used when prefix_lengths is None (normal decode path).
# The C++ fill_params checks .defined() && .size(0) > 0, so an empty tensor
# is treated as "no prefix" and falls back to sequence_lengths + 1.
_EMPTY_PREFIX = torch.empty(0, dtype=torch.int32)
_EMPTY_BLOCK_TABLE = torch.empty(0, dtype=torch.int32)


def _is_zero_prefix_prefill(attention_inputs: PyAttentionInputs) -> bool:
    prefix_lengths = attention_inputs.prefix_lengths
    return (
        prefix_lengths is None
        or prefix_lengths.numel() <= 0
        or prefix_lengths.sum().item() == 0
    )


def _get_block_table_host(attention_inputs: PyAttentionInputs) -> torch.Tensor:
    if attention_inputs.kv_cache_kernel_block_id_host is not None:
        return attention_inputs.kv_cache_kernel_block_id_host
    return _EMPTY_BLOCK_TABLE.to(
        device=attention_inputs.input_lengths.device,
        dtype=torch.int32,
    )


# Constants
DEFAULT_TRT_WORKSPACE_SIZE_MB = (
    512  # Memory workspace size in MB, todo(Yingyi): read from config
)

# Global workspace buffer pool
_g_trt_workspace_pool: list[torch.Tensor] = []
_g_trt_pool_lock = __import__("threading").Lock()


def get_trt_workspace_buffer(device: str = "cuda") -> torch.Tensor:
    """Get a TRT workspace buffer from the pool.

    This function manages a pool of workspace buffers to support multiple
    concurrent instances while avoiding excessive memory allocation.

    Args:
        device: CUDA device to allocate buffer on (default: "cuda")

    Returns:
        Workspace buffer tensor of size DEFAULT_TRT_WORKSPACE_SIZE_MB
    """
    with _g_trt_pool_lock:
        if _g_trt_workspace_pool:
            return _g_trt_workspace_pool.pop()
        else:
            # No available buffer in pool, create a new one
            return torch.zeros(
                DEFAULT_TRT_WORKSPACE_SIZE_MB * 1024 * 1024,
                dtype=torch.uint8,
                device=device,
            )


def release_trt_workspace_buffer(buffer: torch.Tensor) -> None:
    """Release a TRT workspace buffer back to the pool.

    Args:
        buffer: The workspace buffer to release
    """
    with _g_trt_pool_lock:
        _g_trt_workspace_pool.append(buffer)


# ---------------------------------------------------------------------------
# NVFP4 KV Cache Write Op
# ---------------------------------------------------------------------------

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0
NVFP4_BLOCK_SIZE = 16
_E2M1_THRESHOLDS = torch.tensor([0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])

# Global scale factor for online NVFP4 KV cache quantization.
#
# Following the reference formula from nvfp4_quant.py:
#   global_sf = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
#
# For online inference we cannot compute per-tensor amax, so we use a fixed
# estimate. With estimated_amax = FLOAT8_E4M3_MAX (= 448), the formula gives:
#   global_sf = 448 * 6 / 448 = 6.0 = FLOAT4_E2M1_MAX
#
# With global_sf=1.0: block scale = vec_max / E2M1_MAX = vec_max / 6.0
# This keeps scale values small, safe for |kv_values| up to E4M3_MAX * E2M1_MAX (2688).
# The dequantized value = fp4_val * scale_fp8 / global_sf is mathematically independent
# of global_sf (it cancels), but different values give different fp8 rounding behavior.
_DEFAULT_NVFP4_GLOBAL_SF = 1.0


def _quantize_nvfp4_linear_fallback(
    x_2d: torch.Tensor,
    global_sf: torch.Tensor,
    thresholds: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SM90 fallback NVFP4 quantization for linear (non-swizzled) SF layout.

    flashinfer.fp4_quantize relies on SM100+ device code paths. For SM90 we use
    a numerically stable Python fallback that keeps the same storage contract:
      - packed fp4 in E2M1x2 uint8
      - per-16-element block scale in float8_e4m3fn
    """
    if x_2d.shape[-1] % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"head_size must be divisible by {NVFP4_BLOCK_SIZE}, got {x_2d.shape[-1]}"
        )
    device = x_2d.device
    if thresholds is None:
        thresholds = _E2M1_THRESHOLDS.to(device=device, dtype=torch.float32)
    else:
        thresholds = thresholds.to(device=device, dtype=torch.float32)
    global_scale = global_sf.to(device=device, dtype=torch.float32).reshape(1)

    x_groups = x_2d.view(x_2d.shape[0], -1, NVFP4_BLOCK_SIZE)
    vec_max = x_groups.abs().amax(dim=-1)

    sf_value = global_scale * (vec_max / FLOAT4_E2M1_MAX)
    sf_narrow = sf_value.to(torch.float8_e4m3fn)
    sf_narrow_f32 = sf_narrow.to(torch.float32)

    output_scale = torch.where(
        sf_narrow_f32 > 0,
        global_scale / sf_narrow_f32,
        torch.zeros_like(sf_narrow_f32),
    )
    x_q = torch.clamp(
        x_groups * output_scale.unsqueeze(-1), -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX
    )

    # Match TensorRT-LLM floatToE2M1 tie-to-even behavior.
    x_abs = x_q.abs().to(torch.float32)
    e2m1_idx = torch.zeros_like(x_abs, dtype=torch.int64)
    for i in range(7, 0, -1):
        cond = (x_abs > thresholds[i]) | ((x_abs == thresholds[i]) & (i % 2 == 0))
        e2m1_idx = torch.where(cond & (e2m1_idx == 0), i, e2m1_idx)

    sign_bit = (x_q < 0).to(torch.int64) << 3
    e2m1_code = (e2m1_idx | sign_bit).to(torch.uint8)
    packed = (e2m1_code[..., 0::2] | (e2m1_code[..., 1::2] << 4)).contiguous()
    return packed.view(x_2d.shape[0], -1), sf_narrow.view(x_2d.shape[0], -1)


class NVFP4KVCacheWriteOp:
    """Operator for quantizing K/V to NVFP4 and writing to paged KV cache with swizzled scales.

    Quantization follows the reference formula (see nvfp4_quant.py ref_nvfp4_quant):
      1. Per block of 16 elements, compute vec_max = max(|block|)
      2. scale = global_sf * (vec_max / FLOAT4_E2M1_MAX)  →  cast to fp8_e4m3fn
      3. output_scale = global_sf / scale_fp8
      4. quantized = clamp(x * output_scale, -6, 6)  →  round to fp4

    The FMHA kernel dequantizes with: x ≈ fp4 * scale_fp8 * (1 / global_sf)
    """

    def __init__(
        self,
        num_kv_heads: int,
        head_size: int,
        token_per_block: int,
    ) -> None:
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.token_per_block = token_per_block
        self.scale_dim = head_size // NVFP4_BLOCK_SIZE
        self.params = None

        # Global scale factor tensors (on GPU, created lazily)
        self._k_global_sf: Optional[torch.Tensor] = None
        self._v_global_sf: Optional[torch.Tensor] = None
        self._e2m1_thresholds: Optional[torch.Tensor] = None
        # kv_global_scale = (1/k_global_sf, 1/v_global_sf) for FMHA kernel
        self.kv_global_scale: Tuple[float, float] = (
            1.0 / _DEFAULT_NVFP4_GLOBAL_SF,
            1.0 / _DEFAULT_NVFP4_GLOBAL_SF,
        )

        # Pre-compute index tables for scale writing.
        # SM100 trtllm-gen MHA kernel expects swizzled scale layout (HND):
        #   [P, H, T//4, 4, 4, S//4] → permute(0,1,2,4,5,3) → [P, H, T, S]
        # Non-SM100 python-attention fallback reads linear scales, so keep [T, S].
        T = token_per_block
        S = self.scale_dim
        t_idx = torch.arange(T)
        s_idx = torch.arange(S)
        t_grid, s_grid = torch.meshgrid(t_idx, s_idx, indexing="ij")  # [T, S]
        if is_sm_100():
            s_parts = S // 4
            self._swizzle_rows = (t_grid // 4) * 4 + s_grid // s_parts  # [T, S]
            self._swizzle_cols = (s_grid % s_parts) * 4 + t_grid % 4  # [T, S]
        else:
            self._swizzle_rows = t_grid
            self._swizzle_cols = s_grid

    def _ensure_global_sf(self, device: torch.device) -> None:
        """Lazily create global scale factor tensors on the right device."""
        if self._k_global_sf is None or self._k_global_sf.device != device:
            self._k_global_sf = torch.tensor(
                [_DEFAULT_NVFP4_GLOBAL_SF], dtype=torch.float32, device=device
            )
            self._v_global_sf = torch.tensor(
                [_DEFAULT_NVFP4_GLOBAL_SF], dtype=torch.float32, device=device
            )
            self._e2m1_thresholds = _E2M1_THRESHOLDS.to(
                device=device, dtype=torch.float32
            )
            self._swizzle_rows = self._swizzle_rows.to(device)
            self._swizzle_cols = self._swizzle_cols.to(device)

    def set_params(self, params):
        """Set the FlashInferMlaAttnParams to be used by this op."""
        self.params = params

    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
    ) -> None:
        """Quantize key/value to NVFP4 and write to paged KV cache.

        Args:
            key: Key tensor [total_tokens, num_kv_heads, head_dim] bf16/fp16
            value: Value tensor [total_tokens, num_kv_heads, head_dim] bf16/fp16
            kv_cache: KV cache with:
                kv_cache_base: [num_pages, 2, H, T, D//2] uint8 (FP4 packed)
                kv_scale_base: [num_pages, 2, H, T, D//16] float8_e4m3fn (swizzled)
        """
        if kv_cache is None:
            return

        device = key.device
        self._ensure_global_sf(device)
        total_tokens = key.shape[0]

        # 1. Quantize key and value to FP4 + block scales
        k_2d = key.reshape(-1, self.head_size).contiguous()  # [N*H, D]
        v_2d = value.reshape(-1, self.head_size).contiguous()

        if is_sm_100():
            k_fp4, k_sf = flashinfer.fp4_quantize(
                k_2d,
                self._k_global_sf,
                sf_vec_size=NVFP4_BLOCK_SIZE,
                is_sf_swizzled_layout=False,
            )
            v_fp4, v_sf = flashinfer.fp4_quantize(
                v_2d,
                self._v_global_sf,
                sf_vec_size=NVFP4_BLOCK_SIZE,
                is_sf_swizzled_layout=False,
            )
        else:
            k_fp4, k_sf = _quantize_nvfp4_linear_fallback(
                k_2d, self._k_global_sf, self._e2m1_thresholds
            )
            v_fp4, v_sf = _quantize_nvfp4_linear_fallback(
                v_2d, self._v_global_sf, self._e2m1_thresholds
            )

        # Reshape to [total_tokens, H, ...]
        k_fp4 = k_fp4.view(torch.uint8).reshape(
            total_tokens, self.num_kv_heads, self.head_size // 2
        )
        v_fp4 = v_fp4.view(torch.uint8).reshape(
            total_tokens, self.num_kv_heads, self.head_size // 2
        )
        if k_sf.dtype == torch.uint8:
            k_sf = k_sf.view(torch.float8_e4m3fn)
        if v_sf.dtype == torch.uint8:
            v_sf = v_sf.view(torch.float8_e4m3fn)
        k_sf = k_sf.reshape(total_tokens, self.num_kv_heads, self.scale_dim)
        v_sf = v_sf.reshape(total_tokens, self.num_kv_heads, self.scale_dim)

        # 2. Write FP4 data to cache using append_paged_kv_cache
        if kv_cache.kv_cache_base.dim() == 5:
            kv_cache_base = kv_cache.kv_cache_base
        else:
            kv_cache_base = kv_cache.kv_cache_base.view(
                kv_cache.kv_cache_base.shape[0],
                2,
                self.num_kv_heads,
                self.token_per_block,
                self.head_size // 2,
            )
        k_cache_fp4 = kv_cache_base[:, 0]
        v_cache_fp4 = kv_cache_base[:, 1]

        page.append_paged_kv_cache(
            k_fp4,
            v_fp4,
            self.params.batch_indice_d,
            self.params.positions_d,
            (k_cache_fp4, v_cache_fp4),
            self.params.page_indice_d,
            self.params.decode_page_indptr_d,
            self.params.paged_kv_last_page_len_d,
            "HND",
        )

        # 3. Write swizzled block scales to cache
        self._write_swizzled_scales(k_sf, v_sf, kv_cache)

    def _write_swizzled_scales(
        self,
        k_sf: torch.Tensor,
        v_sf: torch.Tensor,
        kv_cache: LayerKVCache,
    ) -> None:
        """Write block scales with swizzle transformation to the scale cache.

        The SM100 trtllm-gen MHA kernel expects scales in a swizzled layout.
        For HND: [P, H, T//4, 4, 4, S//4] → permute(0,1,2,4,5,3) → [P, H, T, S]
        """
        total_tokens = k_sf.shape[0]
        T = self.token_per_block
        S = self.scale_dim
        H = self.num_kv_heads
        device = k_sf.device

        # Compute physical page index and position within page for each token
        positions = self.params.positions_d  # [total_tokens]
        batch_indices = self.params.batch_indice_d  # [total_tokens]
        page_indices = self.params.page_indice_d  # flat page indices
        page_indptr = self.params.decode_page_indptr_d  # [batch_size+1]

        pos_in_page = positions % T  # [total_tokens]
        logical_page = positions // T  # [total_tokens]

        # Get physical page for each token
        batch_page_starts = page_indptr[batch_indices]  # [total_tokens]
        physical_pages = page_indices[
            batch_page_starts + logical_page
        ]  # [total_tokens]

        # Get swizzled row/col for each token's position
        # swizzle_rows/cols: [T, S] pre-computed
        token_swizzle_rows = self._swizzle_rows[pos_in_page]  # [total_tokens, S]
        token_swizzle_cols = self._swizzle_cols[pos_in_page]  # [total_tokens, S]

        # Reshape scale cache to [num_pages, 2, H, T, S]
        # For NVFP4, scale is stored as uint8, view as float8_e4m3fn
        scale_base = kv_cache.kv_scale_base
        if scale_base.dtype == torch.uint8:
            scale_base = scale_base.view(torch.float8_e4m3fn)
        kv_scale_base = scale_base.view(
            kv_cache.kv_scale_base.shape[0],
            2,
            H,
            T,
            S,
        )

        # Build index tensors for scatter write: [total_tokens, H, S]
        page_idx = physical_pages[:, None, None].expand(total_tokens, H, S)
        head_idx = torch.arange(H, device=device)[None, :, None].expand(
            total_tokens, H, S
        )
        row_idx = token_swizzle_rows[:, None, :].expand(total_tokens, H, S)
        col_idx = token_swizzle_cols[:, None, :].expand(total_tokens, H, S)

        # Write K scales (kv_dim=0) and V scales (kv_dim=1)
        kv_scale_base[page_idx, 0, head_idx, row_idx, col_idx] = k_sf
        kv_scale_base[page_idx, 1, head_idx, row_idx, col_idx] = v_sf


class FlashInferTRTLLMParams(object):
    def __init__(
        self,
        batch_size: int,
        max_q_len: int = 0,
        max_kv_len: int = 0,
        max_seq_len: int = 0,
        seq_lens: Optional[torch.Tensor] = None,
        input_lens: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_kv_seqlens: Optional[torch.Tensor] = None,
        kv_global_scale: Optional[Tuple[float, float]] = None,
        use_ragged_prefill: bool = False,
    ):

        self.batch_size = batch_size
        self.max_q_len = max_q_len  # for prefill
        self.max_kv_len = max_kv_len  # for prefill
        self.max_seq_len = max_seq_len  # for decode
        self.seq_lens = seq_lens
        self.input_lens = input_lens
        self.block_tables = block_tables
        self.cu_seqlens = cu_seqlens
        self.cu_kv_seqlens = cu_kv_seqlens
        # (k_global_scale, v_global_scale) for NVFP4 KV cache, None otherwise
        self.kv_global_scale = kv_global_scale
        self.use_ragged_prefill = use_ragged_prefill


# ---------------------------------------------------------------------------
# Triton kernels for CUDA-graph prepare
# ---------------------------------------------------------------------------


@triton.jit
def _convert_block_id_to_kv_offset(
    pid,
    block_id_ptr,
    kv_offset_ptr,
    M,
    total_bm,
    BLOCK_SIZE: tl.constexpr,
):
    """block_id[B, M] -> kv_offset[B, 2, M] with K=id*2, V=id*2+1."""
    bm_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    bm_mask = bm_offsets < total_bm
    batch_idx = bm_offsets // M
    col_idx = bm_offsets % M
    block_id = tl.load(block_id_ptr + bm_offsets, mask=bm_mask)
    out_base = batch_idx * 2 * M + col_idx
    tl.store(kv_offset_ptr + out_base, block_id * 2, mask=bm_mask)
    tl.store(kv_offset_ptr + out_base + M, block_id * 2 + 1, mask=bm_mask)


@triton.jit
def _prepare_cg_decode_kernel(
    src_ptr,
    seq_lens_out_ptr,
    block_id_ptr,
    kv_offset_ptr,
    N,
    M,
    total_bm,
    BLOCK_SIZE: tl.constexpr,
):
    """Decode: seq_lens_out = copy(src), block_id -> kv_offset."""
    pid = tl.program_id(0)
    if pid == 0:
        offsets_n = tl.arange(0, BLOCK_SIZE)
        mask_n = offsets_n < N
        vals = tl.load(src_ptr + offsets_n, mask=mask_n)
        tl.store(seq_lens_out_ptr + offsets_n, vals, mask=mask_n)
    _convert_block_id_to_kv_offset(
        pid,
        block_id_ptr,
        kv_offset_ptr,
        M,
        total_bm,
        BLOCK_SIZE,
    )


@triton.jit
def _prepare_cg_spec_decode_kernel(
    prefix_ptr,
    q_len_ptr,
    seq_lens_out_ptr,
    block_id_ptr,
    kv_offset_ptr,
    N,
    M,
    total_bm,
    BLOCK_SIZE: tl.constexpr,
):
    """Spec-decode: seq_lens_out = prefix + q_len[0], block_id -> kv_offset.

    Only q_len[0] is read because speculative decoding requires all requests
    in the batch to share the same speculative query length.
    """
    pid = tl.program_id(0)
    if pid == 0:
        offsets_n = tl.arange(0, BLOCK_SIZE)
        mask_n = offsets_n < N
        q_len = tl.load(q_len_ptr)
        prefix = tl.load(prefix_ptr + offsets_n, mask=mask_n)
        tl.store(seq_lens_out_ptr + offsets_n, prefix + q_len, mask=mask_n)
    _convert_block_id_to_kv_offset(
        pid,
        block_id_ptr,
        kv_offset_ptr,
        M,
        total_bm,
        BLOCK_SIZE,
    )


@triton.jit
def _prepare_cg_prefill_kernel(
    input_lens_ptr,
    prefix_lens_ptr,
    seq_lens_out_ptr,
    cu_kv_seqlens_out_ptr,
    block_id_ptr,
    kv_offset_ptr,
    page_size,
    N,
    M,
    total_bm,
    BLOCK_SIZE: tl.constexpr,
):
    """Prefill: seq_lens_out = input_lens + prefix_lens,
    cu_kv_seqlens_out = [0, cumsum(ceil_div(seq_lens, page_size))],
    block_id -> kv_offset.
    """
    pid = tl.program_id(0)
    if pid == 0:
        offsets_n = tl.arange(0, BLOCK_SIZE)
        mask_n = offsets_n < N
        input_lens = tl.load(input_lens_ptr + offsets_n, mask=mask_n, other=0)
        prefix_lens = tl.load(prefix_lens_ptr + offsets_n, mask=mask_n, other=0)
        seq_lens = input_lens + prefix_lens
        tl.store(seq_lens_out_ptr + offsets_n, seq_lens, mask=mask_n)
        page_per_seq = (seq_lens + page_size - 1) // page_size
        cu_kv = tl.cumsum(page_per_seq, axis=0)
        tl.store(cu_kv_seqlens_out_ptr + offsets_n + 1, cu_kv, mask=mask_n)
        first_mask = offsets_n == 0
        tl.store(
            cu_kv_seqlens_out_ptr + offsets_n,
            tl.zeros([BLOCK_SIZE], dtype=tl.int32),
            mask=first_mask,
        )
    _convert_block_id_to_kv_offset(
        pid,
        block_id_ptr,
        kv_offset_ptr,
        M,
        total_bm,
        BLOCK_SIZE,
    )


# ---------------------------------------------------------------------------
# CUDA-graph launch parameter types
# ---------------------------------------------------------------------------


class _DecodeCGParams(NamedTuple):
    """Pre-computed CUDA graph launch parameters for decode / spec-decode."""

    grid: tuple
    seq_lens: torch.Tensor
    kv_cache_offset: torch.Tensor
    N: int
    M: int
    total_bm: int
    BLOCK_SIZE: int


class _PrefillCGParams(NamedTuple):
    """Pre-computed CUDA graph launch parameters for prefill."""

    grid: tuple
    seq_lens: torch.Tensor
    cu_kv_seqlens: torch.Tensor
    kv_cache_offset: torch.Tensor
    page_size: int
    N: int
    M: int
    total_bm: int
    BLOCK_SIZE: int


def _compute_cg_grid(batch_size: int, num_blocks_per_seq: int):
    """Compute common CUDA graph grid dimensions and block size."""
    N = batch_size
    M = num_blocks_per_seq
    total_bm = N * M
    BS = max(triton.next_power_of_2(N), 1024)
    grid = (triton.cdiv(total_bm, BS),)
    return grid, N, M, total_bm, BS


def _init_decode_cg_params(
    batch_size,
    kv_cache_block_id_device,
    seq_lens,
    kv_cache_offset,
) -> _DecodeCGParams:
    """Pre-compute CUDA graph launch parameters for decode / spec-decode."""
    grid, N, M, total_bm, BS = _compute_cg_grid(
        batch_size,
        kv_cache_block_id_device.shape[1],
    )
    return _DecodeCGParams(
        grid=grid,
        seq_lens=seq_lens,
        kv_cache_offset=kv_cache_offset,
        N=N,
        M=M,
        total_bm=total_bm,
        BLOCK_SIZE=BS,
    )


def _init_prefill_cg_params(
    batch_size,
    kv_cache_block_id_device,
    seq_lens,
    cu_kv_seqlens,
    kv_cache_offset,
    page_size,
) -> _PrefillCGParams:
    """Pre-compute CUDA graph launch parameters for prefill."""
    grid, N, M, total_bm, BS = _compute_cg_grid(
        batch_size,
        kv_cache_block_id_device.shape[1],
    )
    return _PrefillCGParams(
        grid=grid,
        seq_lens=seq_lens,
        cu_kv_seqlens=cu_kv_seqlens,
        kv_cache_offset=kv_cache_offset,
        page_size=page_size,
        N=N,
        M=M,
        total_bm=total_bm,
        BLOCK_SIZE=BS,
    )


# ---------------------------------------------------------------------------
# Op classes (low-level prepare + forward)
# ---------------------------------------------------------------------------


class FlashInferTRTLLMPrefillOp(object):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
    ):
        self.attn_configs = attn_configs
        self.head_dim = attn_configs.size_per_head
        self.head_num = attn_configs.head_num
        self.scaling = self.head_dim**-0.5
        self.local_head_num = attn_configs.head_num
        self.local_head_kv_num = attn_configs.kv_head_num
        self.seq_size_per_block = attn_configs.kernel_tokens_per_block
        self.workspace_buffer = get_trt_workspace_buffer()
        self.ragged_prefill_wrapper = (
            flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace_buffer,
                backend="auto",
            )
        )

    def __del__(self):
        release_trt_workspace_buffer(self.workspace_buffer)

    def support(self, attention_inputs: PyAttentionInputs):
        return (
            is_sm_100()
            and attention_inputs.is_prefill
            and (
                attention_inputs.kv_cache_kernel_block_id_device is not None
                or (
                    self.attn_configs.kv_cache_dtype == KvCacheDataType.NVFP4
                    and _is_zero_prefix_prefill(attention_inputs)
                )
            )
        )

    def prepare(self, attention_inputs: PyAttentionInputs) -> FlashInferTRTLLMParams:
        batch_size = attention_inputs.input_lengths.size(0)
        if attention_inputs.kv_cache_kernel_block_id_device is None:
            cu_seqlens = attention_inputs.cu_seqlens[: batch_size + 1]
            self.ragged_prefill_wrapper.plan(
                cu_seqlens,
                cu_seqlens,
                self.local_head_num,
                self.local_head_kv_num,
                self.head_dim,
                self.head_dim,
                causal=True,
                q_data_type=get_scalar_type(attention_inputs.dtype),
            )
            seq_lens = attention_inputs.input_lengths.to(
                device="cuda", dtype=torch.int32
            )
            return FlashInferTRTLLMParams(
                batch_size=batch_size,
                max_q_len=attention_inputs.input_lengths.max().item(),
                max_kv_len=attention_inputs.input_lengths.max().item(),
                seq_lens=seq_lens,
                input_lens=attention_inputs.input_lengths,
                cu_seqlens=cu_seqlens,
                cu_kv_seqlens=cu_seqlens,
                use_ragged_prefill=True,
            )
        prefix_lengths = torch.zeros_like(
            attention_inputs.input_lengths,
            device="cuda",
            dtype=attention_inputs.input_lengths.dtype,
        )
        input_lengths = torch.zeros_like(
            attention_inputs.input_lengths,
            device="cuda",
            dtype=attention_inputs.input_lengths.dtype,
        )
        prefix_lengths.copy_(attention_inputs.prefix_lengths, non_blocking=True)
        input_lengths.copy_(attention_inputs.input_lengths, non_blocking=True)
        sequence_lengths = input_lengths + prefix_lengths
        page_size = self.seq_size_per_block
        page_per_seq = (sequence_lengths + page_size - 1) // page_size
        cu_kv_seqlens = torch.zeros(
            attention_inputs.input_lengths.shape[0] + 1,
            device="cuda",
            dtype=attention_inputs.input_lengths.dtype,
        )
        cu_kv_seqlens[1:] = torch.cumsum(page_per_seq, dim=0, dtype=torch.int32)
        return FlashInferTRTLLMParams(
            batch_size=attention_inputs.input_lengths.size(0),
            max_q_len=attention_inputs.input_lengths.max().item(),
            max_kv_len=(
                attention_inputs.prefix_lengths + attention_inputs.input_lengths
            )
            .max()
            .item(),
            seq_lens=sequence_lengths,
            input_lens=attention_inputs.input_lengths,
            block_tables=attention_inputs.kv_cache_kernel_block_id_device,
            cu_seqlens=attention_inputs.cu_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
        )

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        fmha_params: FlashInferTRTLLMParams,
    ) -> torch.Tensor:
        if fmha_params.use_ragged_prefill:
            qkv = q.reshape(q.shape[0], -1)
            query, key, value = _split_qkv(
                qkv,
                self.local_head_num,
                self.local_head_kv_num,
                self.head_dim,
            )
            return self.ragged_prefill_wrapper.run(query, key, value)
        dtype = kv_cache.kv_cache_base.dtype
        is_nvfp4 = dtype == torch.uint8
        q_type = q.dtype
        o_type = q_type

        if is_nvfp4:
            # TRTLLM-GEN FP4 kernels require Q=FP8(E4M3) and O=FP8(E4M3)
            q = q.to(torch.float8_e4m3fn)
            o_type = torch.float8_e4m3fn
        else:
            q = q.to(dtype)
        q = q.contiguous().view(-1, self.local_head_num, self.head_dim)

        kv_cache_sf = None
        if is_nvfp4:
            k_global_scale, v_global_scale = fmha_params.kv_global_scale
            bmm1_scale = k_global_scale * self.scaling
            bmm2_scale = v_global_scale
        else:
            bmm1_scale = self.scaling
            bmm2_scale = 1.0

        if kv_cache:
            if is_nvfp4:
                if kv_cache.kv_cache_base.dim() != 5:
                    kv_cache.kv_cache_base = kv_cache.kv_cache_base.view(
                        kv_cache.kv_cache_base.shape[0],
                        2,
                        self.local_head_kv_num,
                        self.seq_size_per_block,
                        self.head_dim // 2,
                    )
                kv_scale_base = kv_cache.kv_scale_base
                if kv_scale_base.dtype == torch.uint8:
                    kv_scale_base = kv_scale_base.view(torch.float8_e4m3fn)
                kv_scales_5d = kv_scale_base.view(
                    kv_cache.kv_scale_base.shape[0],
                    2,
                    self.local_head_kv_num,
                    self.seq_size_per_block,
                    self.head_dim // 16,
                )
                kv_cache_sf = (kv_scales_5d[:, 0], kv_scales_5d[:, 1])
            else:
                kv_cache.kv_cache_base = kv_cache.kv_cache_base.view(
                    kv_cache.kv_cache_base.shape[0],
                    2,
                    self.local_head_kv_num,
                    self.seq_size_per_block,
                    self.head_dim,
                )

        o = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
            query=q,
            kv_cache=kv_cache.kv_cache_base,
            workspace_buffer=self.workspace_buffer,
            block_tables=fmha_params.block_tables,
            seq_lens=fmha_params.seq_lens,
            max_q_len=fmha_params.max_q_len,
            max_kv_len=fmha_params.max_kv_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            batch_size=fmha_params.batch_size,
            cum_seq_lens_q=fmha_params.cu_seqlens,
            cum_seq_lens_kv=fmha_params.cu_kv_seqlens,
            window_left=-1,
            sinks=None,
            out_dtype=o_type,
            kv_cache_sf=kv_cache_sf,
        )

        return o.view(-1, self.local_head_num * self.head_dim).to(q_type)


class FlashInferTRTLLMDecodeOp(object):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
    ):
        self.attn_configs = attn_configs
        self.head_dim = attn_configs.size_per_head
        self.head_num = attn_configs.head_num
        self.scaling = self.head_dim**-0.5
        self.seq_size_per_block = attn_configs.kernel_tokens_per_block
        self.local_head_num = attn_configs.head_num
        self.local_head_kv_num = attn_configs.kv_head_num
        self.workspace_buffer = get_trt_workspace_buffer()

    def __del__(self):
        release_trt_workspace_buffer(self.workspace_buffer)

    def support(self, attention_inputs: PyAttentionInputs):
        if not is_sm_100():
            return False
        # Note: this max q length is used for mtp decode verification.
        decode_kernel_max_q_len = 11
        if (
            attention_inputs.is_prefill
            and attention_inputs.input_lengths[0] < decode_kernel_max_q_len
            and (attention_inputs.input_lengths == attention_inputs.input_lengths[0])
            .all()
            .item()
        ):
            return True
        return not attention_inputs.is_prefill

    def prepare(self, attention_inputs: PyAttentionInputs) -> FlashInferTRTLLMParams:
        if not attention_inputs.is_prefill:
            # need transfer to cuda, cuda graph can capture the add
            sequence_lengths = torch.ones_like(
                attention_inputs.sequence_lengths,
                device="cuda",
                dtype=attention_inputs.sequence_lengths.dtype,
            )
            sequence_lengths.copy_(
                attention_inputs.sequence_lengths, non_blocking=True
            ).add_(1)
            return FlashInferTRTLLMParams(
                batch_size=attention_inputs.sequence_lengths.size(0),
                max_seq_len=attention_inputs.sequence_lengths.max().item() + 1,
                seq_lens=sequence_lengths,
                block_tables=attention_inputs.kv_cache_kernel_block_id_device,
            )
        else:
            q_len = attention_inputs.input_lengths[0].item()
            sequence_lengths = torch.zeros_like(
                attention_inputs.prefix_lengths,
                device="cuda",
                dtype=attention_inputs.prefix_lengths.dtype,
            )
            sequence_lengths.copy_(
                attention_inputs.prefix_lengths, non_blocking=True
            ).add_(q_len)
            return FlashInferTRTLLMParams(
                batch_size=attention_inputs.prefix_lengths.size(0),
                max_seq_len=attention_inputs.prefix_lengths.max().item() + q_len,
                seq_lens=sequence_lengths,
                block_tables=attention_inputs.kv_cache_kernel_block_id_device,
            )

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        fmha_params: FlashInferTRTLLMParams,
    ) -> torch.Tensor:
        dtype = kv_cache.kv_cache_base.dtype
        is_nvfp4 = dtype == torch.uint8
        q_type = q.dtype
        o_type = q_type

        if is_nvfp4:
            # TRTLLM-GEN FP4 kernels require Q=FP8(E4M3) and O=FP8(E4M3)
            q = q.to(torch.float8_e4m3fn)
            o_type = torch.float8_e4m3fn
        else:
            q = q.to(dtype)
        q = q.contiguous().view(-1, self.local_head_num, self.head_dim)

        kv_cache_sf = None
        if is_nvfp4:
            k_global_scale, v_global_scale = fmha_params.kv_global_scale
            bmm1_scale = k_global_scale * self.scaling
            bmm2_scale = v_global_scale
        else:
            bmm1_scale = self.scaling
            bmm2_scale = 1.0

        if kv_cache:
            if is_nvfp4:
                if kv_cache.kv_cache_base.dim() != 5:
                    kv_cache.kv_cache_base = kv_cache.kv_cache_base.view(
                        kv_cache.kv_cache_base.shape[0],
                        2,
                        self.local_head_kv_num,
                        self.seq_size_per_block,
                        self.head_dim // 2,
                    )
                kv_scale_base = kv_cache.kv_scale_base
                if kv_scale_base.dtype == torch.uint8:
                    kv_scale_base = kv_scale_base.view(torch.float8_e4m3fn)
                kv_scales_5d = kv_scale_base.view(
                    kv_cache.kv_scale_base.shape[0],
                    2,
                    self.local_head_kv_num,
                    self.seq_size_per_block,
                    self.head_dim // 16,
                )
                kv_cache_sf = (kv_scales_5d[:, 0], kv_scales_5d[:, 1])
            else:
                kv_cache.kv_cache_base = kv_cache.kv_cache_base.view(
                    kv_cache.kv_cache_base.shape[0],
                    2,
                    self.local_head_kv_num,
                    self.seq_size_per_block,
                    self.head_dim,
                )

        # Call TRT-LLM kernel
        o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=kv_cache.kv_cache_base,
            workspace_buffer=self.workspace_buffer,
            block_tables=fmha_params.block_tables,
            seq_lens=fmha_params.seq_lens,
            max_seq_len=fmha_params.max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=-1,
            sinks=None,
            out_dtype=o_type,
            q_len_per_req=q.shape[0] // fmha_params.seq_lens.shape[0],
            kv_cache_sf=kv_cache_sf,
        )
        return o.view(-1, self.local_head_num * self.head_dim).to(q_type)


# ---------------------------------------------------------------------------
# Impl classes (CUDA-graph-aware wrappers)
# ---------------------------------------------------------------------------


def _split_qkv(
    qkv: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple:
    """Split QKV tensor into query, key, value tensors."""
    qkv = qkv.reshape(qkv.shape[0], -1)
    q, k, v = torch.split(
        qkv,
        [head_dim * num_heads, head_dim * num_kv_heads, head_dim * num_kv_heads],
        dim=-1,
    )
    query = q.reshape(q.shape[0], num_heads, head_dim)
    key = k.reshape(k.shape[0], num_kv_heads, head_dim)
    value = v.reshape(v.shape[0], num_kv_heads, head_dim)
    return query, key, value


class FlashInferTRTLLMPrefillImpl(FMHAImplBase):

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = FlashInferTRTLLMPrefillOp(attn_configs)
        self.attn_inputs = attn_inputs
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

        self.is_nvfp4 = attn_configs.kv_cache_dtype == KvCacheDataType.NVFP4

        if self.is_nvfp4:
            # NVFP4 path: flashinfer rope + NVFP4 quantize + swizzled write
            self.nvfp4_rope_impl = (
                MhaRotaryEmbeddingOp(attn_configs)
                if attn_configs.rope_config.style != RopeStyle.No
                else None
            )
            self.nvfp4_write_op = NVFP4KVCacheWriteOp(
                attn_configs.kv_head_num,
                attn_configs.size_per_head,
                attn_configs.kernel_tokens_per_block,
            )
            # Shared params for rope and write ops
            self.nvfp4_params = rtp_llm_ops.FlashInferMlaAttnParams()
            if self.nvfp4_rope_impl is not None:
                self.nvfp4_rope_impl.set_params(self.nvfp4_params)
            self.nvfp4_write_op.set_params(self.nvfp4_params)
            prefix = (
                attn_inputs.prefix_lengths
                if attn_inputs.prefix_lengths is not None
                else _EMPTY_PREFIX
            )
            self.nvfp4_params.fill_params(
                prefix,
                attn_inputs.sequence_lengths,
                attn_inputs.input_lengths,
                _get_block_table_host(attn_inputs),
                attn_configs.kernel_tokens_per_block,
            )
            self.fmha_params.kv_global_scale = self.nvfp4_write_op.kv_global_scale
            if attn_inputs.kv_cache_kernel_block_id_device is not None:
                # CG kv_cache_offset: needed by Triton kernel but not read by our ops
                from rtp_kernel.fused_rope_kvcache import convert_offset_to_block_array

                kv_cache_offset = convert_offset_to_block_array(
                    attn_inputs.kv_cache_kernel_block_id_device
                )
            else:
                kv_cache_offset = None
        else:
            # Non-NVFP4 path: fused rope + cache write
            self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQOut(attn_configs)
            self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
            kv_cache_offset = self.rope_params.kv_cache_offset

        self._cg = None
        if (
            attn_inputs.kv_cache_kernel_block_id_device is not None
            and kv_cache_offset is not None
        ):
            self._cg = _init_prefill_cg_params(
                self.fmha_params.batch_size,
                attn_inputs.kv_cache_kernel_block_id_device,
                self.fmha_params.seq_lens,
                self.fmha_params.cu_kv_seqlens,
                kv_cache_offset,
                self.fmha_impl.seq_size_per_block,
            )

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        fmha_impl = FlashInferTRTLLMPrefillOp(attn_configs)
        return fmha_impl.support(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int,
    ) -> torch.Tensor:
        if self.need_rope_kv_cache:
            if self.is_nvfp4:
                if self.nvfp4_rope_impl is not None:
                    query, key, value = self.nvfp4_rope_impl.forward(qkv)
                else:
                    query, key, value = _split_qkv(
                        qkv,
                        self.fmha_impl.local_head_num,
                        self.fmha_impl.local_head_kv_num,
                        self.fmha_impl.head_dim,
                    )
                if self.fmha_params.use_ragged_prefill:
                    fmha_input = torch.cat(
                        (
                            query.reshape(query.shape[0], -1),
                            key.reshape(key.shape[0], -1),
                            value.reshape(value.shape[0], -1),
                        ),
                        dim=-1,
                    )
                else:
                    self.nvfp4_write_op.forward(key, value, kv_cache)
                    fmha_input = query
            else:
                fmha_input = self.rope_kvcache_impl.forward(
                    qkv, kv_cache, self.rope_params
                )
        else:
            fmha_input = qkv

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        if self._cg is None:
            return
        if self.is_nvfp4:
            prefix = (
                attn_inputs.prefix_lengths
                if attn_inputs.prefix_lengths is not None
                else _EMPTY_PREFIX
            )
            self.nvfp4_params.fill_params(
                prefix,
                attn_inputs.sequence_lengths,
                attn_inputs.input_lengths,
                _get_block_table_host(attn_inputs),
                self.fmha_impl.seq_size_per_block,
                True,  # forbid_realloc
            )
        p = self._cg
        _prepare_cg_prefill_kernel[p.grid](
            attn_inputs.input_lengths_d,
            attn_inputs.prefix_lengths_d,
            p.seq_lens,
            p.cu_kv_seqlens,
            attn_inputs.kv_cache_kernel_block_id_device,
            p.kv_cache_offset,
            p.page_size,
            p.N,
            p.M,
            p.total_bm,
            BLOCK_SIZE=p.BLOCK_SIZE,
        )


class FlashInferTRTLLMSpecDecodeImpl(FMHAImplBase):

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = FlashInferTRTLLMDecodeOp(attn_configs)
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

        self.is_nvfp4 = attn_configs.kv_cache_dtype == KvCacheDataType.NVFP4

        if self.is_nvfp4:
            self.nvfp4_rope_impl = (
                MhaRotaryEmbeddingOp(attn_configs)
                if attn_configs.rope_config.style != RopeStyle.No
                else None
            )
            self.nvfp4_write_op = NVFP4KVCacheWriteOp(
                attn_configs.kv_head_num,
                attn_configs.size_per_head,
                attn_configs.kernel_tokens_per_block,
            )
            self.nvfp4_params = rtp_llm_ops.FlashInferMlaAttnParams()
            if self.nvfp4_rope_impl is not None:
                self.nvfp4_rope_impl.set_params(self.nvfp4_params)
            self.nvfp4_write_op.set_params(self.nvfp4_params)
            prefix = (
                attn_inputs.prefix_lengths
                if attn_inputs.prefix_lengths is not None
                else _EMPTY_PREFIX
            )
            self.nvfp4_params.fill_params(
                prefix,
                attn_inputs.sequence_lengths,
                attn_inputs.input_lengths,
                attn_inputs.kv_cache_kernel_block_id_host,
                attn_configs.kernel_tokens_per_block,
            )
            self.fmha_params.kv_global_scale = self.nvfp4_write_op.kv_global_scale
            from rtp_kernel.fused_rope_kvcache import convert_offset_to_block_array

            kv_cache_offset = convert_offset_to_block_array(
                attn_inputs.kv_cache_kernel_block_id_device
            )
        else:
            self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQOut(attn_configs)
            self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
            kv_cache_offset = self.rope_params.kv_cache_offset

        self._cg = _init_decode_cg_params(
            self.fmha_params.batch_size,
            attn_inputs.kv_cache_kernel_block_id_device,
            self.fmha_params.seq_lens,
            kv_cache_offset,
        )

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        if attn_configs.use_mla:
            return False
        fmha_impl = FlashInferTRTLLMDecodeOp(attn_configs)
        return fmha_impl.support(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int,
    ) -> torch.Tensor:
        if self.need_rope_kv_cache:
            if self.is_nvfp4:
                if self.nvfp4_rope_impl is not None:
                    query, key, value = self.nvfp4_rope_impl.forward(qkv)
                else:
                    query, key, value = _split_qkv(
                        qkv,
                        self.fmha_impl.local_head_num,
                        self.fmha_impl.local_head_kv_num,
                        self.fmha_impl.head_dim,
                    )
                self.nvfp4_write_op.forward(key, value, kv_cache)
                fmha_input = query
            else:
                fmha_input = self.rope_kvcache_impl.forward(
                    qkv, kv_cache, self.rope_params
                )
        else:
            fmha_input = qkv

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        if self.is_nvfp4:
            prefix = (
                attn_inputs.prefix_lengths
                if attn_inputs.prefix_lengths is not None
                else _EMPTY_PREFIX
            )
            self.nvfp4_params.fill_params(
                prefix,
                attn_inputs.sequence_lengths,
                attn_inputs.input_lengths,
                attn_inputs.kv_cache_kernel_block_id_host,
                self.fmha_impl.seq_size_per_block,
                True,  # forbid_realloc
            )
        p = self._cg
        if not attn_inputs.is_prefill:
            _prepare_cg_decode_kernel[p.grid](
                attn_inputs.sequence_lengths_plus_1_d,
                p.seq_lens,
                attn_inputs.kv_cache_kernel_block_id_device,
                p.kv_cache_offset,
                p.N,
                p.M,
                p.total_bm,
                BLOCK_SIZE=p.BLOCK_SIZE,
            )
        else:
            _prepare_cg_spec_decode_kernel[p.grid](
                attn_inputs.prefix_lengths_d,
                attn_inputs.input_lengths_d,
                p.seq_lens,
                attn_inputs.kv_cache_kernel_block_id_device,
                p.kv_cache_offset,
                p.N,
                p.M,
                p.total_bm,
                BLOCK_SIZE=p.BLOCK_SIZE,
            )


class FlashInferTRTLLMDecodeImpl(FMHAImplBase):

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = FlashInferTRTLLMDecodeOp(attn_configs)
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

        self.is_nvfp4 = attn_configs.kv_cache_dtype == KvCacheDataType.NVFP4

        if self.is_nvfp4:
            self.nvfp4_rope_impl = (
                MhaRotaryEmbeddingOp(attn_configs)
                if attn_configs.rope_config.style != RopeStyle.No
                else None
            )
            self.nvfp4_write_op = NVFP4KVCacheWriteOp(
                attn_configs.kv_head_num,
                attn_configs.size_per_head,
                attn_configs.kernel_tokens_per_block,
            )
            self.nvfp4_params = rtp_llm_ops.FlashInferMlaAttnParams()
            if self.nvfp4_rope_impl is not None:
                self.nvfp4_rope_impl.set_params(self.nvfp4_params)
            self.nvfp4_write_op.set_params(self.nvfp4_params)
            prefix = (
                attn_inputs.prefix_lengths
                if attn_inputs.prefix_lengths is not None
                else _EMPTY_PREFIX
            )
            self.nvfp4_params.fill_params(
                prefix,
                attn_inputs.sequence_lengths,
                attn_inputs.input_lengths,
                attn_inputs.kv_cache_kernel_block_id_host,
                attn_configs.kernel_tokens_per_block,
            )
            self.fmha_params.kv_global_scale = self.nvfp4_write_op.kv_global_scale
            from rtp_kernel.fused_rope_kvcache import convert_offset_to_block_array

            kv_cache_offset = convert_offset_to_block_array(
                attn_inputs.kv_cache_kernel_block_id_device
            )
        else:
            self.rope_kvcache_impl = FusedRopeKVCacheDecodeOp(attn_configs)
            self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
            kv_cache_offset = self.rope_params.kv_cache_offset

        self._cg = _init_decode_cg_params(
            self.fmha_params.batch_size,
            attn_inputs.kv_cache_kernel_block_id_device,
            self.fmha_params.seq_lens,
            kv_cache_offset,
        )

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        if attn_configs.use_mla:
            return False
        fmha_impl = FlashInferTRTLLMDecodeOp(attn_configs)
        return fmha_impl.support(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int,
    ) -> torch.Tensor:
        if self.need_rope_kv_cache:
            if self.is_nvfp4:
                if self.nvfp4_rope_impl is not None:
                    query, key, value = self.nvfp4_rope_impl.forward(qkv)
                else:
                    query, key, value = _split_qkv(
                        qkv,
                        self.fmha_impl.local_head_num,
                        self.fmha_impl.local_head_kv_num,
                        self.fmha_impl.head_dim,
                    )
                self.nvfp4_write_op.forward(key, value, kv_cache)
                fmha_input = query
            else:
                fmha_input = self.rope_kvcache_impl.forward(
                    qkv, kv_cache, self.rope_params
                )
        else:
            fmha_input = qkv

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        if self.is_nvfp4:
            prefix = (
                attn_inputs.prefix_lengths
                if attn_inputs.prefix_lengths is not None
                else _EMPTY_PREFIX
            )
            self.nvfp4_params.fill_params(
                prefix,
                attn_inputs.sequence_lengths,
                attn_inputs.input_lengths,
                attn_inputs.kv_cache_kernel_block_id_host,
                self.fmha_impl.seq_size_per_block,
                True,  # forbid_realloc
            )
        p = self._cg
        _prepare_cg_decode_kernel[p.grid](
            attn_inputs.sequence_lengths_plus_1_d,
            p.seq_lens,
            attn_inputs.kv_cache_kernel_block_id_device,
            p.kv_cache_offset,
            p.N,
            p.M,
            p.total_bm,
            BLOCK_SIZE=p.BLOCK_SIZE,
        )
