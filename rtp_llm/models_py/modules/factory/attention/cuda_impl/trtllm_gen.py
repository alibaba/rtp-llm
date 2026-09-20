import os
import threading
import weakref
from typing import NamedTuple, Optional

import flashinfer
import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.triton_kernels.common.fused_fp8_qkv_cache import (
    fused_fp8_qkv_cache,
    quantize_fp8_query,
)
from rtp_llm.models_py.utils.arch import is_blackwell, is_sm10x, is_sm12x
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
    FusedRopeKVCachePrefillOpQOut,
    LayerKVCache,
    PyAttentionInputs,
)

# Constants
DEFAULT_TRT_WORKSPACE_SIZE_MB = (
    512  # Memory workspace size in MB, todo(Yingyi): read from config
)

# Global workspace buffer pool
_g_trt_workspace_pool: list[torch.Tensor] = []
_g_trt_pool_lock = threading.Lock()
_g_trt_graph_workspaces = weakref.WeakValueDictionary()
_g_trt_graph_workspace_lock = threading.Lock()


def use_native_fp8_attention(attn_configs: AttentionConfigs) -> bool:
    """Opt in to the Qwen3.5 SM100 FP8-Q/FP8-KV path.

    KV storage uses RTP's existing static unit-scale contract. The kernel page
    size and speculative/CUDA-graph metadata remain independent of this gate.
    """
    return (
        os.getenv("RTP_QWEN35_NATIVE_FP8_ATTN", "0") == "1"
        and attn_configs.dtype == torch.bfloat16
        and attn_configs.kv_cache_dtype == KvCacheDataType.FP8
        and attn_configs.need_rope_kv_cache
        and (
            attn_configs.rope_config.style == RopeStyle.Mrope
            or (
                attn_configs.rope_config.style == RopeStyle.Base
                and attn_configs.rope_config.index_factor == 1
            )
        )
        and attn_configs.size_per_head == 256
        and attn_configs.kernel_tokens_per_block == 64
        and is_sm10x()
    )


def _fused_fp8_prefill_query(impl, qkv: torch.Tensor, kv_cache: LayerKVCache):
    config = impl.attn_configs
    if config.rope_config.style == RopeStyle.Base and (
        impl.rope_params.position_ids is None
        or impl.rope_params.position_ids.numel() == 0
    ):
        raise ValueError("Native FP8 Base RoPE positions were not prepared")
    # QKV-output mode only rotates; do not ask this RTP-Kernel mode to write
    # cache. Fuse all three conversions with the explicit paged-cache store.
    rotated = impl.rope_kvcache_impl.forward(qkv, None, impl.rope_params)
    paged_cache = common.reshape_paged_kv_cache(
        kv_cache.kv_cache_base,
        config.kv_head_num,
        config.kernel_tokens_per_block,
        config.size_per_head,
    )
    return fused_fp8_qkv_cache(
        rotated,
        paged_cache,
        impl.attn_inputs.kv_cache_kernel_block_id_device,
        impl.attn_inputs.cu_seqlens_device,
        impl.attn_inputs.prefix_lengths_device,
        num_q_heads=config.head_num,
        num_kv_heads=config.kv_head_num,
        head_dim=config.size_per_head,
        page_size=config.kernel_tokens_per_block,
        max_query_len=impl.rope_params.max_seq_len,
    )


def _init_native_base_positions(impl):
    """Materialize implicit Base positions once per metadata preparation."""
    impl._base_position_max_q = 0
    positions = impl.rope_params.position_ids
    if (
        impl.native_fp8
        and impl.attn_configs.rope_config.style == RopeStyle.Base
        and (positions is None or positions.numel() == 0)
    ):
        impl._base_position_max_q = max(1, impl.rope_params.max_seq_len)
        capacity = impl._cg.N * impl._base_position_max_q
        impl.rope_params.position_ids = torch.empty(
            capacity,
            device=impl.attn_inputs.cu_seqlens_device.device,
            dtype=torch.int32,
        )
        impl._cg = impl._cg._replace(
            grid=(max(impl._cg.grid[0], triton.cdiv(capacity, impl._cg.BLOCK_SIZE)),)
        )
        # Also covers the eager datatype warmup before graph capture. Replay
        # updates this same buffer from GPU prefix/cu metadata in-place.
        impl.prepare_cuda_graph(impl.attn_inputs)


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


class _TRTGraphWorkspaceLease:
    def __init__(self, table: torch.Tensor):
        # Keep the owner allocation alive: its address must not be recycled
        # while a captured graph can still access this workspace.
        self.storage = table.untyped_storage()
        self.buffer = get_trt_workspace_buffer(device=str(table.device))

    def __del__(self):
        buffer = getattr(self, "buffer", None)
        if buffer is not None:
            release_trt_workspace_buffer(buffer)


def _native_base_graph_workspace(attn_configs, attn_inputs):
    if not (
        use_native_fp8_attention(attn_configs)
        and attn_configs.rope_config.style == RopeStyle.Base
        and attn_inputs.is_cuda_graph
    ):
        return None
    table = attn_inputs.kv_cache_kernel_block_id_device
    if table is None or table.numel() == 0:
        return None
    # CudaGraphRunner allocates one block-table storage per runner/cache tag;
    # every capture bucket only slices it. These graphs already share mutable
    # inputs and are ordered by the runner's forward event. Other runners and
    # eager execution must retain independent scratch, even on the same GPU.
    key = (table.device, table.untyped_storage().data_ptr())
    with _g_trt_graph_workspace_lock:
        lease = _g_trt_graph_workspaces.get(key)
        if lease is None:
            lease = _TRTGraphWorkspaceLease(table)
            _g_trt_graph_workspaces[key] = lease
        return lease


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
def _prepare_base_positions(
    pid,
    prefix_ptr,
    cu_seqlens_ptr,
    positions_ptr,
    N,
    MAX_QUERY_LEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    flat = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    batch = flat // MAX_QUERY_LEN
    local = flat % MAX_QUERY_LEN
    in_bounds = batch < N
    start = tl.load(cu_seqlens_ptr + batch, mask=in_bounds, other=0)
    end = tl.load(cu_seqlens_ptr + batch + 1, mask=in_bounds, other=0)
    prefix = tl.load(prefix_ptr + batch, mask=in_bounds, other=0)
    tl.store(
        positions_ptr + start + local,
        prefix + local,
        mask=in_bounds & (local < end - start),
    )
    # Packed valid rows and trailing graph padding occupy disjoint ranges.
    total = tl.load(cu_seqlens_ptr + N)
    tl.store(positions_ptr + flat, 0, mask=in_bounds & (flat >= total))


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
    base_positions_ptr=None,
    cu_seqlens_ptr=None,
    MAX_BASE_QUERY_LEN: tl.constexpr = 0,
):
    """Spec-decode: seq_lens_out = prefix + q_len, block_id -> kv_offset.

    Valid speculative requests in the batch share the same query length, but
    CUDA-graph padding rows are zeroed independently in `q_len_ptr`. Read
    per-row q_len so padding rows can stay fully masked out instead of
    inheriting the first request's speculative length.
    """
    pid = tl.program_id(0)
    if pid == 0:
        offsets_n = tl.arange(0, BLOCK_SIZE)
        mask_n = offsets_n < N
        q_len = tl.load(q_len_ptr + offsets_n, mask=mask_n, other=0)
        prefix = tl.load(prefix_ptr + offsets_n, mask=mask_n, other=0)
        tl.store(seq_lens_out_ptr + offsets_n, prefix + q_len, mask=mask_n)
    _convert_block_id_to_kv_offset(
        pid,
        block_id_ptr,
        kv_offset_ptr,
        M,
        total_bm,
        BLOCK_SIZE,
    )
    if MAX_BASE_QUERY_LEN > 0:
        _prepare_base_positions(
            pid,
            prefix_ptr,
            cu_seqlens_ptr,
            base_positions_ptr,
            N,
            MAX_BASE_QUERY_LEN,
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
    N,
    M,
    total_bm,
    BLOCK_SIZE: tl.constexpr,
    base_positions_ptr=None,
    cu_seqlens_ptr=None,
    MAX_BASE_QUERY_LEN: tl.constexpr = 0,
):
    """Prefill: seq_lens_out = input_lens + prefix_lens,
    cu_kv_seqlens_out = [0, cumsum(seq_lens)],
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
        cu_kv = tl.cumsum(seq_lens, axis=0)
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
    if MAX_BASE_QUERY_LEN > 0:
        _prepare_base_positions(
            pid,
            prefix_lens_ptr,
            cu_seqlens_ptr,
            base_positions_ptr,
            N,
            MAX_BASE_QUERY_LEN,
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
        workspace_lease: Optional[_TRTGraphWorkspaceLease] = None,
    ):
        self.attn_configs = attn_configs
        self.native_fp8 = use_native_fp8_attention(attn_configs)
        self.head_dim = attn_configs.size_per_head
        self.head_num = attn_configs.head_num
        self.scaling = self.head_dim**-0.5
        self.local_head_num = attn_configs.head_num
        self.local_head_kv_num = attn_configs.kv_head_num
        self.seq_size_per_block = attn_configs.kernel_tokens_per_block
        self._workspace_lease = workspace_lease
        self.workspace_buffer = (
            workspace_lease.buffer
            if workspace_lease is not None
            else get_trt_workspace_buffer()
        )

    def __del__(self):
        if self._workspace_lease is None:
            release_trt_workspace_buffer(self.workspace_buffer)

    def support(self, attention_inputs: PyAttentionInputs):
        # TllmGenFmhaRunner cubin covers sm_90a / sm_100a only; sm_120a
        # (Blackwell consumer, e.g. RTX 5000 Pro) has no binding and the
        # runner throws "Unsupported architecture" (fmhaRunner.cuh:37) on
        # forward. Fall through so dispatch picks the paged
        # PyFlashinferPagedPrefillImpl instead.
        if is_sm12x():
            return False
        return (
            is_blackwell()
            and attention_inputs.is_prefill
            and attention_inputs.kv_cache_kernel_block_id_device is not None
        )

    def prepare(self, attention_inputs: PyAttentionInputs) -> FlashInferTRTLLMParams:
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
        cu_kv_seqlens = torch.zeros(
            attention_inputs.input_lengths.shape[0] + 1,
            device="cuda",
            dtype=attention_inputs.input_lengths.dtype,
        )
        cu_kv_seqlens[1:] = torch.cumsum(sequence_lengths, dim=0, dtype=torch.int32)
        return FlashInferTRTLLMParams(
            batch_size=attention_inputs.input_lengths.size(0),
            max_q_len=attention_inputs.input_lengths.max().item(),
            max_kv_len=sequence_lengths.max().item(),
            seq_lens=sequence_lengths,
            input_lens=attention_inputs.input_lengths,
            block_tables=attention_inputs.kv_cache_kernel_block_id_device,
            cu_seqlens=attention_inputs.cu_seqlens_device,
            cu_kv_seqlens=cu_kv_seqlens,
        )

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        fmha_params: FlashInferTRTLLMParams,
        *,
        q_scale: float = 1.0,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> torch.Tensor:
        o_type = self.attn_configs.dtype if self.native_fp8 else q.dtype
        bmm1_scale = q_scale * k_scale * self.scaling
        bmm2_scale = v_scale
        if kv_cache:
            kv_cache.kv_cache_base = kv_cache.kv_cache_base.view(
                kv_cache.kv_cache_base.shape[0],
                2,
                self.local_head_kv_num,
                self.seq_size_per_block,
                self.head_dim,
            )

        if self.native_fp8 and q.dtype != torch.float8_e4m3fn:
            q = quantize_fp8_query(
                q,
                scale=q_scale,
                kv_cache=kv_cache.kv_cache_base,
                block_table=fmha_params.block_tables,
                seq_lens=fmha_params.seq_lens,
            )

        q = q.contiguous().view(-1, self.local_head_num, self.head_dim)

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
            # TODO: add attention_sink operation or nvfp4 scale factor if needed
            sinks=None,
            out_dtype=o_type,  # model_runner.dtype
        )

        return o.view(-1, self.local_head_num * self.head_dim).to(o_type)


class FlashInferTRTLLMDecodeOp(object):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        workspace_lease: Optional[_TRTGraphWorkspaceLease] = None,
    ):
        self.attn_configs = attn_configs
        self.native_fp8 = use_native_fp8_attention(attn_configs)
        self.head_dim = attn_configs.size_per_head
        self.head_num = attn_configs.head_num
        self.scaling = self.head_dim**-0.5
        self.seq_size_per_block = attn_configs.kernel_tokens_per_block
        self.local_head_num = attn_configs.head_num
        self.local_head_kv_num = attn_configs.kv_head_num
        self._workspace_lease = workspace_lease
        self.workspace_buffer = (
            workspace_lease.buffer
            if workspace_lease is not None
            else get_trt_workspace_buffer()
        )

    def __del__(self):
        if self._workspace_lease is None:
            release_trt_workspace_buffer(self.workspace_buffer)

    def support(self, attention_inputs: PyAttentionInputs):
        if not is_blackwell():
            return False
        # TllmGenFmhaRunner cubin covers sm_90a / sm_100a only; sm_120a
        # (Blackwell consumer, e.g. RTX 5000 Pro) has no binding and the
        # runner throws "Unsupported architecture" (fmhaRunner.cuh:37) on
        # the first decode forward. Fall through so dispatch picks the
        # ragged PyFlashinferPaged path instead.
        if is_sm12x():
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
        *,
        q_scale: float = 1.0,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> torch.Tensor:
        o_type = self.attn_configs.dtype if self.native_fp8 else q.dtype
        bmm1_scale = q_scale * k_scale * self.scaling
        bmm2_scale = v_scale
        # sink: additional value per head in the denominator of the softmax.
        if kv_cache:
            kv_cache.kv_cache_base = kv_cache.kv_cache_base.view(
                kv_cache.kv_cache_base.shape[0],
                2,
                self.local_head_kv_num,
                self.seq_size_per_block,
                self.head_dim,
            )

        if self.native_fp8 and q.dtype != torch.float8_e4m3fn:
            q = quantize_fp8_query(
                q,
                scale=q_scale,
                kv_cache=kv_cache.kv_cache_base,
                block_table=fmha_params.block_tables,
                seq_lens=fmha_params.seq_lens,
            )

        q = q.contiguous().view(-1, self.local_head_num, self.head_dim)

        # Call TRT-LLM kernel
        # raw_out: like q, [bs, acc_q_len, num_q_heads, head_dim] but with output dtype
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
            # TODO: add attention_sink operation or nvfp4 scale factor if needed
            sinks=None,
            out_dtype=o_type,  # model_runner.dtype
            q_len_per_req=q.shape[0] // fmha_params.seq_lens.shape[0],
        )
        return o.view(-1, self.local_head_num * self.head_dim).to(o_type)


# ---------------------------------------------------------------------------
# Impl classes (CUDA-graph-aware wrappers)
# ---------------------------------------------------------------------------


class FlashInferTRTLLMPrefillImpl(FMHAImplBase):

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = FlashInferTRTLLMPrefillOp(
            attn_configs, _native_base_graph_workspace(attn_configs, attn_inputs)
        )
        self.native_fp8 = self.fmha_impl.native_fp8
        self.attn_configs = attn_configs
        rope_op = (
            FusedRopeKVCachePrefillOpQKVOut
            if self.native_fp8
            else FusedRopeKVCachePrefillOpQOut
        )
        self.rope_kvcache_impl = rope_op(attn_configs)
        self.attn_inputs = attn_inputs
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

        self._cg = _init_prefill_cg_params(
            self.fmha_params.batch_size,
            attn_inputs.kv_cache_kernel_block_id_device,
            self.fmha_params.seq_lens,
            self.fmha_params.cu_kv_seqlens,
            self.rope_params.kv_cache_offset,
        )
        _init_native_base_positions(self)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        if common.requires_native_bf16_fp8_prefill(attn_configs) and not (
            use_native_fp8_attention(attn_configs)
            and attn_inputs.context_parallel_info is None
        ):
            return False
        fmha_impl = FlashInferTRTLLMPrefillOp(attn_configs)
        return fmha_impl.support(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int,
    ) -> torch.Tensor:
        if self.native_fp8:
            assert kv_cache is not None
            fmha_input = _fused_fp8_prefill_query(self, qkv, kv_cache)
        elif self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        p = self._cg
        _prepare_cg_prefill_kernel[p.grid](
            attn_inputs.input_lengths_device,
            attn_inputs.prefix_lengths_device,
            p.seq_lens,
            p.cu_kv_seqlens,
            attn_inputs.kv_cache_kernel_block_id_device,
            p.kv_cache_offset,
            p.N,
            p.M,
            p.total_bm,
            BLOCK_SIZE=p.BLOCK_SIZE,
            base_positions_ptr=self.rope_params.position_ids,
            cu_seqlens_ptr=attn_inputs.cu_seqlens_device,
            MAX_BASE_QUERY_LEN=self._base_position_max_q,
        )


class FlashInferTRTLLMSpecDecodeImpl(FMHAImplBase):

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = FlashInferTRTLLMDecodeOp(
            attn_configs, _native_base_graph_workspace(attn_configs, attn_inputs)
        )
        self.native_fp8 = self.fmha_impl.native_fp8
        rope_op = (
            FusedRopeKVCachePrefillOpQKVOut
            if self.native_fp8
            else FusedRopeKVCachePrefillOpQOut
        )
        self.rope_kvcache_impl = rope_op(attn_configs)
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

        self._cg = _init_decode_cg_params(
            self.fmha_params.batch_size,
            attn_inputs.kv_cache_kernel_block_id_device,
            self.fmha_params.seq_lens,
            self.rope_params.kv_cache_offset,
        )
        _init_native_base_positions(self)

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
        if self.native_fp8:
            assert kv_cache is not None
            fmha_input = _fused_fp8_prefill_query(self, qkv, kv_cache)
        elif self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        p = self._cg
        if not attn_inputs.is_prefill:
            _prepare_cg_decode_kernel[p.grid](
                attn_inputs.sequence_lengths_plus_1_device,
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
                attn_inputs.prefix_lengths_device,
                attn_inputs.input_lengths_device,
                p.seq_lens,
                attn_inputs.kv_cache_kernel_block_id_device,
                p.kv_cache_offset,
                p.N,
                p.M,
                p.total_bm,
                BLOCK_SIZE=p.BLOCK_SIZE,
                base_positions_ptr=self.rope_params.position_ids,
                cu_seqlens_ptr=attn_inputs.cu_seqlens_device,
                MAX_BASE_QUERY_LEN=self._base_position_max_q,
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
        self.rope_kvcache_impl = FusedRopeKVCacheDecodeOp(attn_configs)
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

        self._cg = _init_decode_cg_params(
            self.fmha_params.batch_size,
            attn_inputs.kv_cache_kernel_block_id_device,
            self.fmha_params.seq_lens,
            self.rope_params.kv_cache_offset,
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
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        p = self._cg
        _prepare_cg_decode_kernel[p.grid](
            attn_inputs.sequence_lengths_plus_1_device,
            p.seq_lens,
            attn_inputs.kv_cache_kernel_block_id_device,
            p.kv_cache_offset,
            p.N,
            p.M,
            p.total_bm,
            BLOCK_SIZE=p.BLOCK_SIZE,
        )
