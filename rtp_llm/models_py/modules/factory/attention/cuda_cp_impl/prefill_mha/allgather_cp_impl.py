import logging
import os
from typing import Optional

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    ParamsBase,
    PyAttentionInputs,
    fill_mla_params,
)

logger = logging.getLogger(__name__)

_cp_trt_workspace_buffer: Optional[torch.Tensor] = None


def _page_counts(lengths: torch.Tensor, page_size: int) -> torch.Tensor:
    """Pages needed per request, ``ceil(length / page_size)``."""
    return torch.div(lengths + page_size - 1, page_size, rounding_mode="floor")


def _contiguous_logical_pages(
    page_counts: torch.Tensor, max_pages: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Pack each request's logical page ids consecutively, zero-padding short rows.

    Row ``i`` is ``[start_i, start_i + 1, ...]`` where ``start_i`` is the running
    sum of the preceding page counts.
    """
    starts = torch.zeros_like(page_counts)
    torch.cumsum(page_counts[:-1], dim=0, out=starts[1:])
    ramp = torch.arange(max_pages, dtype=page_counts.dtype, device=page_counts.device)
    table = torch.where(ramp < page_counts.unsqueeze(1), starts.unsqueeze(1) + ramp, 0)
    table = table.to(torch.int32)
    return table if device is None else table.to(device)


def _build_cp_sharded_params_block_table(
    prefix_lengths: torch.Tensor,
    input_lengths: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Build logical page ids used only by ``fill_mla_params`` metadata.

    ``fill_mla_params`` indexes one entry per logical page, so handing it a
    CP-sharded physical table — which holds only this rank's page-RR shard — reads
    out of bounds on long cache hits. These ids never reach physical cache I/O:
    writes go through ``cp_kv_slot_mapping`` and prefix reads through the
    all-gathered pool, both of which keep using the real rank-local table.
    """
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")

    total_lengths = prefix_lengths.to(
        device="cpu", dtype=torch.int64
    ) + input_lengths.to(device="cpu", dtype=torch.int64)
    if not total_lengths.numel():
        return torch.zeros((0, 0), dtype=torch.int32)
    page_counts = _page_counts(total_lengths, page_size)
    return _contiguous_logical_pages(page_counts, int(page_counts.max()))


@triton.jit
def _contiguous_prefix_page_table_kernel(
    prefix_lengths_ptr,
    table_ptr,
    table_stride,
    BATCH: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    own_pages = (tl.load(prefix_lengths_ptr + req) + PAGE_SIZE - 1) // PAGE_SIZE

    # Exclusive prefix sum of page counts. The batch is small enough that an
    # O(BATCH) scan per program beats launching a separate cumsum kernel.
    page_offset = 0
    for other in tl.range(0, BATCH):
        other_pages = (tl.load(prefix_lengths_ptr + other) + PAGE_SIZE - 1) // PAGE_SIZE
        page_offset += tl.where(other < req, other_pages, 0)

    for start in tl.range(0, MAX_PAGES, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        tl.store(
            table_ptr + req * table_stride + offs,
            tl.where(offs < own_pages, page_offset + offs, 0),
            mask=offs < MAX_PAGES,
        )


_logical_page_ramp_buffer: Optional[torch.Tensor] = None


def _logical_page_ramp(size: int, device: torch.device) -> torch.Tensor:
    """Process-lifetime ``[0, 1, 2, ...]`` buffer, grown on demand."""
    global _logical_page_ramp_buffer
    # An indexless "cuda" compares unequal to the "cuda:0" a tensor reports, which
    # would reallocate on every call and defeat the caching.
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    ramp = _logical_page_ramp_buffer
    if ramp is None or ramp.numel() < size or ramp.device != device:
        ramp = torch.arange(max(size, 8192), dtype=torch.int32, device=device)
        _logical_page_ramp_buffer = ramp
    return ramp


def _build_contiguous_prefix_page_table(
    prefix_lengths: torch.Tensor,
    page_size: int,
    max_pages: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Page table for a prefix pool whose pages are packed per request in logical
    order — the layout ``gather_cp_sharded_prefix_pool`` produces. The rank-local
    physical table cannot address it: under page-RR sharding that table only holds
    this rank's 1/cp_size shard, so its entry ``p`` is not logical page ``p``.

    ``prefix_lengths`` may live on either side.
    """
    device = device or (
        prefix_lengths.device if prefix_lengths.is_cuda else torch.device("cuda")
    )
    lengths_host = prefix_lengths.to(device="cpu", dtype=torch.int32)
    batch_size = int(lengths_host.numel())
    if not batch_size or not max_pages:
        return torch.empty((batch_size, max_pages), dtype=torch.int32, device=device)

    page_counts = _page_counts(lengths_host, page_size)
    if bool((page_counts == max_pages).all()):
        # A uniform batch needs no zero padding, so the table is exactly the ramp.
        # Sharing the persistent buffer is safe because the table is read-only.
        total = batch_size * max_pages
        return _logical_page_ramp(total, device)[:total].view(batch_size, max_pages)

    table = torch.empty((batch_size, max_pages), dtype=torch.int32, device=device)
    _contiguous_prefix_page_table_kernel[(batch_size,)](
        prefix_lengths if prefix_lengths.is_cuda else lengths_host.to(device),
        table,
        table.stride(0),
        BATCH=batch_size,
        PAGE_SIZE=page_size,
        MAX_PAGES=max_pages,
        BLOCK=min(triton.next_power_of_2(max_pages), 1024),
    )
    return table


def get_cp_trt_workspace_buffer() -> torch.Tensor:
    global _cp_trt_workspace_buffer
    if _cp_trt_workspace_buffer is None:
        _cp_trt_workspace_buffer = get_trt_workspace_buffer()
    return _cp_trt_workspace_buffer


from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from flashinfer.cascade import merge_state
from flashinfer.page import append_paged_kv_cache
from flashinfer.prefill import trtllm_batch_context_with_kv_cache

# FlashAttention-4 paged attention (Blackwell). Unlike trtllm_batch_context it
# supports an fp8 paged KV cache.
try:
    from flash_attn.cute import flash_attn_varlen_func as _fa4_varlen_func

    _HAS_FA4 = True
except Exception:  # pragma: no cover - FA4 only shipped on cuda13 x86 (Blackwell)
    _fa4_varlen_func = None
    _HAS_FA4 = False


def _env_enabled(name: str, *, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _use_fa4_cp_paged(*, has_prefix: bool = False, fp8_kv_cache: bool = False) -> bool:
    """Select FA4 for the CP prefill paths, including cache hits when opted in.

    FA4 runs uniform FP8 against an FP8 cache (q quantized to match), while
    FlashInfer keeps q in BF16 and dequantizes only the cached prefix. The two are
    therefore not numerically interchangeable per request, so a process serving
    both cold and cache-hit traffic must put both on one backend:
    ``RTP_LLM_CP_PREFILL_FA4_PREFIX`` moves it to FA4, and without it an FP8
    cache-serving process stays entirely on FlashInfer.
    """
    if not _HAS_FA4 or not _env_enabled("RTP_LLM_CP_PREFILL_FA4", default=True):
        return False
    if _env_enabled("RTP_LLM_CP_PREFILL_FA4_PREFIX", default=False):
        return True
    cache_serving_process = _env_enabled("REUSE_CACHE", default=False)
    return not has_prefix and not (cache_serving_process and fp8_kv_cache)


def _match_q_to_kv(q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    """FA4 requires q/k/v to share one dtype, and the KV cache is never re-cast, so
    the bf16 q activation follows the cache: an fp8 cache yields uniform-fp8
    attention (scale=1.0, no descale), a bf16 cache stays bf16.
    """
    return q if q.dtype == kv.dtype else q.to(kv.dtype)


from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
    cast_kv_for_cache_append,
    fill_fp8_kv_cache_scale,
    gather_cp_sharded_prefix_pool,
    plan_prefix_paged_attention,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    get_py_flashinfer_workspace_buffer,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.trtllm_gen import (
    get_trt_workspace_buffer,
)


@triton.jit
def _fused_restore_packed_kv_kernel(
    packed_ptr,
    unpad_ptr,
    k_ptr,
    v_ptr,
    TOTAL: tl.constexpr,
    NK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # int64 throughout: TOTAL is token_count * 2 * num_kv_heads * head_dim, which
    # for a 1M-token context is already half of int32's range, and a wrapped
    # negative offset would still pass ``offs < TOTAL`` and scribble out of bounds.
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < TOTAL
    per_token = 2 * NK
    token = offs // per_token
    field = offs - token * per_token
    src_token = tl.load(unpad_ptr + token, mask=mask, other=0).to(tl.int64)
    vals = tl.load(packed_ptr + src_token * per_token + field, mask=mask, other=0.0)

    is_k = field < NK
    tl.store(k_ptr + token * NK + field, vals, mask=mask & is_k)
    tl.store(v_ptr + token * NK + (field - NK), vals, mask=mask & ~is_k)


def _fused_restore_packed_kv(
    packed_kv: torch.Tensor,
    unpad_indices: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_count = int(unpad_indices.numel())
    restore_k = torch.empty(
        token_count,
        num_kv_heads,
        head_dim,
        device=packed_kv.device,
        dtype=packed_kv.dtype,
    )
    restore_v = torch.empty_like(restore_k)
    if token_count == 0:
        return restore_k, restore_v

    nk = num_kv_heads * head_dim
    total = token_count * 2 * nk
    _fused_restore_packed_kv_kernel[(triton.cdiv(total, 256),)](
        packed_kv,
        unpad_indices,
        restore_k,
        restore_v,
        TOTAL=total,
        NK=nk,
        BLOCK=256,
    )
    return restore_k, restore_v


def _generate_q_indices_device(
    cp_chunk_lengths, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized ``generate_q_indices`` returning int64 CUDA tensors.

    Same first-half/second-half zigzag split as ``cp_utils.generate_q_indices``,
    but each per-chunk range is built with ``torch.arange`` on ``device`` and
    concatenated. This avoids the O(seq) Python ``list.extend(range(...))`` build
    plus ``torch.tensor(list)`` H2D, which stalls the GPU for milliseconds at the
    start of long-context prefill (65536-token part1 ranges).
    """
    parts0, parts1 = [], []
    offset = 0
    for chunk_len in cp_chunk_lengths:
        chunk_len = int(chunk_len)
        half0 = (chunk_len + 1) // 2
        parts0.append(
            torch.arange(offset, offset + half0, device=device, dtype=torch.int64)
        )
        parts1.append(
            torch.arange(
                offset + half0, offset + chunk_len, device=device, dtype=torch.int64
            )
        )
        offset += chunk_len
    empty = torch.empty(0, device=device, dtype=torch.int64)
    return (
        torch.cat(parts0) if parts0 else empty,
        torch.cat(parts1) if parts1 else empty,
    )


def _generate_full_causal_kv_indices_device(
    cp_chunk_lengths, cp_rank: int, cp_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized ``generate_full_causal_kv_indices`` returning int64 CUDA tensors.

    Matches ``cp_utils.generate_full_causal_kv_indices`` exactly (full causal KV
    range per Q-half under zigzag load balancing) but builds the contiguous
    ranges with ``torch.arange`` on ``device`` instead of Python lists.
    """
    parts0, parts1 = [], []
    seq_offset = 0
    for chunk_len in cp_chunk_lengths:
        chunk_len = int(chunk_len)
        assert chunk_len % 2 == 0
        h = chunk_len // 2
        end_part0 = h * (cp_rank + 1)
        if end_part0 > 0:
            parts0.append(
                torch.arange(
                    seq_offset, seq_offset + end_part0, device=device, dtype=torch.int64
                )
            )
        end_part1 = h * (2 * cp_size - cp_rank)
        if end_part1 > 0:
            parts1.append(
                torch.arange(
                    seq_offset, seq_offset + end_part1, device=device, dtype=torch.int64
                )
            )
        seq_offset += chunk_len * cp_size
    empty = torch.empty(0, device=device, dtype=torch.int64)
    return (
        torch.cat(parts0) if parts0 else empty,
        torch.cat(parts1) if parts1 else empty,
    )


class PCPAllGatherAttnOp:
    # One line per process, not per layer per forward: the backend choice is fixed
    # by config, and a silent fallback to FlashInfer is otherwise invisible.
    _logged_fa4_fused = False

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
        backend: str = "auto",  # "auto", "fa2", or "fa3"
        causal: bool = True,
        kv_layout: str = "NHD",  # "NHD" or "HND"
    ):
        """
        Args:
            config: Model configuration
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (for GQA/MQA)
            head_dim: Dimension of each head
            backend: FlashInfer backend ("auto", "fa2", or "fa3")
            causal: Whether to use causal masking
            kv_layout: KV cache layout ("NHD" or "HND")
        """
        super().__init__()
        self.attn_inputs = attn_inputs
        self.attn_configs = attn_configs
        self.num_qo_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.backend = backend
        self.kv_layout = kv_layout

        assert causal == True
        self.device = torch.cuda.current_device()
        self.workspace_buffer = get_py_flashinfer_workspace_buffer()

        self.cp_info = attn_inputs.context_parallel_info

        self.prefill_cp_rank = parallelism_config.tp_rank
        self.prefill_cp_size = parallelism_config.tp_size

        # CP page-RR KV sharding geometry (mirrors MSAAttention / DeviceData::props):
        #   sharded = prefill_cp enabled AND kv_cache_sharded AND raw tp_size>1.
        # When sharded, each rank's local pool / block table only holds the
        # 1/cp_size physical blocks it owns, so writing the full all-gathered
        # sequence with FlashInfer append_paged_kv_cache (which has no -1 skip and
        # reads the full block table) would index past the sharded block table.
        # In that case we route the local-pool write through the sharding-aware
        # C++ writer (mha_kv_write_cache) + cp_kv_slot_mapping (non-owned -> -1).
        cp_cfg = parallelism_config.prefill_cp_config
        self._kv_sharded = bool(
            getattr(cp_cfg, "kv_cache_sharded", False) and self.prefill_cp_size > 1
        )
        self._cp_size = self.prefill_cp_size if self._kv_sharded else 1
        self._cp_rank = self.prefill_cp_rank if self._kv_sharded else 0
        self._fp8_kv_cache = attn_configs.kv_cache_dtype == KvCacheDataType.FP8

        self.seq_size_per_block = (
            attn_configs.kernel_tokens_per_block or attn_configs.tokens_per_block
        )

        self.q0_idx = self.q1_idx = None
        self.kv0_idx = self.kv1_idx = None
        self.kv_restore_unpad_indices = None
        self._fa4_prefix_page_table = None
        self._fa4_seq_lens_part0 = None
        self._fa4_seq_lens_part1 = None
        self._fa4_max_kv_len_part0 = 0
        self._fa4_max_kv_len_part1 = 0
        self._fa4_pool_lengths = None
        self._fa4_fused = False
        self._fa4_no_prefix = False

        self.prefill_wrappers = {
            "ragged": {
                name: BatchPrefillWithRaggedKVCacheWrapper(
                    self.workspace_buffer,
                    kv_layout=kv_layout,
                    backend=backend,
                )
                for name in ["part0", "part1"]
            },
            "paged": {
                "prefix": BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    kv_layout="HND",
                    backend=backend,
                ),
            },
        }
        self._can_use_trtllm_paged_context = self._can_use_trtllm_paged_context()

    def _should_use_forward_opt(self) -> bool:
        value = os.environ.get("RTP_LLM_CP_PREFILL_FORWARD_OPT", "1").strip()
        if value.lower() in ("0", "false", "no", "off"):
            return False
        # Paged-context backend must be available: FA4 (preferred) or trtllm.
        if not self._fa4_no_prefix and not self._can_use_trtllm_paged_context:
            return False
        # An fp8 KV cache would make this a mixed bf16-q / fp8-KV pass, which neither
        # FA4 (one dtype across q/k/v) nor trtllm supports here.
        if self.has_prefix or self._kv_sharded or self._fp8_kv_cache:
            return False

        return True

    def _can_use_trtllm_paged_context(self) -> bool:
        if torch.cuda.get_device_capability()[0] != 10:
            return False
        try:
            from flashinfer.artifacts import ArtifactPath, CheckSumHash
            from flashinfer.jit.attention.modules import get_artifact

            return bool(
                get_artifact(
                    f"{ArtifactPath.TRTLLM_GEN_FMHA}/checksums.txt",
                    CheckSumHash.TRTLLM_GEN_FMHA,
                )
            )
        except Exception as e:
            logger.warning("Disable CP prefill TRTLLM paged context: %s", e)
            return False

    def _physical_block_table(self) -> torch.Tensor:
        """Physical paged-cache block table (per-rank, CP-RR compact under
        sharding). Same table GLM5/DSV4/MSA use for paged cache I/O — addresses
        physical pages, not the (possibly token-level) kernel block table."""
        phys = getattr(self.attn_inputs, "kv_cache_block_id_device", None)
        if isinstance(phys, torch.Tensor) and phys.numel() > 0:
            return phys
        return self.attn_inputs.kv_cache_kernel_block_id_device

    def support(self, attention_inputs: PyAttentionInputs) -> bool:
        return attention_inputs.is_prefill

    def prepare(self, attention_inputs: PyAttentionInputs) -> ParamsBase:
        cu_seqlens = attention_inputs.cu_seqlens[
            : attention_inputs.input_lengths.size(0) + 1
        ]
        padding_mask = self.cp_info.prefill_qkv_padding_mask
        kv_restore_indices = self.cp_info.prefill_qkv_restore_indice
        self.kv_restore_unpad_indices = kv_restore_indices[padding_mask == 1]

        qo_indptr = cu_seqlens // 2

        self.has_prefix = self.attn_inputs.prefix_lengths.any().item()
        # Both backend choices are settled once here: the rest of prepare() skips
        # building metadata the chosen backend never reads, so forward must not
        # re-derive them and risk disagreeing.
        fa4_fused = self.has_prefix and _use_fa4_cp_paged(
            has_prefix=True, fp8_kv_cache=self._fp8_kv_cache
        )
        self._fa4_fused = fa4_fused
        self._fa4_no_prefix = _use_fa4_cp_paged(fp8_kv_cache=self._fp8_kv_cache)

        self.q0_idx, self.q1_idx = _generate_q_indices_device(
            self.cp_info.prefill_cp_chunk_lengths, self.device
        )
        if fa4_fused:
            # The fused path never gathers the extend K/V out of the all-gathered
            # activation, so these indices — part1's alone spans
            # ``2 * cp_size - rank`` chunks per request — would go unread.
            self.kv0_idx = self.kv1_idx = None
        else:
            kv0_idx, kv1_idx = _generate_full_causal_kv_indices_device(
                self.cp_info.prefill_cp_chunk_lengths,
                self.prefill_cp_rank,
                self.prefill_cp_size,
                self.device,
            )
            self.kv0_idx = kv_restore_indices[kv0_idx]
            self.kv1_idx = kv_restore_indices[kv1_idx]

        kv_block_id_host = self.attn_inputs.kv_cache_kernel_block_id_host
        if kv_block_id_host is None:
            kv_block_id_host = self.attn_inputs.kv_cache_block_id_host
        tokens_per_block = (
            self.attn_configs.kernel_tokens_per_block
            or self.attn_configs.tokens_per_block
        )
        if self._kv_sharded:
            kv_block_id_host = _build_cp_sharded_params_block_table(
                self.attn_inputs.prefix_lengths,
                self.cp_info.prefill_actual_input_lengths_cpu,
                tokens_per_block,
            )

        params = fill_mla_params(
            self.attn_inputs.prefix_lengths,
            self.attn_inputs.sequence_lengths,
            self.cp_info.prefill_actual_input_lengths_cpu,
            kv_block_id_host,
            tokens_per_block,
        )

        if fa4_fused and not PCPAllGatherAttnOp._logged_fa4_fused:
            PCPAllGatherAttnOp._logged_fa4_fused = True
            logger.info(
                "CP prefill cache hits take the fused FA4 paged path "
                "(kv_cache_dtype=%s, kv_sharded=%s)",
                self.attn_configs.kv_cache_dtype,
                self._kv_sharded,
            )

        self._plan_ragged(qo_indptr, plan_wrappers=not fa4_fused)
        q_lens = qo_indptr[1:] - qo_indptr[:-1]
        self._trtllm_max_q_len = int(q_lens.max().item())
        self._use_forward_opt = self._should_use_forward_opt()
        if self._use_forward_opt:
            # Only _forward_opt reads this metadata, and a cache hit never selects
            # that path, so building it unconditionally is pure overhead.
            (
                self._trtllm_seq_lens_part0,
                self._trtllm_cu_kv_pages_part0,
                self._trtllm_max_kv_len_part0,
            ) = self._build_trtllm_paged_context_metadata(self.kv_indptr_part0)
            (
                self._trtllm_seq_lens_part1,
                self._trtllm_cu_kv_pages_part1,
                self._trtllm_max_kv_len_part1,
            ) = self._build_trtllm_paged_context_metadata(self.kv_indptr_part1)
        if fa4_fused:
            self._prepare_fa4_prefix_metadata()
        elif self.has_prefix:
            plan_prefix_paged_attention(
                self.prefill_wrappers["paged"]["prefix"],
                cu_seqlens,
                attention_inputs.prefix_lengths,
                params,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.seq_size_per_block,
                device=self.device,
                kv_data_type=(
                    torch.float8_e4m3fn if self._fp8_kv_cache else torch.bfloat16
                ),
                contiguous_page_indices=self._kv_sharded,
            )
        return params

    def _prepare_fa4_prefix_metadata(self) -> None:
        """Per-forward metadata for the fused FA4 paged passes.

        Each CP part reads its prefix and its extend in one causal pass, so its key
        extent is the prefix plus the same causal extent ``kv_indptr_part{0,1}``
        encodes for the ragged passes: ``h * (rank + 1)`` and
        ``h * (2 * cp_size - rank)`` for a per-part query length ``h``.

        Under CP sharding the pool comes from ``gather_cp_sharded_prefix_pool``,
        which is page-granular, so the gathered extent is the page-aligned ceiling
        of part1's (which dominates part0's at every rank) and each part's
        ``seqused_k`` truncates inside the last page. Only that gathered pool needs
        the synthetic logical page table; the rank-local physical table already
        lists each request's logical pages in order, new tokens included.
        """
        page_size = self.seq_size_per_block
        # Keeping the arithmetic on host tensors avoids one device->host sync per
        # scalar read, three per layer per forward. The engine's qo_indptr and
        # prefix_lengths are already host-resident, so to("cpu") is a no-op here and
        # only copies for a caller that hands over device-resident lengths.
        host = {"device": "cpu", "dtype": torch.int32}
        q_lens = (self.qo_indptr[1:] - self.qo_indptr[:-1]).to(**host)
        prefix_lengths = self.attn_inputs.prefix_lengths.to(**host)
        seq_lens_part0 = prefix_lengths + q_lens * (self.prefill_cp_rank + 1)
        seq_lens_part1 = prefix_lengths + q_lens * (
            2 * self.prefill_cp_size - self.prefill_cp_rank
        )
        self._fa4_max_kv_len_part0 = int(seq_lens_part0.max())
        self._fa4_max_kv_len_part1 = int(seq_lens_part1.max())

        # Stays on the host: the gather plan wants its lengths there.
        self._fa4_pool_lengths = _page_counts(seq_lens_part1, page_size) * page_size

        # seqused_k is read on device; one copy carries both parts.
        seq_lens = torch.stack((seq_lens_part0, seq_lens_part1)).to(
            self.device, non_blocking=True
        )
        self._fa4_seq_lens_part0 = seq_lens[0]
        self._fa4_seq_lens_part1 = seq_lens[1]

        if self._kv_sharded:
            self._fa4_prefix_page_table = _build_contiguous_prefix_page_table(
                self._fa4_pool_lengths,
                page_size,
                int(self._fa4_pool_lengths.max()) // page_size,
                device=self.device,
            )
        else:
            self._fa4_prefix_page_table = (
                self._physical_block_table().to(torch.int32).contiguous()
            )

    def _plan_ragged(self, qo_indptr: torch.Tensor, *, plan_wrappers: bool) -> None:
        """Publish the per-part query/key offsets, which always define each CP
        part's causal extent, and plan the FlashInfer ragged wrappers unless the
        caller settled on a backend that never runs them.
        """
        self.qo_indptr = qo_indptr
        kv_indptr_part0 = qo_indptr * (self.prefill_cp_rank + 1)
        kv_indptr_part1 = qo_indptr * (2 * self.prefill_cp_size - self.prefill_cp_rank)
        self.kv_indptr_part0 = kv_indptr_part0
        self.kv_indptr_part1 = kv_indptr_part1
        if not plan_wrappers:
            return
        common_params = {
            "num_qo_heads": self.num_qo_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim_qk": self.head_dim,
            "causal": True,
            "q_data_type": torch.bfloat16,
        }
        self.prefill_wrappers["ragged"]["part0"].plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr_part0,
            **common_params,
        )
        self.prefill_wrappers["ragged"]["part1"].plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr_part1,
            **common_params,
        )

    def _run_ragged_part(
        self,
        part: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        return_lse: bool = False,
    ):
        return self.prefill_wrappers["ragged"][part].run(q, k, v, return_lse=return_lse)

    def _build_trtllm_paged_context_metadata(
        self, kv_indptr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        seq_lens = (kv_indptr[1:] - kv_indptr[:-1]).to(
            device=self.device, dtype=torch.int32
        )
        pages_per_seq = _page_counts(seq_lens, self.seq_size_per_block)
        cu_kv_pages = torch.empty(
            seq_lens.numel() + 1, device=self.device, dtype=torch.int32
        )
        cu_kv_pages[0] = 0
        torch.cumsum(pages_per_seq, dim=0, out=cu_kv_pages[1:])
        return seq_lens, cu_kv_pages, int(seq_lens.max().item())

    def _run_trtllm_paged_context(
        self,
        q: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_lens: torch.Tensor,
        cu_kv_pages: torch.Tensor,
        max_kv_len: int,
        block_tables: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if block_tables is None:
            block_tables = self._physical_block_table()
        out = trtllm_batch_context_with_kv_cache(
            query=q,
            kv_cache=kv_cache_tensor,
            workspace_buffer=get_cp_trt_workspace_buffer(),
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_q_len=self._trtllm_max_q_len,
            max_kv_len=max_kv_len,
            bmm1_scale=self.head_dim**-0.5,
            bmm2_scale=1.0,
            batch_size=seq_lens.numel(),
            cum_seq_lens_q=self.qo_indptr,
            cum_seq_lens_kv=cu_kv_pages,
            window_left=-1,
            sinks=None,
            out_dtype=q.dtype,
        )
        return out

    def _run_fa4_paged_context(
        self,
        q: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_lens: torch.Tensor,
        max_kv_len: int,
        block_tables: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """FA4 paged causal attention over the HND pool.

        The HND pool ``[blocks, 2, H_kv, page, D]`` is fed to FA4 (which wants
        ``[num_pages, page, H_kv, D]``) via a zero-copy ``transpose(1, 2)`` view;
        FA4's TMA handles the strided K/V at full speed.
        """
        if block_tables is None:
            block_tables = self._physical_block_table()
        k = kv_cache_tensor[:, 0].transpose(1, 2)
        v = kv_cache_tensor[:, 1].transpose(1, 2)
        q = _match_q_to_kv(q, k)
        out = _fa4_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=self.qo_indptr,
            max_seqlen_q=self._trtllm_max_q_len,
            max_seqlen_k=int(max_kv_len),
            seqused_k=seq_lens,
            page_table=block_tables,
            causal=True,
            softmax_scale=self.head_dim**-0.5,
        )
        return out[0] if isinstance(out, tuple) else out

    def _run_paged_context(
        self,
        q: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_lens: torch.Tensor,
        cu_kv_pages: torch.Tensor,
        max_kv_len: int,
    ) -> torch.Tensor:
        """Dispatch the CP paged-context attention to FA4 (default) or trtllm."""
        if self._fa4_no_prefix:
            return self._run_fa4_paged_context(q, kv_cache_tensor, seq_lens, max_kv_len)
        return self._run_trtllm_paged_context(
            q, kv_cache_tensor, seq_lens, cu_kv_pages, max_kv_len
        )

    def _run_fa4_ragged(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ) -> torch.Tensor:
        """FA4 varlen (ragged) causal attention over contiguous per-part k/v.

        Causal offset is ``seqlen_k - seqlen_q`` (bottom-right aligned), matching
        the fmha and trtllm ragged paths.
        """
        max_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
        max_k = int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item())
        q = _match_q_to_kv(q, k)
        out, _ = _fa4_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            causal=True,
            softmax_scale=self.head_dim**-0.5,
            return_lse=False,
        )
        return out

    def _run_fa4_paged_part(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seqused_k: torch.Tensor,
        max_kv_len: int,
    ) -> torch.Tensor:
        """One causal FA4 paged pass covering a CP part's prefix *and* its extend.

        The KV cache write precedes attention, so the pool already holds this
        forward's new K/V and a single pass can span both regions. ``causal=True``
        reproduces the zigzag mask exactly: FA4 aligns the last query with the last
        key, and this part's last query is the new token at absolute position
        ``seqused_k - 1``, so query ``j`` reaches ``prefix + c * h + j`` for chunk
        ``c`` — the per-chunk causal extent that ``kv_indptr_part{0,1}`` encodes for
        the ragged passes.
        """
        out, _ = _fa4_varlen_func(
            _match_q_to_kv(q, k),
            k,
            v,
            cu_seqlens_q=self.qo_indptr,
            max_seqlen_q=self._trtllm_max_q_len,
            max_seqlen_k=max_kv_len,
            seqused_k=seqused_k,
            page_table=self._fa4_prefix_page_table,
            causal=True,
            softmax_scale=self.head_dim**-0.5,
            return_lse=False,
        )
        return out

    def _forward_opt(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        params: ParamsBase = None,
    ) -> Optional[torch.Tensor]:
        qkv = qkv.reshape(qkv.shape[0], -1)
        q_size = self.head_dim * self.num_qo_heads
        packed_kv_size = 2 * self.head_dim * self.num_kv_heads
        q = qkv[:, :q_size]
        packed_kv = qkv[:, q_size : q_size + packed_kv_size].contiguous()

        all_packed_kv = all_gather(packed_kv, group=Group.TP).reshape(
            packed_kv.shape[0] * self.prefill_cp_size,
            2,
            self.num_kv_heads,
            self.head_dim,
        )
        q_reshaped = q.reshape(-1, self.num_qo_heads, self.head_dim)

        restore_k, restore_v = _fused_restore_packed_kv(
            all_packed_kv,
            self.kv_restore_unpad_indices,
            self.num_kv_heads,
            self.head_dim,
        )
        restore_token_count = restore_k.size(0)
        batch_indices = params.batch_indice_d.narrow(0, 0, restore_token_count)
        positions = params.positions_d.narrow(0, 0, restore_token_count)
        kv_cache_tensor = kv_cache.kv_cache_base.view(
            -1, 2, self.num_kv_heads, self.seq_size_per_block, self.head_dim
        )
        append_paged_kv_cache(
            append_key=restore_k,
            append_value=restore_v,
            batch_indices=batch_indices,
            positions=positions,
            paged_kv_cache=kv_cache_tensor,
            kv_indices=params.page_indice_d,
            kv_indptr=params.decode_page_indptr_d,
            kv_last_page_len=params.paged_kv_last_page_len_d,
            kv_layout="HND",
        )

        q0 = torch.index_select(q_reshaped, 0, self.q0_idx).contiguous()
        q1 = torch.index_select(q_reshaped, 0, self.q1_idx).contiguous()

        output = torch.empty_like(q_reshaped)
        output[self.q0_idx] = self._run_paged_context(
            q0,
            kv_cache_tensor,
            self._trtllm_seq_lens_part0,
            self._trtllm_cu_kv_pages_part0,
            self._trtllm_max_kv_len_part0,
        )
        output[self.q1_idx] = self._run_paged_context(
            q1,
            kv_cache_tensor,
            self._trtllm_seq_lens_part1,
            self._trtllm_cu_kv_pages_part1,
            self._trtllm_max_kv_len_part1,
        )
        return output

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        params: ParamsBase = None,
    ) -> torch.Tensor:
        if self._use_forward_opt:
            output = self._forward_opt(qkv, kv_cache, params)
            if output is not None:
                return output

        qkv = qkv.reshape(qkv.shape[0], -1)
        q, k, v = torch.split(
            qkv,
            [
                self.head_dim * self.num_qo_heads,
                self.head_dim * self.num_kv_heads,
                self.head_dim * self.num_kv_heads,
            ],
            dim=-1,
        )
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        all_keys = all_gather(k, group=Group.TP).reshape(
            k.shape[0] * self.prefill_cp_size, self.num_kv_heads, self.head_dim
        )
        all_values = all_gather(v, group=Group.TP).reshape(
            v.shape[0] * self.prefill_cp_size, self.num_kv_heads, self.head_dim
        )
        q_reshaped = q.reshape(-1, self.num_qo_heads, self.head_dim)
        del k, v

        # TODO: make write local kvcache async
        restore_k = all_keys[self.kv_restore_unpad_indices]
        restore_v = all_values[self.kv_restore_unpad_indices]
        nnz = restore_k.size(0)
        batch_indices = params.batch_indice_d.narrow(0, 0, nnz)
        positions = params.positions_d.narrow(0, 0, nnz)
        kv_cache_tensor = kv_cache.kv_cache_base.view(
            -1, 2, self.num_kv_heads, self.seq_size_per_block, self.head_dim
        )
        if self._kv_sharded:
            # The local block table only holds this rank's 1/cp_size owned blocks, so
            # map each all-gathered token to a physical slot (non-owned and
            # out-of-capacity become -1, which the writer skips). What lands is
            # exactly this rank's page-RR shard, matching the MSA layers and decode's
            # per-rank pool reader.
            from rtp_llm.models_py.modules.dsv4.fp8._cp_slot_mapping import (
                cp_kv_slot_mapping,
            )
            from rtp_llm.ops.compute_ops import rtp_llm_ops

            bt = self._physical_block_table().to(torch.int64)
            slot_mapping = cp_kv_slot_mapping(
                positions.to(torch.int64),
                bt,
                batch_indices.to(torch.int64),
                self.seq_size_per_block,  # tokens_per_block
                self.seq_size_per_block,  # kv_eb (entries per block, ratio=1)
                1,  # ratio (uncompressed MHA/GQA K/V)
                self._cp_size,
                self._cp_rank,
                owner_tokens_per_block=self.seq_size_per_block,
            )
            rtp_llm_ops.mha_kv_write_cache(
                # This writer casts to FP8 itself, so it takes activation dtypes;
                # pre-cast float8 inputs violate its contract.
                restore_k.contiguous(),
                restore_v.contiguous(),
                kv_cache_tensor,
                slot_mapping,
            )
        else:
            append_k, append_v = cast_kv_for_cache_append(
                restore_k, restore_v, kv_cache, self.attn_configs.kv_cache_dtype
            )
            append_paged_kv_cache(
                append_key=append_k,
                append_value=append_v,
                batch_indices=batch_indices,
                positions=positions,
                paged_kv_cache=kv_cache_tensor,
                kv_indices=params.page_indice_d,
                kv_indptr=params.decode_page_indptr_d,
                kv_last_page_len=params.paged_kv_last_page_len_d,
                kv_layout="HND",
            )
            # Append-path only: params.page_indice_d is full-length and would index
            # past a sharded block table.
            fill_fp8_kv_cache_scale(
                kv_cache,
                params,
                batch_indices,
                positions,
                num_kv_heads=self.num_kv_heads,
                page_size=self.seq_size_per_block,
                kv_cache_dtype=self.attn_configs.kv_cache_dtype,
            )

        q0 = torch.index_select(q_reshaped, 0, self.q0_idx).contiguous()
        q1 = torch.index_select(q_reshaped, 0, self.q1_idx).contiguous()

        if self._fa4_fused:
            # The cache write above already placed this forward's K/V in the pool, so
            # each CP part is a single causal paged pass spanning its prefix and its
            # extend: no ragged pass, no LSE merge, and the extend K/V never has to be
            # gathered out of the all-gathered activation.
            pool = kv_cache_tensor
            if self._kv_sharded:
                pool = gather_cp_sharded_prefix_pool(
                    kv_cache_tensor,
                    self._physical_block_table(),
                    self._fa4_pool_lengths,
                    page_size=self.seq_size_per_block,
                    cp_size=self._cp_size,
                    cp_rank=self._cp_rank,
                )
            # Both parts read one pool, so these zero-copy views are shared. No
            # prefix dequantisation is materialised: the pool is read in its stored
            # dtype and q follows it.
            k = pool[:, 0].transpose(1, 2)
            v = pool[:, 1].transpose(1, 2)
            output = torch.empty_like(q_reshaped)
            output[self.q0_idx] = self._run_fa4_paged_part(
                q0, k, v, self._fa4_seq_lens_part0, self._fa4_max_kv_len_part0
            )
            output[self.q1_idx] = self._run_fa4_paged_part(
                q1, k, v, self._fa4_seq_lens_part1, self._fa4_max_kv_len_part1
            )
            return output

        k0 = torch.index_select(all_keys, 0, self.kv0_idx).contiguous()
        k1 = torch.index_select(all_keys, 0, self.kv1_idx).contiguous()
        v0 = torch.index_select(all_values, 0, self.kv0_idx).contiguous()
        v1 = torch.index_select(all_values, 0, self.kv1_idx).contiguous()
        if self.has_prefix:
            # FlashInfer keeps q and the extend in bf16 and dequantises only the
            # cached prefix, so a cache hit stays a non-causal paged pass over the
            # prefix plus a causal ragged pass per CP part, combined through the LSE.
            prefix_kv_cache_tensor = kv_cache_tensor
            if self._kv_sharded:
                prefix_kv_cache_tensor = gather_cp_sharded_prefix_pool(
                    kv_cache_tensor,
                    self._physical_block_table(),
                    self.attn_inputs.prefix_lengths,
                    page_size=self.seq_size_per_block,
                    cp_size=self._cp_size,
                    cp_rank=self._cp_rank,
                )
            prefix_out, prefix_lse = self.prefill_wrappers["paged"]["prefix"].run(
                q_reshaped, prefix_kv_cache_tensor, return_lse=True
            )
            out0, lse0 = self._run_ragged_part("part0", q0, k0, v0, return_lse=True)
            out1, lse1 = self._run_ragged_part("part1", q1, k1, v1, return_lse=True)

            out0, _ = merge_state(
                v_a=prefix_out[self.q0_idx],
                s_a=prefix_lse[self.q0_idx],
                v_b=out0,
                s_b=lse0,
            )
            out1, _ = merge_state(
                v_a=prefix_out[self.q1_idx],
                s_a=prefix_lse[self.q1_idx],
                v_b=out1,
                s_b=lse1,
            )
            output = torch.empty_like(q_reshaped)
            output[self.q0_idx] = out0
            output[self.q1_idx] = out1
            return output
        else:
            output = torch.empty_like(q_reshaped)
            if self._fa4_no_prefix:
                # Cast the gathered extend K/V to the cache dtype so an fp8 cache
                # still gets uniform-fp8 attention; a no-op for bf16.
                kv_dtype = kv_cache_tensor.dtype
                output[self.q0_idx] = self._run_fa4_ragged(
                    q0,
                    k0.to(kv_dtype),
                    v0.to(kv_dtype),
                    self.qo_indptr,
                    self.kv_indptr_part0,
                )
                output[self.q1_idx] = self._run_fa4_ragged(
                    q1,
                    k1.to(kv_dtype),
                    v1.to(kv_dtype),
                    self.qo_indptr,
                    self.kv_indptr_part1,
                )
            else:
                output[self.q0_idx] = self._run_ragged_part("part0", q0, k0, v0)
                output[self.q1_idx] = self._run_ragged_part("part1", q1, k1, v1)
            return output
