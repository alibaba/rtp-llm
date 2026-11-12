"""Pinned SG late sparse scoring over the actual local native indexer pool.

The caller mints publication only after SG dense selection and final row
compaction. Candidate values / visible bounds are then trusted producer
outputs, not arbitrary user tensors. Execution failures never retry serially.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from weakref import ref

import torch

from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache, _source_tensor_key


SOURCE, PAGE, BLOCK, CANDIDATES = 20, 256, 8, 2048


def _raw(value):
    return (isinstance(value, torch.Tensor) and value.layout == torch.strided
            and not value.is_neg() and not value.is_conj() and not value.requires_grad)


@dataclass
class Publication:
    cache: object
    candidates: torch.Tensor
    visible: torch.Tensor
    pool: object
    nonce: object
    keys: tuple
    source_read_proof: object = field(default=None, repr=False)
    _plan: object = field(default=None, repr=False)


@dataclass(frozen=True)
class PagedPlan:
    _publication_ref: object
    blocks: torch.Tensor
    phys_blocks: torch.Tensor
    visible: torch.Tensor
    valid_lens: torch.Tensor
    page_table: torch.Tensor
    request_ids: torch.Tensor
    metadata: torch.Tensor
    pool_view: torch.Tensor
    keys: tuple

    @property
    def publication(self):
        return self._publication_ref()


def _table(cache, pool, kind="indexer_kv"):
    # L20 has already read its exact source bank. Do not introduce another
    # mirror/DtoH when making or consuming a paged schedule.
    mirrors = getattr(cache, "_local_source_native_read_mirrors", {})
    if pool.framework_tag not in mirrors:
        return None
    return cache._verified_local_source_native_table((SOURCE, kind), pool)


def _valid(publication):
    if not isinstance(publication, Publication):
        return False
    cache, pool = publication.cache, publication.pool
    if (type(cache) is not SharedNativeCache or not getattr(cache, "_local_sg_source_enabled", False)
            or cache._local_source_native_read_stats["closed"]
            or cache._local_source_native_read_nonce is not publication.nonce
            or cache._local_source_native_read_nonce is not cache._local_source_native_read_birth_nonce
            or cache.pools.get((SOURCE, "indexer_kv")) is not pool
            or torch.is_grad_enabled() or not cache._is_prefill or cache.sharded):
        return False
    owners = publication.candidates, publication.visible
    main = cache.pools.get((SOURCE, "global_kv_1"))
    return (all(_raw(value) for value in owners)
            and tuple(_source_tensor_key(value) for value in owners) == publication.keys
            and publication.source_read_proof is not None and cache._check_local_source_reads()
            and cache._local_source_reads.get((SOURCE, 0)) is publication.source_read_proof
            and main is not None and _table(cache, main, "global_kv_1") is not None
            and _table(cache, pool) is not None)


def make_publication(cache, candidates, visible, *, source=SOURCE):
    """Trusted completion hook, after final SG L20 candidate publication.

    The caller guarantees candidates in [-1,ceil(visible/8)), unique except
    padding, with the newest block included, and 0<=visible<=request length.
    SG's native bitmap transform does not validate arbitrary positive IDs.
    """
    if (source != SOURCE or type(cache) is not SharedNativeCache
            or not getattr(cache, "_local_sg_source_enabled", False)
            or torch.is_grad_enabled() or len(cache.lengths) != 1
            or cache.geometry.ratio(SOURCE) != 1 or not cache._is_prefill or cache.sharded
            or not all(_raw(value) for value in (candidates, visible))
            or candidates.dtype != torch.int32 or candidates.ndim != 2
            or not 0 < candidates.shape[0] <= 4096 or candidates.shape[1] != CANDIDATES
            or not candidates.is_contiguous() or visible.dtype not in (torch.int32, torch.int64)
            or visible.shape != (candidates.shape[0],) or not visible.is_contiguous()
            or candidates.device != visible.device or not BLOCK * CANDIDATES < cache.lengths[0] <= 131072):
        return None
    pool = cache.pools.get((SOURCE, "indexer_kv"))
    if (pool is None or not _raw(pool.tensor) or not _raw(pool.table)
            or pool.tensor.dtype != torch.uint8 or pool.tensor.ndim != 2
            or not pool.tensor.is_contiguous() or pool.tensor.device != candidates.device
            or (pool.entries, pool.span, pool.tensor.stride(0)) != (PAGE, PAGE, PAGE * 68)
            or pool.table.dtype != torch.int32 or pool.table.stride(1) != 1
            or torch._C._is_alias_of(candidates, pool.tensor)
            or torch._C._is_alias_of(visible, pool.tensor)
            or torch._C._is_alias_of(candidates, pool.table)
            or torch._C._is_alias_of(visible, pool.table)
            or torch._C._is_alias_of(candidates, visible)):
        return None
    publication = Publication(cache, candidates, visible, pool,
                              cache._local_source_native_read_nonce,
                              tuple(_source_tensor_key(value) for value in (candidates, visible)),
                              source_read_proof=cache._local_source_reads.get((SOURCE, 0)))
    if not _valid(publication):
        return None
    mirror = _table(cache, pool)
    pages = (cache.lengths[0] + PAGE - 1) // PAGE
    if mirror.shape[1] < pages:
        return None
    physical = tuple(map(int, mirror[0, :pages].tolist()))  # CPU mirror only.
    if not physical or min(physical) < 0 or max(physical) >= pool.tensor.shape[0]:
        return None
    # Physical page zero and repeated/reordered physical IDs are legal reads.
    return publication


@lru_cache(maxsize=1)
def _deep_gemm():
    try:
        import deep_gemm
    except ImportError:
        return None
    return deep_gemm if all(callable(getattr(deep_gemm, name, None)) for name in (
        "get_paged_sparse_mqa_logits_metadata", "fp8_fp4_paged_sparse_mqa_logits",
    )) else None


@lru_cache(maxsize=1)
def _sort_module():
    from rtp_llm.models_py.kernels.ppu.jit.sglang_jit import load_sglang_kernel

    return load_sglang_kernel(
        "prefill_candidate_table", "csrc/deepseek_v4/candidate_block_table.cuh",
        "sglang::CandidateBlockTableKernel<false>::transform", (8, 9),
        bundle="prefill_candidate_table",
    )


def _plan_keys(plan):
    return tuple(_source_tensor_key(value) for value in (
        plan.blocks, plan.phys_blocks, plan.visible, plan.valid_lens,
        plan.page_table, plan.request_ids, plan.metadata,
    )) + (_source_tensor_key(plan.pool_view)[:-1],)


def prepare(cache, candidates, visible, *, publication):
    """One SG native table/schedule, reused by L24/28/32/36; None is prelaunch."""
    if (not _valid(publication) or publication.cache is not cache
            or publication.candidates is not candidates or publication.visible is not visible):
        return None
    if publication._plan is not None:
        plan = publication._plan
        return plan if plan.keys == _plan_keys(plan) else None
    dg = _deep_gemm()
    if dg is None:
        return None
    sort = _sort_module()  # Resolve dependencies before mutating private buffers.
    rows, count = candidates.shape[0], cache.lengths[0]
    pages = (count + PAGE - 1) // PAGE
    table = publication.pool.table[:1, :pages].expand(rows, -1).contiguous()
    blocks, phys = candidates.clone(), torch.empty_like(candidates)
    lengths = visible.to(torch.int32).contiguous()
    # Exact SG candidate_row_lens integer formula. No float computation or
    # host download; newest block is part of the trusted SG publication.
    kept = torch.minimum((lengths + BLOCK - 1) // BLOCK, torch.full_like(lengths, CANDIDATES))
    valid_lens = torch.where(lengths > 0, (kept - 1) * BLOCK + (lengths - 1) % BLOCK + 1, 0)
    request_ids = torch.tensor(tuple(row & ~1 for row in range(rows)), dtype=torch.int32, device=visible.device)
    sort.forward(blocks, lengths, table, phys, PAGE)
    metadata = dg.get_paged_sparse_mqa_logits_metadata(lengths, table, request_ids, PAGE,
                                                      blocks, torch.int8, BLOCK)
    pool = publication.pool.tensor
    view = pool.as_strided((pool.shape[0], PAGE, 1, 68), (pool.stride(0), 68, 68, 1))
    provisional = PagedPlan(ref(publication), blocks, phys, lengths, valid_lens, table,
                           request_ids, metadata, view, ())
    plan = PagedPlan(ref(publication), blocks, phys, lengths, valid_lens, table,
                    request_ids, metadata, view, _plan_keys(provisional))
    publication._plan = plan
    return plan


def score(q_payload, q_sf, weights, plan):
    """Exact SG paged sparse provider boundary; BF16 weights and logits."""
    if not isinstance(plan, PagedPlan) or not _valid(plan.publication) or plan.keys != _plan_keys(plan):
        raise RuntimeError("stale local paged sparse plan")
    rows = plan.blocks.shape[0]
    if (not all(_raw(value) for value in (q_payload, q_sf, weights))
            or q_payload.dtype != torch.int8 or q_payload.shape != (rows, 32, 64)
            or q_sf.dtype != torch.int32 or q_sf.shape != (rows, 32)
            or not q_payload.is_contiguous() or not q_sf.is_contiguous()
            or weights.dtype not in (torch.bfloat16, torch.float32) or weights.shape != (rows, 32)
            or weights.stride(1) != 1 or weights.stride(0) < 32
            or any(value.device != plan.pool_view.device for value in (q_payload, q_sf, weights))):
        raise ValueError("unsupported local paged sparse query metadata")
    logits = _deep_gemm().fp8_fp4_paged_sparse_mqa_logits(
        (q_payload.view(rows, 1, 32, 64), q_sf.view(rows, 1, 32)),
        plan.pool_view, weights.to(torch.bfloat16), plan.metadata, CANDIDATES, BLOCK,
    )
    if logits.dtype != torch.bfloat16 or logits.shape != (rows, CANDIDATES * BLOCK):
        raise RuntimeError("unexpected SG paged sparse logits ABI")
    return logits
