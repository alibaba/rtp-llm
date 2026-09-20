"""DeepSeek V4.1 shared global attention over RTP's typed CP/PD pools.

The initial eager implementation deliberately shares the established SWA and
FlashMLA kernels with V4. Global compression is non-overlapping (ratio 1/2),
and index keys are projected from its normalized, pre-RoPE latent. Only source
layers write global pools; consumers reuse source KV and index selections.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.attn_type import (
    CSA_KV,
    CSA_STATE,
    HCA_KV,
    INDEXER_KV,
    SWA_KV,
)
from rtp_llm.models_py.modules.dsv4.cp import _cp_restore_gathered_full_2d
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_deepselect as prefill_deepselect
from rtp_llm.models_py.modules.dsv4.fp8 import (
    _v41_prefill_candidates as prefill_candidates,
)
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_indexer as prefill_indexer
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_topk as prefill_topk
from rtp_llm.models_py.modules.dsv4.fp8 import (
    _v41_sparse_prefill_indexer as sparse_prefill_indexer,
)
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_swa_triton as swa_codec
from rtp_llm.models_py.modules.dsv4.fp8._cp_slot_mapping import (
    cp_kv_slot_mapping,
    cp_state_slot_mapping,
)
from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import (
    require_pool_tokens_per_block,
)
from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_triton import (
    FP4_GLOBAL_ENTRY_BYTES,
    FP4_INDEXER_ENTRY_BYTES,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    _ATTN_TYPE_ENUM_BY_INT,
    AttentionFP8,
    _get_cp_comm_stream,
)
from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb, precompute_freqs_cis
from rtp_llm.models_py.modules.dsv4.utils import _v4_fp8_linear
from rtp_llm.models_py.utils.memory import dispose_tensor
from rtp_llm.ops.compute_ops import rtp_llm_ops

# Bound both CP hidden-state transfers and the subsequent projections.
_PRODUCE_GLOBAL_TILE_ROWS = 32768

# Single-shot CP x-gather ceiling in GLOBAL padded rows. Below it, one bounded
# all-gather of the raw hidden states replaces the per-(request, zigzag-half)
# owner-projected broadcast tiles. Mid-size multi-request prefill batches
# otherwise issue one small NCCL broadcast per segment on every kv-source
# layer, and each broadcast kernel's duration is dominated by inter-rank
# arrival skew rather than transfer (measured ~220ms of broadcast-kernel wait
# per 6-request c8 forward, while the single-shot path shows none). Above the
# ceiling the tiled path keeps the ~20x smaller projected-fp32 transfer volume
# for large-context single requests. Default covers the 8x8192-token
# shared-prefix batch geometry; set DSV41_SMALL_CP_X_GATHER_MAX_ROWS=32768 to
# restore the previous boundary.
_SMALL_CP_X_GATHER_MAX_ROWS = int(
    os.environ.get("DSV41_SMALL_CP_X_GATHER_MAX_ROWS", "65536")
)


def _use_small_cp_x_gather(cp_ctx) -> bool:
    """Whether the single-shot raw-x all-gather path serves this forward."""
    return cp_ctx.padded_seq_len <= _SMALL_CP_X_GATHER_MAX_ROWS


class _V41AsyncXGather:
    """One in-flight, globally ordered hidden-state tile."""

    __slots__ = ("work", "completion_event", "gathered")

    def __init__(self, work, completion_event, gathered):
        self.work = work
        self.completion_event = completion_event
        self.gathered = gathered


def _prefill_x_tile_plan(cp_ctx):
    """Visit each request's zigzag halves in global order, excluding padding."""
    lengths = cp_ctx.input_lengths_global_host
    if lengths is None:
        lengths = cp_ctx.input_lengths_global.tolist()
    local_base = 0
    for length, chunk in zip(lengths, cp_ctx.chunk_lengths_per_req):
        half = chunk // 2
        for segment in range(2 * cp_ctx.cp_size):
            rows = min(half, max(0, length - segment * half))
            front = segment < cp_ctx.cp_size
            owner = segment if front else 2 * cp_ctx.cp_size - segment - 1
            local_start = local_base + (0 if front else half)
            for offset in range(0, rows, _PRODUCE_GLOBAL_TILE_ROWS):
                yield owner, local_start + offset, min(
                    _PRODUCE_GLOBAL_TILE_ROWS, rows - offset
                )
        local_base += chunk


def _start_prefill_x_gather_async(x: torch.Tensor, cp_ctx, tile, buffer, weights=()):
    """Transfer a bounded hidden-state batch or owner-projected zigzag tile."""
    from rtp_llm.models_py.distributed import collective_torch
    from rtp_llm.models_py.distributed.collective_torch import Group

    owner, start, rows = tile
    process_group = collective_torch._get_group(Group.TP)
    gathered = buffer[:rows]
    if owner == cp_ctx.cp_rank:
        if weights:
            from rtp_llm.models_py.modules.dsv4.fp8.compressor import (
                _linear_bf16_bf16_fp32,
            )

            head = weights[0].shape[0]
            for i, weight in enumerate(weights):
                gathered[:, i * head : (i + 1) * head].copy_(
                    _linear_bf16_bf16_fp32(x[start : start + rows], weight)
                )
        else:
            gathered.copy_(x[start : start + rows])
    current = torch.cuda.current_stream(x.device)
    stream = _get_cp_comm_stream(x.device)
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        work = (
            torch.distributed.all_gather_into_tensor(
                gathered, x.contiguous(), group=process_group, async_op=True
            )
            if owner is None
            else torch.distributed.broadcast(
                gathered,
                src=torch.distributed.get_global_rank(process_group, owner),
                group=process_group,
                async_op=True,
            )
        )
        completion_event = torch.cuda.Event()
        completion_event.record(stream)
    return _V41AsyncXGather(work, completion_event, gathered)


def _wait_prefill_x_gather(handle: _V41AsyncXGather) -> torch.Tensor:
    current = torch.cuda.current_stream(handle.gathered.device)
    current.wait_event(handle.completion_event)
    handle.work.wait()
    return handle.gathered


def _prefill_x_tiles(x, cp_ctx, plan, pending, buffers, weights=()):
    offset = 0
    next_buffer = 1
    while pending is not None:
        tile = _wait_prefill_x_gather(pending)
        next_tile = next(plan, None)
        pending = (
            _start_prefill_x_gather_async(
                x, cp_ctx, next_tile, buffers[next_buffer], weights
            )
            if next_tile is not None
            else None
        )
        yield offset, tile
        offset += tile.shape[0]
        next_buffer = 1 - next_buffer
        del tile


def rms_norm(x, weight, eps):
    """Preserve the BF16 normalization boundary used by V4.1."""
    # BF16 CUDA sites run on the framework native RMSNorm (the same
    # ``rtp_llm_ops.rmsnorm`` single-launch kernel the V4 path uses in
    # ``_rmsnorm_weighted``): fp32 accumulation, at most one BF16 ulp from
    # this reference formula. FP32-boundary sites (compressor projections)
    # keep the reference path; the op is BF16-only (fp32 input would NaN).
    if (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim >= 2
        and x.numel() > 0
        and x.is_contiguous()
        and weight.is_cuda
        and weight.device == x.device
        and weight.dtype == torch.bfloat16
        and weight.ndim == 1
        and weight.shape[0] == x.shape[-1]
        and weight.is_contiguous()
    ):
        flat = x.view(-1, x.shape[-1])
        out = torch.empty_like(flat)
        rtp_llm_ops.rmsnorm(
            out, flat, weight, eps, torch.cuda.current_stream().cuda_stream
        )
        return out.view(x.shape)
    xf = x.float()
    return (
        xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight.float()
    ).to(x.dtype)


def compress_pairs(values, scores, norm_weight, eps):
    """Pool consecutive non-overlapping pairs independently for each channel."""
    weights = torch.softmax(scores.float(), dim=-2)
    latent = (values.float() * weights).sum(-2)
    return rms_norm(latent, norm_weight, eps).to(torch.bfloat16)


def rope_only(x, freqs, rope_dim):
    apply_rotary_emb(x[..., -rope_dim:].unsqueeze(0), freqs)
    return x


def fp8_roundtrip(x):
    scale = (x.float().abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-12)
    return (x.float() / scale).to(torch.float8_e4m3fn).float() * scale


def fp4_roundtrip(x):
    """Fake-quantize the trailing 128-dim to the indexer MX FP4 form.

    Keeps the non-paged decode fallback numerically identical to the
    DeepGEMM paged scorer (same group-32 UE8M0 quantization of Q).
    """
    from rtp_llm.models_py.modules.dsv4.fp8._v41_prefill_indexer import _fp4_rows_torch

    shape = x.shape
    values = _fp4_rows_torch(x.reshape(-1, shape[-1]).float().contiguous())[2]
    return values.view(shape)


def select_candidate_blocks(logits, visible, block_size, topk_blocks):
    """V4.1 candidate max-pooling, including the newest visible block."""
    nblocks = (logits.shape[1] + block_size - 1) // block_size
    padded = F.pad(
        logits, (0, nblocks * block_size - logits.shape[1]), value=-torch.inf
    )
    scores = padded.view(logits.shape[0], nblocks, block_size).amax(-1)
    newest = ((visible - 1).clamp_min(0) // block_size).long()
    scores.scatter_(
        1, newest[:, None], torch.where(visible > 0, torch.inf, -torch.inf)[:, None]
    )
    values, ids = scores.topk(min(topk_blocks, nblocks), dim=-1)
    return torch.where(values > -torch.inf, ids, -1).int()


def mask_candidate_logits(logits, candidates, block_size):
    nblocks = (logits.shape[1] + block_size - 1) // block_size
    allowed = torch.zeros(
        logits.shape[0], nblocks + 1, dtype=torch.bool, device=logits.device
    )
    ids = torch.where(candidates >= 0, candidates, nblocks).long()
    allowed.scatter_(1, ids, True)
    columns = torch.arange(logits.shape[1], device=logits.device) // block_size
    return logits.masked_fill_(~allowed[:, columns], -torch.inf)


def _apply_prefill_candidates(shared, logits, visible, rows, block_size, topk, publish):
    """Publish candidates once and reuse their bounded bitmap across HCA sources.

    Single-request chunks use row slices, so output/bitmap views need no
    gather or scatter. Ragged requests retain their original row mapping.
    If the complete bitmap would exceed its budget, consumers rebuild only
    the current scoring chunk's bitmap from the same stored candidate IDs.
    """
    candidates = shared["candidates"]
    cache = shared.get("prefill_candidate_mask")
    flags = (
        cache[1][rows]
        if cache is not None and cache[0] is candidates and cache[2] == block_size
        else None
    )
    if publish:
        result = prefill_candidates.select_candidates(
            logits,
            visible,
            block_size,
            topk,
            out=candidates[rows],
            flags=flags,
            build_bitmap=not shared.get("prefill_sparse_candidates", False),
        )
        if result is None:
            block_ids = select_candidate_blocks(logits, visible, block_size, topk)
            candidates[rows, : block_ids.shape[1]] = block_ids
            # A partially initialized bitmap must never reach a consumer.
            shared.pop("prefill_candidate_mask", None)
        elif not isinstance(rows, slice):
            candidates[rows] = result[0]
            if flags is not None:
                cache[1][rows] = result[1]
    else:
        if flags is None:
            flags = prefill_candidates.build_flags(
                candidates[rows], logits.shape[1], block_size
            )
        if flags is None or not prefill_candidates.mask_candidates(
            logits, flags, block_size
        ):
            mask_candidate_logits(logits, candidates[rows], block_size)


def _prefill_sparse_plan(shared, key, candidates, visible, key_count, block_size):
    """Forward-local, byte-bounded reuse across the four HCA index sources.

    ``key`` identifies request, row range, logical K width and compression.
    The candidate tensor itself identifies the current L20 publication.
    Requests and their prefix positions are invariant within one forward.
    Plans beyond the aggregate limit are rebuilt for that chunk, not retained.
    """
    source = shared["candidates"]
    cache = shared.get("prefill_sparse_plans")
    if cache is None or cache[0] is not source:
        cache = [source, {}, 0]
        shared["prefill_sparse_plans"] = cache
    if key in cache[1]:
        return cache[1][key]
    plan = sparse_prefill_indexer.prepare_plan(
        candidates, visible, key_count, block_size
    )
    if plan is None:
        raise RuntimeError(
            "V4.1 sparse prefill plan rejected a supported scoring chunk"
        )
    limit = max(
        0, int(os.environ.get("DSV41_SPARSE_PREFILL_PLAN_MAX_BYTES", 256 * 1024**2))
    )
    if cache[2] + plan.nbytes <= limit:
        cache[1][key] = plan
        cache[2] += plan.nbytes
    return plan


class AttentionV41FP8(AttentionFP8):
    def __init__(self, *args, v41_config, shared_attention, **kwargs):
        ratio = int(kwargs["compress_ratio"])
        if ratio not in (0, 1, 2):
            raise ValueError(f"V4.1 compression ratio must be 0, 1 or 2, got {ratio}")
        # The inherited constructor owns projections and SWA, not the V4
        # compressor/indexer (their checkpoint topology differs from V4.1).
        kwargs["compress_ratio"] = 0
        super().__init__(*args, **kwargs)
        self.compress_ratio = ratio
        self.skip_post_q_norm = True
        self.v41_config = v41_config
        # Store a plain dict: registering source modules would create cycles.
        self._shared_attention = shared_attention
        shared_attention.setdefault("layers", {})[self.layer_id] = self
        self.index_topk = int(kwargs["index_topk"])
        self.index_n_heads = int(kwargs["index_n_heads"])
        self.index_head_dim = int(kwargs["index_head_dim"])
        self.is_kv_source = self.layer_id in v41_config.get("kv_source_layer_ids", [])
        self.is_index_source = self.layer_id in v41_config.get(
            "index_source_layer_ids", []
        )
        self.kv_source_layer_id = None
        self.index_source_layer_id = None
        if ratio:
            self.kv_source_layer_id = max(
                i for i in v41_config["kv_source_layer_ids"] if i <= self.layer_id
            )
            self.index_source_layer_id = max(
                i for i in v41_config["index_source_layer_ids"] if i <= self.layer_id
            )
            self._rope_base = kwargs["compress_rope_theta"]
            self._rope_o_seq_len = kwargs["original_seq_len"]
            self.freqs_cis = precompute_freqs_cis(
                self._rope_dim,
                self._rope_max_seq_len,
                self._rope_o_seq_len,
                self._rope_base,
                self._rope_factor,
                self._rope_beta_fast,
                self._rope_beta_slow,
            )
        from rtp_llm.utils.model_weight import W

        w = kwargs["layer_weights"]
        if self.is_kv_source:
            self.global_wkv = w[W.v4_compressor_wkv]
            self.global_norm = w[W.v4_compressor_norm]
            self.global_wgate = w[W.v4_compressor_wgate] if ratio == 2 else None
            self.index_wk = w[W.v41_indexer_wk]
            self.index_k_norm = w[W.v41_indexer_k_norm]
        if self.is_index_source:
            self.index_wq = _v4_fp8_linear(
                w[W.v4_indexer_wq_b_w], w[W.v4_indexer_wq_b_s]
            )
            self.index_weights = w[W.v4_indexer_weights_proj_w]
        self._pool_spec[CSA_STATE] = (torch.float32, 2 * self.head_dim)
        self._pool_spec[SWA_KV] = (torch.uint8, swa_codec.ENTRY_BYTES)
        # V4.1-Flash FP4 pools: GLOBAL regions 288B/entry, INDEX_K 68B/entry
        # (fixed layouts; see _v41_fp4_triton and DSV4CacheConfigHelper).
        self._pool_spec[CSA_KV] = (torch.uint8, FP4_GLOBAL_ENTRY_BYTES)
        self._pool_spec[HCA_KV] = (torch.uint8, FP4_GLOBAL_ENTRY_BYTES)
        self._pool_spec[INDEXER_KV] = (torch.uint8, FP4_INDEXER_ENTRY_BYTES)
        self._wo_a_groups_v41 = torch.nn.ModuleList()
        scale_block = self.wo_a_w.shape[1] // self.wo_a_s.shape[1]
        for group in range(self.n_groups):
            begin, end = group * self.o_lora_rank, (group + 1) * self.o_lora_rank
            self._wo_a_groups_v41.append(
                _v4_fp8_linear(
                    self.wo_a_w[begin:end].contiguous(),
                    self.wo_a_s[begin // scale_block : end // scale_block].contiguous(),
                )
            )

    def _project_output(self, o, freqs, out=None):
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_output_projection

        o = o.reshape(-1, self.n_heads, self.head_dim)
        if _v41_output_projection.is_supported(
            o, freqs, self._wo_a_stk_w, self._wo_a_stk_s
        ):
            projected = _v41_output_projection.grouped_output_projection(
                o, freqs, self._wo_a_stk_w, self._wo_a_stk_s
            )
            return self.wo_b(projected, out=out)
        apply_rotary_emb(
            o[..., -self.rope_head_dim :].unsqueeze(0), freqs, inverse=True
        )
        grouped = o.reshape(o.shape[0], self.n_groups, -1)
        projected = torch.stack(
            [
                linear(grouped[:, g].contiguous())
                for g, linear in enumerate(self._wo_a_groups_v41)
            ],
            dim=1,
        )
        return self.wo_b(projected.flatten(1), out=out)

    def _prefill_output_proj_into(self, o, freqs_cis, *, out):
        self._project_output(o, freqs_cis, out=out)

    def _ensure_freqs_cis_bound(self):
        # V4.1 has no independent compressor module or nested index compressor.
        return

    def _swa_entries_per_block(self):
        if self._swa_cp_byte_sliced():
            raw = self._pool_raw_u8(SWA_KV)
            if raw is not None:
                return raw.shape[1] * self._cp_ctx.cp_size // swa_codec.ENTRY_BYTES
        return self._pool_entries_per_block(SWA_KV)

    def _decode_write_swa_fp8(self, kv, bsz, q_len, attn_metadata):
        slots = attn_metadata.pool_write_slot_mappings.get(SWA_KV)
        pool = self._pool_view_3d_fp8(SWA_KV)
        if slots is not None and pool is not None:
            swa_codec.quantize_and_insert_swa_k_cache(
                kv.reshape(-1, self.head_dim), pool, slots[: bsz * q_len]
            )

    def _prefill_write_swa_fp8_paged(self, common, kv_full):
        meta = common.swa_meta
        if meta is None or meta.slot_mapping is None:
            return
        kv = kv_full.reshape(-1, self.head_dim).to(torch.bfloat16)
        if self._swa_cp_byte_sliced():
            raw = self._pool_raw_u8(SWA_KV)
            if raw is not None:
                swa_codec.quantize_and_insert_k_cache_cp_byte_sliced(
                    kv,
                    raw,
                    meta.slot_mapping,
                    full_entries_per_block=self._swa_entries_per_block(),
                    cp_rank=common.cp_ctx.cp_rank,
                    cp_size=common.cp_ctx.cp_size,
                    compaction=meta.slot_compaction,
                )
        else:
            pool = self._pool_view_3d_fp8(SWA_KV)
            if pool is not None:
                swa_codec.quantize_and_insert_swa_k_cache(kv, pool, meta.slot_mapping)

    def _build_shared_prefill_meta(self, *args, **kwargs):
        # Every V4.1 layer needs the full SWA prefix metadata, including global
        # layers. Build through the mature SWA planner with this layer's RoPE.
        ratio = self.compress_ratio
        self.compress_ratio = 0
        kwargs["reuse_common_meta"] = None
        try:
            return super()._build_shared_prefill_meta(*args, **kwargs)
        finally:
            self.compress_ratio = ratio

    def _materialize_prefill_q(self, qkv, common):
        if qkv.q is not None:
            return qkv
        rows = qkv.qr.shape[0]
        q_out = common.workspace.prefill_q(rows).view(
            rows, self.n_heads * self.head_dim
        )
        q = self._lin(self.wq_b, qkv.qr, out=q_out).view(
            -1, self.n_heads, self.head_dim
        )
        return qkv._replace(q=rope_only(q, common.freqs_cis, self.rope_head_dim))

    def _begin_forward(self):
        if self.layer_id == min(self._shared_attention["layers"]):
            self._shared_attention["global"] = {}
            self._shared_attention["topk"] = {}
            self._shared_attention["candidates"] = None
            self._shared_attention.pop("candidate_mask", None)
            self._shared_attention.pop("prefill_candidate_mask", None)
            self._shared_attention.pop("prefill_sparse_candidates", None)
            self._shared_attention.pop("prefill_sparse_plans", None)
            # ``_prefill_chunk_meta`` caches per-source-group chunk offsets for
            # the duration of one forward; drop them with the rest of the
            # per-forward shared state.
            self._shared_attention.pop("prefill_chunk_meta", None)
            # The layer-invariant sparse index plan (global/SWA gather indices
            # + per-row lengths) is cached per index-source group and dropped
            # with the rest of the per-forward shared state.
            self._shared_attention.pop("prefill_index_plan", None)

    def _owner(self):
        return self._shared_attention["layers"][self.kv_source_layer_id]

    def _global_region(self):
        return CSA_KV if self.compress_ratio == 2 else HCA_KV

    def _source_pool(self, region):
        if self._kv_cache is None:
            return None
        layer_id = self.layer_id if region == SWA_KV else self.kv_source_layer_id
        base = self._kv_cache.get_layer_cache(
            layer_id, _ATTN_TYPE_ENUM_BY_INT[region]
        ).kv_cache_base
        if base is None or base.numel() == 0:
            return None
        dtype, width = self._pool_spec[region]
        stride_bytes = base.shape[1] * base.element_size()
        eb = stride_bytes // (width * dtype.itemsize)
        raw = base.view(torch.uint8)
        if dtype == torch.uint8:
            return raw.as_strided((base.shape[0], eb, width), (stride_bytes, width, 1))
        return raw.view(dtype).view(-1, width)

    def _source_entries(self, region, pool):
        if region == CSA_STATE:
            raw = self._kv_cache.get_layer_cache(
                self.kv_source_layer_id, _ATTN_TYPE_ENUM_BY_INT[region]
            ).kv_cache_base
            return raw.shape[1] * raw.element_size() // (2 * self.head_dim * 4)
        return pool.shape[1]

    def _slots(self, region, positions, req_ids, *, state_end=None):
        pool = self._source_pool(region)
        if pool is None or self._block_tables_by_type is None:
            return None
        bt = self._block_tables_by_type[region]
        eb = self._source_entries(region, pool)
        tpb = require_pool_tokens_per_block(self._kv_cache, region=region)
        cp = self._cp_ctx
        sharded = cp is not None and cp.cp_size > 1 and cp.kv_cache_sharded
        if region == CSA_STATE:
            return cp_state_slot_mapping(
                positions,
                bt,
                req_ids,
                eb,
                tpb,
                cp.cp_size if sharded else 1,
                cp.cp_rank if sharded else 0,
                seq_end_per_req=state_end,
            )
        ratio = self.compress_ratio
        if sharded:
            return cp_kv_slot_mapping(
                positions,
                bt,
                req_ids,
                tpb,
                eb,
                ratio,
                cp.cp_size,
                cp.cp_rank,
                owner_tokens_per_block=self._kv_cache.seq_size_per_block,
            )
        col = positions // tpb
        valid = (col >= 0) & (col < bt.shape[1]) & ((positions + 1) % ratio == 0)
        block = bt[req_ids, col.clamp(0, bt.shape[1] - 1)].long()
        return torch.where(
            valid & (block > 0), block * eb + positions.remainder(tpb) // ratio, -1
        )

    def _gather_shards(self, value):
        if value.numel() == 0:
            return value
        cp = self._cp_ctx
        if cp is not None and cp.cp_size > 1 and cp.kv_cache_sharded:
            from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

            all_reduce(value, Group.TP, inplace=True)
        return value

    def _read_state(self, positions, req_ids):
        slots = self._slots(CSA_STATE, positions, req_ids)
        if slots is None:
            return torch.zeros(
                len(positions), 2 * self.head_dim, device=positions.device
            )
        pool = self._source_pool(CSA_STATE)
        state = pool[slots.clamp_min(0)].clone()
        state.masked_fill_((slots < 0).unsqueeze(-1), 0)
        return self._gather_shards(state)

    def _write_states(self, values, scores, positions, req_ids, seq_ends):
        slots = self._slots(CSA_STATE, positions, req_ids, state_end=seq_ends)
        if slots is not None:
            pool = self._source_pool(CSA_STATE)
            data = torch.cat((values, scores), -1).float()
            # Scatter without a boolean-mask compact (``pool[slots[valid]]``
            # forces a device->host ``nonzero`` sync to size the gather).
            # Invalid slots (-1) clamp to the sentinel slot 0 and are re-written
            # with their existing value, i.e. a no-op.
            idx = slots.clamp_min(0)
            pool[idx] = torch.where((slots >= 0)[:, None], data, pool[idx])

    def _produce_global(
        self,
        x_full,
        positions,
        req_ids,
        starts,
        lengths,
        *,
        prefill,
        projected_tiles=None,
    ):
        """Publish global pools in sequence order, carrying pairs across tiles."""
        owner = self._owner()
        ratio = self.compress_ratio
        from rtp_llm.models_py.modules.dsv4.fp8.compressor import _linear_bf16_bf16_fp32

        main_pool = self._source_pool(self._global_region())
        index_pool = self._source_pool(INDEXER_KV)
        rows = int(positions.shape[0])
        cp = getattr(self, "_cp_ctx", None)
        if cp is not None and cp.input_lengths_global_host is not None:
            lengths_host = cp.input_lengths_global_host
            prefixes_host = cp.prefix_lengths_host or (0,) * len(lengths_host)
        else:
            prefixes_host, lengths_host = torch.stack((starts, lengths)).tolist()
        ends = [s + l for s, l in zip(prefixes_host, lengths_host)]
        pair_ranges = []
        base = 0
        for start, length in zip(prefixes_host, lengths_host):
            pair_ranges.append((base + (1 - start % 2), base + length))
            base += length
        previous = None
        if ratio == 2:
            # Read the previous token before tail writes can reuse its ring slot.
            previous = self._read_state(
                (starts - 1).clamp_min(0),
                torch.arange(len(starts), device=x_full.device),
            )
        # Warmup (pool unbound) still needs the full keys for the shared
        # materialization, so it keeps every tile's keys; the pool path drops
        # them right after quantization.
        warm_keys = [] if main_pool is None else None
        carry = None
        tiles = projected_tiles
        if tiles is None:
            tiles = (
                (t0, x_full[t0 : t0 + _PRODUCE_GLOBAL_TILE_ROWS])
                for t0 in range(0, max(rows, 1), _PRODUCE_GLOBAL_TILE_ROWS)
            )
        for t0, x_tile in tiles:
            t1 = t0 + x_tile.shape[0]
            pos_tile = positions[t0:t1]
            req_tile = req_ids[t0:t1]
            values = (
                x_tile[:, : self.head_dim]
                if projected_tiles is not None
                else _linear_bf16_bf16_fp32(x_tile, owner.global_wkv)
            )
            if ratio == 2:
                scores = (
                    x_tile[:, self.head_dim :]
                    if projected_tiles is not None
                    else _linear_bf16_bf16_fp32(x_tile, owner.global_wgate)
                )
                # The seam row's previous value/score is the last row of the
                # previous tile; the first tile's head is replaced by the
                # per-request state read below, exactly as the single-shot
                # roll's head was.
                if carry is not None:
                    head_values, head_scores = carry
                else:
                    head_values, head_scores = values[:1], scores[:1]
                value_prev = torch.cat((head_values, values[:-1]), 0)
                score_prev = torch.cat((head_scores, scores[:-1]), 0)
                first = pos_tile == starts[req_tile]
                # Replace the boolean-mask index ``previous[req_ids[first]]`` (a
                # device->host ``nonzero`` sync) with an integer gather on the full
                # ``req_ids`` + a ``torch.where`` select. Same result, no sync.
                value_prev = torch.where(
                    first[:, None],
                    previous[req_tile, : self.head_dim].to(values.dtype),
                    value_prev,
                )
                score_prev = torch.where(
                    first[:, None],
                    previous[req_tile, self.head_dim :].to(scores.dtype),
                    score_prev,
                )
                # Host request metadata gives fixed-size pair boundaries without
                # a CUDA nonzero synchronization for every projection tile.
                pair_indices = []
                for first, end in pair_ranges:
                    begin = max(first, t0 + (first - t0) % 2)
                    if begin < min(end, t1):
                        pair_indices.append(
                            torch.arange(
                                begin - t0, min(end, t1) - t0, 2, device=x_full.device
                            )
                        )
                boundary_idx = (
                    torch.cat(pair_indices)
                    if pair_indices
                    else torch.empty(0, dtype=torch.long, device=x_full.device)
                )
                latent = compress_pairs(
                    torch.stack((value_prev[boundary_idx], values[boundary_idx]), 1),
                    torch.stack((score_prev[boundary_idx], scores[boundary_idx]), 1),
                    owner.global_norm,
                    self.eps,
                )
                self._write_states(values, scores, pos_tile, req_tile, starts + lengths)
                boundary_pos, boundary_req = (
                    pos_tile[boundary_idx],
                    req_tile[boundary_idx],
                )
                carry = (values[-1:].clone(), scores[-1:].clone())
            else:
                latent = rms_norm(values, owner.global_norm, self.eps).to(
                    torch.bfloat16
                )
                # ratio == 1: every position is a boundary — skip the all-True
                # boolean-mask compaction (``positions[torch.ones_like(...)]`` is a
                # pure device->host ``nonzero`` sync that selects everything).
                boundary_pos, boundary_req = pos_tile, req_tile
            freqs = self.freqs_cis[(boundary_pos // ratio) * ratio]
            global_keys = rope_only(latent.clone(), freqs, self.rope_head_dim)
            index_keys = rope_only(
                rms_norm(
                    F.linear(latent, owner.index_wk), owner.index_k_norm, self.eps
                ),
                freqs,
                self.rope_head_dim,
            )
            if main_pool is not None:
                from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_triton import (
                    quantize_and_insert_k_cache_fp4,
                    quantize_indexer_k_fp4,
                )

                quantize_and_insert_k_cache_fp4(
                    global_keys.contiguous(),
                    main_pool,
                    self._slots(self._global_region(), boundary_pos, boundary_req),
                )
                quantize_indexer_k_fp4(
                    index_keys.contiguous(),
                    self._slots(INDEXER_KV, boundary_pos, boundary_req),
                    index_pool,
                )
            else:
                warm_keys.append((boundary_pos, boundary_req, global_keys, index_keys))
        if warm_keys:
            boundary_pos = torch.cat([tile[0] for tile in warm_keys])
            boundary_req = torch.cat([tile[1] for tile in warm_keys])
            global_keys = torch.cat([tile[2] for tile in warm_keys])
            index_keys = torch.cat([tile[3] for tile in warm_keys])
        result = []
        fused_indexer = prefill and prefill_indexer.is_supported(
            x_full.device, getattr(self, "index_n_heads", 0), index_keys.shape[-1]
        )
        for b, end in enumerate(ends):
            count = int(end) // ratio
            idx = torch.arange(count, device=x_full.device, dtype=torch.long)
            pos = (idx + 1) * ratio - 1
            req = torch.full_like(pos, b)
            if main_pool is None:
                mask = boundary_req == b
                g = global_keys[mask]
                k = (
                    prefill_indexer.PrefillIndexerKeys(
                        *prefill_indexer.quantize_indexer_k_reference(index_keys[mask])
                    )
                    if fused_indexer
                    else fp8_roundtrip(index_keys[mask])
                )
            else:
                if fused_indexer:
                    k = prefill_indexer.gather_indexer_keys(
                        index_pool,
                        lambda p, r: self._slots(INDEXER_KV, p, r),
                        count,
                        b,
                        ratio,
                        self._cp_ctx,
                        self._kv_cache.seq_size_per_block,
                    )
                else:
                    from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_triton import (
                        dequantize_indexer_k_fp4,
                    )

                    k = dequantize_indexer_k_fp4(
                        index_pool, self._slots(INDEXER_KV, pos, req)
                    )
                    self._gather_shards(k)
                if prefill:
                    from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_triton import (
                        dequantize_k_cache_bytes_fp4,
                        gather_k_cache_bytes_fp4,
                    )

                    # Byte-first transport: all-reduce the raw 288B FP4 pool
                    # bytes (each byte has exactly one owner, so the SUM
                    # reassembles them bit-exactly) and dequantize on the
                    # receiving side — 288B per entry on the wire instead of
                    # the 1024B dequantized rows.
                    raw_global = gather_k_cache_bytes_fp4(
                        main_pool, self._slots(self._global_region(), pos, req)
                    )
                    self._gather_shards(raw_global)
                    g = dequantize_k_cache_bytes_fp4(raw_global)
                else:
                    g = None
            result.append((g, k))
        self._shared_attention["global"] = {self.layer_id: result}
        # ``_prefill_chunk_meta`` caches per-request chunk offsets derived from
        # these globals; republishing them invalidates that cache.
        self._shared_attention.pop("prefill_chunk_meta", None)

    def _select_indices(self, x, qr, positions, req_ids):
        shared = self._shared_attention
        if not self.is_index_source:
            return shared["topk"][self.index_source_layer_id]
        globals_by_req = shared["global"][self.kv_source_layer_id]
        out = torch.full(
            (x.shape[0], self.index_topk), -1, dtype=torch.int32, device=x.device
        )
        config = getattr(self, "v41_config", {})
        candidate_source = int(config.get("candidate_source_layer_id", -1))
        candidate_blocks = int(config.get("candidate_topk_blocks", 0))
        candidate_size = int(config.get("candidate_block_size", 0))
        publish_candidates = (
            self.layer_id == candidate_source
            and candidate_blocks > 0
            and candidate_size > 0
        )
        if publish_candidates:
            shared.pop("prefill_candidate_mask", None)
            shared.pop("prefill_sparse_plans", None)
            shared["prefill_sparse_candidates"] = (
                os.environ.get("DSV41_SPARSE_PREFILL_INDEXER", "1") != "0"
                and self.index_topk == 512
                and self.index_n_heads == 32
                and candidate_size == 8
                and prefill_deepselect.is_available(x.device)
            )
            max_blocks = max(
                (
                    (len(k) + candidate_size - 1) // candidate_size
                    for _, k in globals_by_req
                ),
                default=0,
            )
            shared["candidates"] = (
                torch.full(
                    (x.shape[0], candidate_blocks),
                    -1,
                    dtype=torch.int32,
                    device=x.device,
                )
                if max_blocks > candidate_blocks
                else None
            )
            if (
                shared["candidates"] is not None
                and not shared["prefill_sparse_candidates"]
                and x.is_cuda
                and os.environ.get("DSV41_FUSED_PREFILL_CANDIDATES", "1") != "0"
                and prefill_candidates.bitmap_is_bounded(
                    x.shape[0], max_blocks * candidate_size, candidate_size
                )
            ):
                shared["prefill_candidate_mask"] = (
                    shared["candidates"],
                    torch.empty(
                        (
                            x.shape[0],
                            prefill_candidates.bitmap_words(
                                max_blocks * candidate_size, candidate_size
                            ),
                        ),
                        dtype=torch.int32,
                        device=x.device,
                    ),
                    candidate_size,
                )
        # Short prompts select all causal candidates, as in the vLLM backend.
        if max((len(k) for _, k in globals_by_req), default=0) <= self.index_topk:
            ids = torch.arange(self.index_topk, device=x.device)
            out.copy_(
                torch.where(
                    ids[None] < ((positions + 1) // self.compress_ratio)[:, None],
                    ids[None],
                    -1,
                )
            )
        else:
            q = self._lin(self.index_wq, qr).view(
                -1, self.index_n_heads, self.index_head_dim
            )
            q = rope_only(q, self.freqs_cis[positions], self.rope_head_dim)
            weights = (
                F.linear(x, self.index_weights).float()
                * (self.index_head_dim * self.index_n_heads) ** -0.5
            )
            fused_indexer = isinstance(
                globals_by_req[0][1], prefill_indexer.PrefillIndexerKeys
            )
            if fused_indexer:
                q_fp4, q_sf = prefill_indexer.quantize_indexer_q(q)
            else:
                q = fp8_roundtrip(q)
            single_request = len(globals_by_req) == 1
            for b, (_, keys) in enumerate(globals_by_req):
                # One-argument ``torch.where`` is ``nonzero``: it syncs to size
                # its output. With a single request every token belongs to
                # request 0, so the row selection is just the full range.
                rows = (
                    torch.arange(positions.shape[0], device=positions.device)
                    if single_request
                    else torch.where(req_ids == b)[0]
                )
                key_count = len(keys)
                candidates = shared.get("candidates")
                probe = slice(0, 1) if single_request else rows[:1]
                sparse_request = (
                    fused_indexer
                    and candidates is not None
                    and candidate_source >= 0
                    and self.layer_id > candidate_source
                    and self.index_topk == 512
                    and prefill_deepselect.is_available(x.device)
                    and sparse_prefill_indexer.is_supported(
                        q_fp4[probe],
                        q_sf[probe],
                        keys,
                        weights[probe],
                        candidates[probe],
                        positions[probe],
                        candidate_size,
                        self.index_topk,
                    )
                )
                chunk_rows = (
                    sparse_prefill_indexer.MAX_CHUNK_ROWS
                    if sparse_request
                    else (
                        prefill_indexer.logits_chunk_rows(key_count)
                        if fused_indexer
                        else 64
                    )
                )
                for chunk_index, chunk in enumerate(rows.split(chunk_rows)):
                    output_rows = (
                        slice(chunk_index * chunk_rows, (chunk_index + 1) * chunk_rows)
                        if single_request
                        else chunk
                    )
                    visible = (positions[chunk] + 1) // self.compress_ratio
                    if sparse_request:
                        with record_function_range(
                            "dsv41.prefill.indexer.sparse_logits"
                        ):
                            plan = _prefill_sparse_plan(
                                shared,
                                (
                                    b,
                                    chunk_index,
                                    len(chunk),
                                    key_count,
                                    self.compress_ratio,
                                    candidate_size,
                                ),
                                candidates[output_rows],
                                visible,
                                key_count,
                                candidate_size,
                            )
                            logits = sparse_prefill_indexer.score(
                                q_fp4[output_rows],
                                q_sf[output_rows],
                                keys,
                                weights[output_rows],
                                plan,
                            )
                            if logits is None:
                                raise RuntimeError(
                                    "V4.1 sparse prefill score layout changed after dispatch"
                                )
                        with record_function_range("dsv41.prefill.indexer.deepselect"):
                            selected = prefill_deepselect.try_select_sparse_tokens(
                                logits, plan.end
                            )
                            if selected is None:
                                raise RuntimeError(
                                    "V4.1 sparse prefill DeepSelect layout is unsupported"
                                )
                            mapped = sparse_prefill_indexer.remap(
                                selected,
                                plan,
                                logits=logits,
                                out=out[output_rows] if single_request else None,
                            )
                            if mapped is None:
                                raise RuntimeError(
                                    "V4.1 sparse prefill remap layout is unsupported"
                                )
                            if not single_request:
                                out[output_rows] = mapped
                        continue
                    if fused_indexer:
                        with record_function_range(
                            "dsv41.prefill.indexer.fused_logits"
                        ):
                            logits = prefill_indexer.score_indexer_chunk(
                                q_fp4[chunk],
                                q_sf[chunk],
                                keys.quant,
                                keys.scale,
                                weights[chunk],
                                visible,
                            )
                    else:
                        logits = torch.einsum(
                            "thd,kd->thk", q[chunk], keys.float()
                        ).relu_()
                        logits = (logits * weights[chunk, :, None]).sum(1)
                        ids = torch.arange(key_count, device=x.device)
                        logits.masked_fill_(ids[None] >= visible[:, None], -torch.inf)
                    candidates = shared.get("candidates")
                    if candidates is not None:
                        if publish_candidates or (
                            candidate_source >= 0 and self.layer_id > candidate_source
                        ):
                            _apply_prefill_candidates(
                                shared,
                                logits,
                                visible,
                                output_rows,
                                candidate_size,
                                candidate_blocks,
                                publish_candidates,
                            )
                    k = min(self.index_topk, key_count)
                    selected = prefill_topk.try_select_tokens(logits, visible, k)
                    if selected is None:
                        scores, selected = logits.topk(k, dim=-1)
                        selected = torch.where(scores.isfinite(), selected, -1).int()
                    out[output_rows, :k] = selected
        shared["topk"] = {self.layer_id: out}
        return out

    def _host_prefill_lengths(self, common) -> list:
        """Per-request new-token counts as host ints, without a GPU->CPU sync.

        Under CP the authoritative host copy already lives on
        ``CPContext.input_lengths_global_host`` (derived from the framework's
        CPU ``prefill_actual_input_lengths_cpu``); reading it avoids the
        stream-synchronizing ``.tolist()`` on the GPU tensor. Fall back to
        ``.tolist()`` only when that host copy is unavailable (non-CP warmup).
        """
        cp = common.cp_ctx
        if cp is not None and cp.input_lengths_global_host is not None:
            return list(cp.input_lengths_global_host)
        t = common.cp_ctx.input_lengths_global if common.cp_on else common.input_lengths
        return t.detach().tolist()

    def _host_prefill_prefixes(self, common) -> list:
        """Per-request KV prefix lengths as host ints (no GPU->CPU sync)."""
        cp = common.cp_ctx
        if cp is not None and cp.prefix_lengths_host is not None:
            return list(cp.prefix_lengths_host)
        return common.prefix_lengths.detach().tolist()

    def _swa_prefill_workspace(self, qkv, common):
        """Return BF16 [request, prefix tail + new tokens] for sparse prefill."""
        lengths_host = self._host_prefill_lengths(common)
        if not common.any_cont:
            return list(qkv.kv_full.split(lengths_host)), [0] * len(lengths_host)
        buf = self._swa_prefill_concat(qkv, common)
        prefixes_host = self._host_prefill_prefixes(common)
        tails = [min(int(p), self.window_size - 1) for p in prefixes_host]
        return [
            buf[b, : tails[b] + int(lengths_host[b])] for b in range(common.batch_size)
        ], [int(p) - t for p, t in zip(prefixes_host, tails)]

    def _swa_prefill_concat(self, qkv, common):
        """Read the 528B prefix before a new chunk overwrites the SWA ring."""
        meta = common.swa_meta
        B, D = common.batch_size, self.head_dim
        buf = torch.zeros(B, meta.M, D, dtype=torch.bfloat16, device=qkv.kv_full.device)
        if meta.prefix_len_max > 0:
            if self._swa_cp_byte_sliced():
                swa_codec.dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                    out=buf,
                    k_cache_raw=self._pool_raw_u8(SWA_KV),
                    slot_mapping=meta.cache_slot_mapping,
                    gather_lens=meta.cache_gather_lens,
                    offset=0,
                    full_entries_per_block=self._swa_entries_per_block(),
                    cp_size=common.cp_ctx.cp_size,
                    cp_rank=common.cp_ctx.cp_rank,
                    compaction=meta.cache_compaction,
                )
            else:
                swa_codec.dequantize_and_gather_k_cache_slots(
                    out=buf,
                    k_cache=self._pool_view_3d_fp8(SWA_KV),
                    slot_mapping=meta.cache_slot_mapping,
                    gather_lens=meta.cache_gather_lens,
                    offset=0,
                )
        buf.view(-1, D).index_copy_(0, meta.slot_in_flat, qkv.kv_full)
        return buf

    def _prefill_chunk_meta(
        self, globals_by_req, swa, swa_starts, req_ids, device, *, common
    ):
        """Per-request chunk offsets for the sparse-prefill index build.

        ``offsets``/``ns``/``swstart`` are pure functions of the KV-source
        globals' shapes and of the SWA window shape, and both are fixed for a
        whole ``kv_source_layer_id`` group. Build batched metadata from the
        existing device lengths/prefixes, avoiding pageable host uploads even
        on the first layer of a group. ``_produce_global`` invalidates this
        cache whenever it republishes ``shared["global"]``.
        """
        shared = self._shared_attention
        cached = shared.get("prefill_chunk_meta")
        if cached is not None:
            return cached
        offsets, global_sizes = [], []
        offset = 0
        for (g, _), sw in zip(globals_by_req, swa):
            offsets.append(offset)
            global_sizes.append(g.shape[0])
            offset += g.shape[0] + sw.shape[0]
        if len(offsets) == 1:
            # Single-request prefill: ``req_ids`` is all zeros, so indexing a
            # one-element tensor by it is a pure broadcast. Use the host scalars
            # directly and skip the device tensors (and their H2D copy) entirely.
            meta = (offsets[0], global_sizes[0], swa_starts[0])
        else:
            if self._source_pool(self._global_region()) is None:
                # Pool-free warmup materializes only this chunk's closed
                # groups, so its shapes need not include prefix history.
                def device_values(values):
                    return torch.cat(
                        [
                            torch.full((1,), v, dtype=torch.long, device=device)
                            for v in values
                        ]
                    )

                offsets_d = device_values(offsets)
                sizes = device_values(global_sizes)
                starts_d = device_values(swa_starts)
            else:
                lengths = (
                    common.cp_ctx.input_lengths_global
                    if common.cp_on
                    else common.input_lengths
                ).long()
                prefixes = common.prefix_lengths.long()
                tails = prefixes.clamp(max=self.window_size - 1)
                sizes = (prefixes + lengths) // self.compress_ratio
                chunk_sizes = sizes + lengths + tails
                offsets_d = chunk_sizes.cumsum(0) - chunk_sizes
                starts_d = prefixes - tails
            meta = (
                offsets_d[req_ids, None],
                sizes[req_ids, None],
                starts_d[req_ids, None],
            )
        shared["prefill_chunk_meta"] = meta
        return meta

    def _forward_prefill(self, x, positions, shared_input_quant=None):
        self._begin_forward()
        common = self._prefill_common_setup(x, positions)
        qkv = self._prefill_compute_qkv(
            x, common, shared_input_quant=shared_input_quant
        )
        # Launch the first bounded hidden-state transfer after the QKV gather.
        # It overlaps SWA work; subsequent transfers overlap global projection.
        x_gather = None
        small_cp = common.cp_on and _use_small_cp_x_gather(common.cp_ctx)
        if self.is_kv_source and common.cp_on:
            if small_cp:
                # One bounded all-gather avoids per-half launches on short requests.
                first_tile = (None, 0, common.cp_ctx.padded_seq_len)
                weights = ()
                x_buffers = [x.new_empty((first_tile[2], x.shape[-1]))]
            else:
                owner = self._owner()
                weights = (
                    (owner.global_wkv, owner.global_wgate)
                    if self.compress_ratio == 2
                    else (owner.global_wkv,)
                )
                x_plan = iter(_prefill_x_tile_plan(common.cp_ctx))
                first_tile = next(x_plan)
                # Replicated projection weights let only the owner compute each
                # row. Transfer FP32 values/scores, preserving their pool boundary.
                x_buffers = [
                    torch.empty(
                        (_PRODUCE_GLOBAL_TILE_ROWS, len(weights) * self.head_dim),
                        device=x.device,
                        dtype=torch.float32,
                    )
                    for _ in range(2)
                ]
            with record_function_range("dsv41.prefill.x_gather.async_start"):
                x_gather = _start_prefill_x_gather_async(
                    x, common.cp_ctx, first_tile, x_buffers[0], weights
                )
        # Read old SWA tails before writes wrap over them in long prefill chunks.
        swa, swa_starts = (
            self._swa_prefill_workspace(qkv, common)
            if self.compress_ratio
            else (None, None)
        )
        swa_only_workspace = (
            self._swa_prefill_concat(qkv, common)
            if not self.compress_ratio
            and common.any_cont
            and self._kv_cache is not None
            else None
        )
        self._prefill_write_swa_fp8_paged(common, qkv.kv_full)
        if not self.compress_ratio:
            if swa_only_workspace is not None:
                qkv = self._materialize_prefill_q(qkv, common)
                dispose_tensor(qkv.kv_full)
                meta = common.swa_meta
                return self._flash_mla_sparse_fwd_chunked_projected(
                    q=qkv.q,
                    kv=swa_only_workspace.view(-1, 1, self.head_dim),
                    indices=meta.combined_indices.unsqueeze(1),
                    topk_length=meta.combined_lens,
                    freqs_cis=common.freqs_cis,
                    profile_name="dsv41.prefill.swa_concat.flash_mla",
                )
            return self._forward_prefill_swa_only(qkv, common)
        positions = (
            common.cp_ctx.global_positions.long()
            if common.cp_on
            else common.position_ids.long()
        )
        req_ids = common.req_id_per_token.long()
        starts = common.prefix_lengths.long()
        lengths = (
            common.cp_ctx.input_lengths_global if common.cp_on else common.input_lengths
        ).long()
        if self.is_kv_source:
            rows = common.cp_ctx.seq_len_full if common.cp_on else x.shape[0]
            x_full, projected_tiles = x, None
            if small_cp:
                x_full = _cp_restore_gathered_full_2d(
                    _wait_prefill_x_gather(x_gather), common.cp_ctx
                )
            elif common.cp_on:
                projected_tiles = _prefill_x_tiles(
                    x, common.cp_ctx, x_plan, x_gather, x_buffers, weights
                )
            x_gather = None
            if common.cp_on:
                del x_buffers
            ids_full = torch.repeat_interleave(
                torch.arange(common.batch_size, device=x.device),
                lengths,
                output_size=rows,
            )
            cu = torch.cat(
                (torch.zeros(1, device=x.device, dtype=torch.long), lengths.cumsum(0))
            )
            pos_full = (
                torch.arange(rows, device=x.device) - cu[ids_full] + starts[ids_full]
            )
            self._produce_global(
                x_full,
                pos_full,
                ids_full,
                starts,
                lengths,
                prefill=True,
                projected_tiles=projected_tiles,
            )
            del x_full
        selected = self._select_indices(x, qkv.qr, positions, req_ids)
        globals_by_req = self._shared_attention["global"][self.kv_source_layer_id]
        offsets, ns, swstart = self._prefill_chunk_meta(
            globals_by_req, swa, swa_starts, req_ids, x.device, common=common
        )
        chunks = []
        for (g, _), sw in zip(globals_by_req, swa):
            chunks.extend((g, sw))
        # Cross-layer index plan cache: within one index-source group every
        # layer consumes the same ``selected`` tensor against the same
        # per-forward positions / chunk offsets / SWA starts, so the sparse
        # gather indices and row lengths are bit-identical across the group.
        # Build once per group (keyed by the selected tensor's identity) and
        # reuse — the argsort/cat/pad chain drops from once per layer to once
        # per group. Dropped with the rest of the per-forward shared state in
        # ``_begin_forward``.
        plan = self._shared_attention.get("prefill_index_plan")
        if plan is not None and plan[0] is selected:
            indices, lens = plan[1], plan[2]
        else:
            swpos = (
                positions[:, None]
                - self.window_size
                + 1
                + torch.arange(self.window_size, device=x.device)[None]
            )
            swidx = torch.where(swpos >= swstart, offsets + ns + swpos - swstart, -1)
            global_idx = torch.where(selected >= 0, offsets + selected, -1)
            indices = torch.cat((global_idx, swidx), -1).int()
            # FlashMLA's length bounds count a compact valid prefix.
            indices = indices.gather(1, torch.argsort(indices < 0, dim=-1, stable=True))
            lens = (indices >= 0).sum(-1).int()
            if indices.shape[1] % 64:
                indices = F.pad(indices, (0, 64 - indices.shape[1] % 64), value=-1)
            self._shared_attention["prefill_index_plan"] = (selected, indices, lens)
        qkv = self._materialize_prefill_q(qkv, common)
        return self._flash_mla_sparse_fwd_chunked_projected(
            q=qkv.q,
            kv=torch.cat(chunks).unsqueeze(1),
            indices=indices.unsqueeze(1),
            topk_length=lens,
            freqs_cis=common.freqs_cis,
            profile_name="dsv41.prefill.shared_global",
        )

    def _produce_global_decode(self, x, positions, req_ids, starts):
        """Fixed-shape global writes and index reads, safe for CUDA graphs."""
        from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_triton import (
            dequantize_indexer_k_fp4,
            quantize_and_insert_k_cache_fp4,
            quantize_indexer_k_fp4,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.compressor import _linear_bf16_bf16_fp32

        B, S, _ = x.shape
        from rtp_llm.models_py.modules.dsv4.fp8._v41_decode_global import (
            try_produce_global,
        )

        if not try_produce_global(self, x, positions, req_ids, starts):
            flat = x.reshape(B * S, -1)
            values = _linear_bf16_bf16_fp32(flat, self.global_wkv)
            if self.compress_ratio == 2:
                scores = _linear_bf16_bf16_fp32(flat, self.global_wgate)
                previous = self._read_state(
                    (starts - 1).clamp_min(0), torch.arange(B, device=x.device)
                )
                first = (torch.arange(B * S, device=x.device) % S == 0)[:, None]
                value_prev = torch.where(
                    first, previous[req_ids, : self.head_dim], values.roll(1, 0)
                )
                score_prev = torch.where(
                    first, previous[req_ids, self.head_dim :], scores.roll(1, 0)
                )
                latent = compress_pairs(
                    torch.stack((value_prev, values), 1),
                    torch.stack((score_prev, scores), 1),
                    self.global_norm,
                    self.eps,
                )
                slots = self._slots(CSA_STATE, positions, req_ids, state_end=starts + S)
                state_rows = torch.cat((values, scores), -1)
                # Slot zero is the allocator's unallocated sentinel; invalid rows
                # may overwrite it, but no live request ever reads that slot.
                self._source_pool(CSA_STATE).index_copy_(
                    0,
                    slots.clamp_min(0),
                    torch.where((slots >= 0)[:, None], state_rows, 0.0),
                )
            else:
                latent = rms_norm(values, self.global_norm, self.eps).to(torch.bfloat16)
            freqs = self.freqs_cis[
                (positions // self.compress_ratio) * self.compress_ratio
            ]
            global_keys = rope_only(latent.clone(), freqs, self.rope_head_dim)
            index_keys = rope_only(
                rms_norm(F.linear(latent, self.index_wk), self.index_k_norm, self.eps),
                freqs,
                self.rope_head_dim,
            )
            quantize_and_insert_k_cache_fp4(
                global_keys.contiguous(),
                self._source_pool(self._global_region()),
                self._slots(self._global_region(), positions, req_ids),
            )
            index_pool = self._source_pool(INDEXER_KV)
            quantize_indexer_k_fp4(
                index_keys.contiguous(),
                self._slots(INDEXER_KV, positions, req_ids),
                index_pool,
            )
        index_pool = self._source_pool(INDEXER_KV)
        capacity = self._rope_max_seq_len // self.compress_ratio
        from rtp_llm.models_py.modules.dsv4.fp8 import (
            _v41_decode_indexer as decode_indexer,
        )

        cp = self._cp_ctx
        sharded = cp is not None and cp.cp_size > 1 and cp.kv_cache_sharded
        table = self._block_tables_by_type[INDEXER_KV][:B]
        raw_tpb = require_pool_tokens_per_block(self._kv_cache, region=INDEXER_KV)
        logical_entries = raw_tpb // self.compress_ratio
        if (
            not sharded
            and raw_tpb % self.compress_ratio == 0
            and index_pool.is_contiguous()
            and logical_entries in (index_pool.shape[1], index_pool.shape[1] // 2)
            and capacity <= table.shape[1] * logical_entries
            and decode_indexer.is_supported(
                x.device, index_pool.shape[1], self.index_n_heads, self.index_head_dim
            )
        ):
            # Keep the owner's original FP8 bytes and paged layout. In V4.1's
            # uniform INDEXER_KV pool, ratio 2 only fills half of each page.
            self._shared_attention["global"] = {
                self.layer_id: decode_indexer.DecodeIndexerKeys(
                    index_pool, table, capacity, logical_entries
                )
            }
            return
        ids = torch.arange(capacity, device=x.device, dtype=torch.long).expand(B, -1)
        read_positions = (ids + 1) * self.compress_ratio - 1
        read_req = torch.arange(B, device=x.device)[:, None].expand_as(ids)
        slots = self._slots(
            INDEXER_KV, read_positions.reshape(-1), read_req.reshape(-1)
        ).view(B, capacity)
        slots = torch.where(
            ids < ((starts + S) // self.compress_ratio)[:, None], slots, -1
        )
        keys = dequantize_indexer_k_fp4(index_pool, slots.reshape(-1)).view(
            B, capacity, self.index_head_dim
        )
        self._shared_attention["global"] = {self.layer_id: keys}

    def _select_indices_decode(self, x, qr, positions):
        """Capture-static index scoring with device-side causal/candidate masks."""
        shared = self._shared_attention
        if not self.is_index_source:
            return shared["topk"][self.index_source_layer_id]
        B, S, _ = x.shape
        T = B * S
        keys = shared["global"][self.kv_source_layer_id]
        from rtp_llm.models_py.modules.dsv4.fp8 import (
            _v41_decode_indexer as decode_indexer,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.indexer import _run_topk_v3

        paged = isinstance(keys, decode_indexer.DecodeIndexerKeys)
        capacity = keys.capacity if paged else keys.shape[1]
        flat = x.reshape(T, -1)
        q = self._lin(self.index_wq, qr.reshape(T, -1)).view(
            T, self.index_n_heads, self.index_head_dim
        )
        weights = F.linear(flat, self.index_weights)
        visible_per_token = (positions + 1) // self.compress_ratio
        if paged:
            all_logits = decode_indexer.score_decode_indexer(
                q.view(B, S, self.index_n_heads, self.index_head_dim),
                weights.view(B, S, self.index_n_heads),
                self.freqs_cis,
                keys.pool,
                keys.block_table,
                visible_per_token.view(B, S),
                max_ctx_len=capacity,
                rope_head_dim=self.rope_head_dim,
                logical_entries_per_block=keys.logical_entries_per_block,
                positions=positions.reshape(T),
            )
            if all_logits is None:
                raise RuntimeError(
                    "V4.1 paged indexer support changed within a forward"
                )
        else:
            weights = (
                weights.float() * (self.index_head_dim * self.index_n_heads) ** -0.5
            )
            freqs = self.freqs_cis[positions]
            q = fp4_roundtrip(rope_only(q, freqs, self.rope_head_dim))
        config = self.v41_config
        candidate_source = int(config.get("candidate_source_layer_id", -1))
        candidate_blocks = int(config.get("candidate_topk_blocks", 0))
        candidate_size = int(config.get("candidate_block_size", 0))
        publish = (
            self.layer_id == candidate_source
            and candidate_blocks > 0
            and candidate_size > 0
        )
        use_candidates = (
            candidate_size > 0
            and (capacity + candidate_size - 1) // candidate_size > candidate_blocks
        )
        if publish:
            shared.pop("candidate_mask", None)
            shared["candidates"] = (
                torch.empty(T, candidate_blocks, dtype=torch.int32, device=x.device)
                if use_candidates
                else None
            )
        visible_i32 = visible_per_token.to(torch.int32)
        if paged:
            from rtp_llm.models_py.modules.dsv4.fp8 import (
                _v41_decode_topk as decode_topk,
            )

            candidates = shared.get("candidates")
            cached_mask = shared.get("candidate_mask")
            compatible_mask = (
                candidate_size > 0
                and cached_mask is not None
                and cached_mask[0] is candidates
                and cached_mask[1].shape
                == (T, (capacity + candidate_size - 1) // candidate_size)
                and cached_mask[2] == candidate_size
            )
            if decode_topk.is_supported(all_logits, visible_i32, self.index_topk) and (
                candidates is None or publish or compatible_mask
            ):
                # DeepGEMM already returned rows for all B*S tokens. Keep the
                # selection batched instead of launching once per request.
                if candidates is not None:
                    if publish:
                        candidates, flags = decode_topk.select_candidates(
                            all_logits, visible_i32, candidate_size, candidate_blocks
                        )
                        shared["candidates"] = candidates
                        shared["candidate_mask"] = (candidates, flags, candidate_size)
                    elif candidate_source >= 0 and self.layer_id > candidate_source:
                        decode_topk.mask_candidates(
                            all_logits, cached_mask[1], candidate_size
                        )
                output = decode_topk.select_tokens(
                    all_logits, visible_i32, self.index_topk
                )
                shared["topk"] = {self.layer_id: output}
                return output
        output = torch.full(
            (T, self.index_topk), -1, dtype=torch.int32, device=x.device
        )
        columns = None if paged else torch.arange(capacity, device=x.device)
        for b in range(B):
            for start in range(b * S, (b + 1) * S, 16):
                end = min(start + 16, (b + 1) * S)
                if paged:
                    logits = all_logits[start:end]
                else:
                    logits = torch.einsum("thd,kd->thk", q[start:end], keys[b]).relu_()
                    logits = (logits * weights[start:end, :, None]).sum(1)
                visible = visible_per_token[start:end]
                if not paged:
                    logits.masked_fill_(columns[None] >= visible[:, None], -torch.inf)
                candidates = shared.get("candidates")
                if candidates is not None:
                    if publish:
                        candidates[start:end] = select_candidate_blocks(
                            logits, visible, candidate_size, candidate_blocks
                        )
                    elif candidate_source >= 0 and self.layer_id > candidate_source:
                        mask_candidate_logits(
                            logits, candidates[start:end], candidate_size
                        )
                count = min(self.index_topk, capacity)
                # Radix-select TopK (the shared V4 indexer kernel) serves the
                # production paged decode shape; the per-row lengths subsume
                # the -inf/isfinite masking. Output order is unspecified (the
                # op contract); the branch below pins short rows ascending.
                if not (
                    paged
                    and logits.is_cuda
                    and count == self.index_topk
                    and _run_topk_v3(
                        logits,
                        visible_i32[start:end],
                        output[start:end],
                        count,
                        capacity,
                    )
                ):
                    scores, selected = logits.topk(count, dim=-1)
                    selected = torch.where(scores.isfinite(), selected, -1)
                    output[start:end, :count] = selected.int()
                # Preserve the short-context ascending-order shortcut without
                # a CPU branch; it also avoids unstable ties between zero keys.
                dense = torch.arange(count, device=x.device)[None].expand(
                    end - start, -1
                )
                dense = torch.where(dense < visible[:, None], dense, -1)
                output[start:end, :count] = torch.where(
                    (visible <= self.index_topk)[:, None],
                    dense,
                    output[start:end, :count],
                )
        shared["topk"] = {self.layer_id: output}
        return output

    def _decode_global_slots(self, selected, req):
        """Translate compressed indices without materializing token positions."""
        region = self._global_region()
        pool = self._source_pool(region)
        table = self._block_tables_by_type[region]
        raw_tpb = require_pool_tokens_per_block(self._kv_cache, region=region)
        logical_entries = raw_tpb // self.compress_ratio
        cp = self._cp_ctx
        sharded = cp is not None and cp.cp_size > 1 and cp.kv_cache_sharded
        if (
            os.environ.get("DSV41_FUSED_DECODE_SLOTS", "1") != "0"
            and selected.is_cuda
            and selected.dtype == torch.int32
            and req.dtype == torch.int32
            and table.dtype == torch.int32
            and selected.shape[1] > 0
            and not sharded
            and raw_tpb % self.compress_ratio == 0
            and pool.shape[1] == logical_entries
        ):
            from rtp_llm.models_py.modules.dsv4.fp8.decode.paged_topk_translator import (
                translate_local_to_global_slots,
            )

            return translate_local_to_global_slots(
                req,
                table,
                selected,
                entries_per_block=pool.shape[1],
                tokens_per_block_for_block_table=logical_entries,
            )
        slots = self._slots(
            region,
            (selected.long().clamp_min(0) + 1).reshape(-1) * self.compress_ratio - 1,
            req[:, None].expand_as(selected).reshape(-1),
        ).reshape_as(selected)
        slots.masked_fill_(selected < 0, -1)
        return slots.int()

    def _forward_decode_body(self, x, metadata):
        from rtp_llm.models_py.modules.dsv4.fp8.decode.compute_qkv import (
            decode_compute_qkv,
        )

        self._begin_forward()
        B, S, _ = x.shape
        T = B * S
        positions = metadata.position_ids[:T].long()
        qkv = decode_compute_qkv(self, x, positions)
        self._decode_write_swa_fp8(qkv.kv, B, S, metadata)
        if not self.compress_ratio:
            o = self._forward_decode_swa_only(qkv.q, B, S, metadata)
        else:
            if self.is_kv_source:
                req = metadata.req_id_per_token[:T].long()
                starts = metadata.start_pos[:B].long()
                self._produce_global_decode(x, positions, req, starts)
            selected = self._select_indices_decode(x, qkv.qr, positions)
            from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_decode_attn import (
                fp4_dual_decode_attention,
            )
            from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
                get_or_build_sched_meta,
            )

            slots = self._decode_global_slots(selected, metadata.req_id_per_token[:T])
            o = fp4_dual_decode_attention(
                q=qkv.q,
                swa_pool_3d=self._pool_view_3d_fp8(SWA_KV),
                global_pool_3d=self._source_pool(self._global_region()),
                attn_sink=self.attn_sink,
                swa_topk_3d=metadata.swa_global_slots[:T]
                .view(B, S, self.window_size)
                .contiguous(),
                global_topk_3d=slots.int().view(B, S, self.index_topk).contiguous(),
                swa_block_table=metadata.pool_block_tables[SWA_KV][:B],
                sched_meta=get_or_build_sched_meta(
                    metadata,
                    batch_size=B,
                    q_len=S,
                    num_heads=self.n_heads,
                    topk=self.window_size,
                    extra_attn_type=self._global_region(),
                ),
                fp8_op=self._get_fp8_decode_op(),
            )
        out = self._project_output(o, qkv.freqs_cis)
        self._prefill_output_all_reduce(out)
        return out.view(B, S, self.dim)
