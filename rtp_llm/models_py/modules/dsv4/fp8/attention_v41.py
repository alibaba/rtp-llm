"""DeepSeek V4.1 shared global attention over RTP's typed CP/PD pools.

The initial eager implementation deliberately shares the established SWA and
FlashMLA kernels with V4. Global compression is non-overlapping (ratio 1/2),
and index keys are projected from its normalized, pre-RoPE latent. Only source
layers write global pools; consumers reuse source KV and index selections.
"""

from __future__ import annotations

import os
from bisect import bisect_left

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4._rope_only_triton import rope_only_inplace
from rtp_llm.models_py.modules.dsv4.attn_type import (
    CSA_KV,
    CSA_STATE,
    HCA_KV,
    INDEXER_KV,
    SWA_KV,
)
from rtp_llm.models_py.modules.dsv4.bounded_replay import (
    enabled as bounded_replay_enabled,
)
from rtp_llm.models_py.modules.dsv4.chunk_env import (
    FLASH_MLA_SPARSE_Q_CHUNK as _FLASH_MLA_SPARSE_Q_CHUNK,
)
from rtp_llm.models_py.modules.dsv4.cp import _cp_restore_gathered_full_2d
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_deepselect as prefill_deepselect
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_indexer_q_triton as indexer_q_fusion
from rtp_llm.models_py.modules.dsv4.fp8 import (
    _v41_prefill_candidates as prefill_candidates,
)
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_global as prefill_global
from rtp_llm.models_py.modules.dsv4.fp8 import (
    _v41_prefill_index_plan as prefill_index_plan,
)
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_indexer as prefill_indexer
from rtp_llm.models_py.modules.dsv4.fp8 import (
    _v41_prefill_kv_workspace as prefill_kv_workspace,
)
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_metadata as prefill_metadata
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

# Global padded-row ceiling for raw CP gather. Mixed batches use large GEMMs
# through 128K rows instead of per-request zigzag projections. Keep the existing
# 64K single-request ceiling; long single requests retain projected transport.
# At hidden=5120 the 128K gather plus restored BF16 storage is bounded to 2.5 GiB.
_SMALL_CP_X_GATHER_MAX_ROWS = int(
    os.environ.get("DSV41_SMALL_CP_X_GATHER_MAX_ROWS", "131072")
)
# Serial groups retain at most one BF16 gather + restore (1280 MiB for
# hidden=5120), plus three group-sized int64 index temporaries (1.5 MiB).
_CP_X_GROUP_MAX_ROWS = 65536
_CP_X_GROUP_MAX_BYTES = 2 * 65536 * 5120 * 2 + 3 * 65536 * 8


def _use_small_cp_x_gather(cp_ctx) -> bool:
    """Whether the single-shot raw-x all-gather path serves this forward."""
    lengths = getattr(cp_ctx, "input_lengths_global_host", None)
    limit = _SMALL_CP_X_GATHER_MAX_ROWS
    if lengths is None or len(lengths) <= 1:
        limit = min(limit, 65536)
    return cp_ctx.padded_seq_len <= limit


def _prefill_x_group_plan(x, cp_ctx):
    """Bound consecutive whole requests; unsupported layouts keep owner tiles."""
    lengths = getattr(cp_ctx, "input_lengths_global_host", None)
    chunks = getattr(cp_ctx, "chunk_lengths_per_req", None)
    restore = getattr(cp_ctx, "unpad_restore", None)
    limit = min(_CP_X_GROUP_MAX_ROWS, _SMALL_CP_X_GATHER_MAX_ROWS)
    if (
        limit <= 0
        or cp_ctx.cp_size != 4
        or _use_small_cp_x_gather(cp_ctx)
        or lengths is None
        or len(lengths) <= 1
        or chunks is None
        or len(chunks) != len(lengths)
        or getattr(cp_ctx, "gather_restore_positions", None) is not None
        or getattr(cp_ctx, "swa_replay_start", None) is not None
        or getattr(cp_ctx, "swa_replay_starts_host", None) is not None
        or x.ndim != 2
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or x.shape[0] != cp_ctx.chunk_length
        or sum(chunks) != cp_ctx.chunk_length
        or sum(lengths) != cp_ctx.seq_len_full
        or cp_ctx.padded_seq_len != cp_ctx.cp_size * cp_ctx.chunk_length
        or restore is None
        or restore.ndim != 1
        or restore.numel() != cp_ctx.seq_len_full
        or restore.dtype != torch.long
        or restore.device != x.device
    ):
        return None
    row_bytes = x.shape[1] * x.element_size()

    def fits(local_rows, real_rows):
        padded = cp_ctx.cp_size * local_rows
        return (
            padded <= limit
            and (padded + real_rows) * row_bytes + 3 * real_rows * 8
            <= _CP_X_GROUP_MAX_BYTES
        )

    groups = []
    local_start = real_start = local_end = real_end = 0
    for length, chunk in zip(lengths, chunks):
        if length <= 0 or chunk != 2 * ((length + 7) // 8) or not fits(chunk, length):
            return None
        if not fits(local_end + chunk - local_start, real_end + length - real_start):
            groups.append((local_start, local_end, real_start, real_end))
            local_start, real_start = local_end, real_end
        local_end += chunk
        real_end += length
    groups.append((local_start, local_end, real_start, real_end))
    return tuple(groups)


def _prefill_raw_x_groups(x, cp_ctx, groups):
    """Group only transport; preserve every original owner/producer tile shape."""
    segments = iter(_prefill_x_tile_plan(cp_ctx))
    t0 = 0
    for local_start, local_end, real_start, real_end in groups:
        local_rows = local_end - local_start
        indices = cp_ctx.unpad_restore[real_start:real_end]
        restore = torch.div(indices, cp_ctx.chunk_length, rounding_mode="floor")
        restore.mul_(local_rows)
        restore.add_(torch.remainder(indices, cp_ctx.chunk_length)).sub_(local_start)
        buffer = x.new_empty((cp_ctx.cp_size * local_rows, x.shape[1]))
        pending = _start_prefill_x_gather_async(
            x[local_start:local_end],
            cp_ctx,
            (None, 0, buffer.shape[0]),
            buffer,
        )
        gathered = _wait_prefill_x_gather(pending)
        full = gathered.index_select(0, restore)
        del pending, gathered, buffer, restore, indices
        try:
            while t0 < real_end:
                _, _, rows = next(segments)
                # Complete-request groups never cut an original owner segment.
                assert t0 + rows <= real_end
                offset = t0 - real_start
                yield t0, full[offset : offset + rows]
                t0 += rows
        finally:
            del full


def _prefill_projected_x_groups(x, cp_ctx, groups, weights, *, whole_groups=False):
    """Project each original owner segment, then gather bounded FP32 groups."""
    from rtp_llm.models_py.modules.dsv4.fp8.compressor import _linear_bf16_bf16_fp32

    batched = None
    if whole_groups and x.is_cuda:
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_batched_producer as batched

    segments = iter(_prefill_x_tile_plan(cp_ctx))
    head = weights[0].shape[0]
    t0 = 0
    for local_start, local_end, real_start, real_end in groups:
        local_rows = local_end - local_start
        send = torch.zeros(
            (local_rows, head * len(weights)), device=x.device, dtype=torch.float32
        )
        projections, group_tiles = [], []
        offsets, columns = [], []
        while t0 < real_end:
            owner, start, rows = next(segments)
            assert t0 + rows <= real_end
            group_tiles.append((t0, rows))
            if owner == cp_ctx.cp_rank:
                offset = start - local_start
                for column, weight in enumerate(weights):
                    # Keep the native GEMM's original M, N and contiguous output.
                    projections.append(
                        _linear_bf16_bf16_fp32(x[start : start + rows], weight)
                    )
                    offsets.append(offset)
                    columns.append(column * head)
            t0 += rows
        if projections:
            if batched is None or not batched.pack_projected_group(
                send, projections, offsets, columns
            ):
                destinations = [
                    send[offset : offset + projection.shape[0], column : column + head]
                    for projection, offset, column in zip(projections, offsets, columns)
                ]
                torch._foreach_copy_(destinations, projections)
                del destinations
        del projections, offsets, columns
        buffer = send.new_empty((cp_ctx.cp_size * local_rows, send.shape[1]))
        pending = _start_prefill_x_gather_async(
            send, cp_ctx, (None, 0, buffer.shape[0]), buffer
        )
        gathered = _wait_prefill_x_gather(pending)
        full = None
        if batched is not None:
            full = batched.restore_projected_group(
                gathered,
                cp_ctx.unpad_restore,
                local_start,
                local_rows,
                cp_ctx.chunk_length,
                real_start,
                real_end - real_start,
            )
        if full is None:
            indices = cp_ctx.unpad_restore[real_start:real_end]
            restore = torch.div(indices, cp_ctx.chunk_length, rounding_mode="floor")
            restore.mul_(local_rows)
            restore.add_(torch.remainder(indices, cp_ctx.chunk_length)).sub_(
                local_start
            )
            full = gathered.index_select(0, restore)
            del restore, indices
        del pending, gathered, buffer, send
        try:
            if whole_groups:
                yield real_start, full
            else:
                for start, rows in group_tiles:
                    offset = start - real_start
                    yield start, full[offset : offset + rows]
        finally:
            del full


def _prefill_request_row_slices(common):
    """Build host query boundaries before entering the attention layer loop.

    CP packs each request's two zigzag halves, including padding, together.
    Global unpadded input lengths therefore cannot describe local query rows.
    Non-CP metadata can require one device read here; layers only use slices.
    """
    if common.batch_size == 1:
        return (slice(0, common.seqlen),)
    if common.cp_on:
        lengths = getattr(common.cp_ctx, "chunk_lengths_per_req", None)
    else:
        lengths = common.input_lengths
        if lengths is not None:
            lengths = lengths.detach().cpu().tolist()
    if lengths is None:
        return None
    if len(lengths) != common.batch_size:
        raise ValueError("V4.1 query row lengths do not match the request count")
    slices = []
    start = 0
    for length in lengths:
        if length < 0:
            raise ValueError("V4.1 query row lengths must be nonnegative")
        stop = start + length
        slices.append(slice(start, stop))
        start = stop
    if start != common.seqlen:
        raise ValueError("V4.1 query row lengths do not cover the local queries")
    return tuple(slices)


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
        # NCCL runs on its internal stream; order the event after completion.
        # With the default nonblocking Work wait, this only adds a GPU edge.
        work.wait()
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


def _prefill_q_rope(x, freqs, rope_dim):
    """Rotate the BF16 projection in place without an FP32 Q-sized temporary."""
    if x.numel() == 0 or rope_dim == 0:
        return x
    if (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and freqs.device == x.device
        and freqs.dtype == torch.complex64
        and freqs.shape == (x.shape[0], rope_dim // 2)
        and 0 < rope_dim <= x.shape[-1]
        and rope_dim % 2 == 0
    ):
        rope_only_inplace(x[..., -rope_dim:], freqs)
        return x
    return rope_only(x, freqs, rope_dim)


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
    retained = shared.get("ced_candidate_rows") if publish else None
    if (
        retained is not None
        and isinstance(rows, slice)
        and rows.start is not None
        and rows.stop is not None
        and rows.step in (None, 1)
    ):
        first = bisect_left(retained, rows.start)
        if first == len(retained) or retained[first] >= rows.stop:
            # Keep intersecting chunks intact, including their TopK tie shape.
            return
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
        # Read once: changing this policy in a running cache namespace is unsafe.
        self.swa_bounded_replay = bounded_replay_enabled() and self.layer_id >= 21
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

    def can_fuse_prefill_attn_norm_input_quant(self, x, norm_weight) -> bool:
        # Small eager inputs regressed in the standalone wrapper benchmark.
        if x.ndim != 2 or x.shape[0] < 32768 or x.shape[1] != 5120:
            return False
        linear = getattr(self, "wq_a_wkv", None)
        if linear is None:
            return False
        from rtp_llm.models_py.modules.dsv4._v41_fused_qkv import (
            is_supported as projection_supported,
        )
        from rtp_llm.models_py.modules.dsv4.fp8._v41_norm_quant import (
            is_supported as norm_supported,
        )

        return norm_supported(x, norm_weight) and projection_supported(
            linear, x, self.q_norm, self.q_lora_rank
        )

    def prefill_fused_attn_norm_input_quant(self, x, norm_weight, norm_eps):
        from rtp_llm.models_py.modules.dsv4.fp8._v41_norm_quant import (
            rmsnorm_group32_quant,
        )

        with record_function_range("dsv41.prefill.qkv.fused_attn_norm_input_quant"):
            result = rmsnorm_group32_quant(x, norm_weight, norm_eps, out_norm=x)
        if result is None:
            raise ValueError("V4.1 norm/quant requires the supported prefill gate")
        norm, quantized, scales = result
        return norm, (quantized, scales)

    def _project_output(self, o, freqs, out=None):
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_output_projection

        o = o.reshape(-1, self.n_heads, self.head_dim)
        if _v41_output_projection.is_supported(
            o, freqs, self._wo_a_stk_w, self._wo_a_stk_s
        ):
            from rtp_llm.models_py.modules.dsv4.utils import V41MXFP8Linear

            if type(self.wo_b) is V41MXFP8Linear:
                quantized = _v41_output_projection.try_grouped_output_quant(
                    o, freqs, self._wo_a_stk_w, self._wo_a_stk_s
                )
                if quantized is not None:
                    return self.wo_b.forward_quantized(*quantized, out=out)

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

    def _prefill_write_swa_fp8_paged(self, common, kv_full, *, fresh_out=None):
        from rtp_llm.models_py.modules.dsv4.cp import cp_swa_replay_starts

        meta = common.swa_meta
        if meta is None or meta.slot_mapping is None:
            return
        kv = kv_full.reshape(-1, self.head_dim).to(torch.bfloat16)
        slots, compaction = meta.slot_mapping, meta.slot_compaction
        replay_start = getattr(common.cp_ctx, "swa_replay_start", None)
        if (
            replay_start is not None
            and getattr(common.cp_ctx, "swa_replay_starts_host", None) is None
        ):
            if kv.shape[0] != self.window_size:
                raise ValueError(
                    "bounded decoder KV must contain exactly 128 rows; "
                    "selected replay rows mismatch"
                )
            slots = slots[replay_start:]
            if compaction is not None:
                compaction = compaction._replace(
                    compact_slots=compaction.compact_slots[replay_start:]
                )
        elif cp_swa_replay_starts(common.cp_ctx) is not None:
            positions = common.cp_ctx.gather_restore_positions
            if positions is None or kv.shape[0] != positions.numel():
                raise ValueError("bounded decoder KV must match selected replay rows")
            slots = slots.index_select(0, positions)
            if compaction is not None:
                compaction = compaction._replace(
                    compact_slots=compaction.compact_slots.index_select(0, positions)
                )
        if self._swa_cp_byte_sliced():
            raw = self._pool_raw_u8(SWA_KV)
            if raw is not None:
                swa_codec.quantize_and_insert_k_cache_cp_byte_sliced(
                    kv,
                    raw,
                    slots,
                    full_entries_per_block=self._swa_entries_per_block(),
                    cp_rank=common.cp_ctx.cp_rank,
                    cp_size=common.cp_ctx.cp_size,
                    compaction=compaction,
                    fresh_out=fresh_out,
                    fresh_slots=meta.slot_in_flat if fresh_out is not None else None,
                )
        else:
            pool = self._pool_view_3d_fp8(SWA_KV)
            if pool is not None:
                swa_codec.quantize_and_insert_swa_k_cache(kv, pool, slots)

    def _v41_prefill_meta_cache_key(self, ratio: int, args, kwargs):
        """Per-forward cache key for the broadcast V4.1 prefill meta build.

        ``build_and_propagate_prefill_meta_fp8`` builds one meta per
        compress-ratio bucket (V4.1 buckets: 0 / 2 / 1). The bucket builds
        receive identical inputs; only the RoPE kind differs (base for ratio 0,
        compressed for ratio 1/2 — every compressed layer's table is built from
        the same model-level rope parameters, so value-identical). Keying on the
        rope parameters plus the per-forward input identities lets the second
        and third bucket builds return the first build's result instead of
        re-running the SWA planner chain (whose byte-sliced slot compaction
        synchronizes via ``nonzero``/``unique``).

        The cache entry pins the input tensors, so ``id()`` recycling cannot
        alias a live key; the entry is dropped by
        ``build_and_propagate_prefill_meta_fp8`` at the start of every forward
        and by ``_begin_forward`` before the layer loop runs.
        """

        def opt(name, index):
            if name in kwargs:
                return kwargs[name]
            if index < len(args):
                return args[index]
            return None

        x = args[0] if args else kwargs.get("x")
        positions = args[1] if len(args) > 1 else kwargs.get("positions")
        if ratio:
            rope_key = (
                "cmp",
                self._rope_base,
                self._rope_max_seq_len,
                self._rope_o_seq_len,
                self._rope_factor,
                self._rope_beta_fast,
                self._rope_beta_slow,
            )
        else:
            rope_key = (
                "base",
                self._rope_dim,
                self._rope_max_seq_len,
                self._rope_factor,
                self._rope_beta_fast,
                self._rope_beta_slow,
            )
        tensors = (
            opt("sp_per_req", 2),
            opt("cu_seqlens", 3),
            opt("input_lengths", 5),
            opt("prefix_lengths", 6),
            opt("position_ids", 7),
            opt("req_id_per_token", 8),
        )
        cp_ctx = getattr(self, "_cp_ctx", None)
        kv_cache = getattr(self, "_kv_cache", None)
        block_tables = getattr(self, "_block_tables_by_type", None)
        return (
            rope_key,
            getattr(self, "_swa_cache_region", SWA_KV),
            int(x.shape[0]) if x is not None else -1,
            str(x.device) if x is not None else "",
            int(positions) if isinstance(positions, int) else id(positions),
            opt("batch_size", 4) or 1,
            opt("max_seqlen_q", 9) or 0,
            tuple(None if t is None else id(t) for t in tensors),
            0 if cp_ctx is None else id(cp_ctx),
            0 if kv_cache is None else id(kv_cache),
            0 if block_tables is None else id(block_tables),
        )

    def _build_shared_prefill_meta(self, *args, **kwargs):
        # Every V4.1 layer needs the full SWA prefix metadata, including global
        # layers. Build through the mature SWA planner with this layer's RoPE.
        ratio = self.compress_ratio
        self.compress_ratio = 0
        # Only compressed bounded consumers bypass prefix concat. Pure SWA
        # still reads combined_indices even if bounded replay was requested.
        kwargs["swa_write_only"] = (
            bool(getattr(self, "swa_bounded_replay", False)) and ratio != 0
        )
        reuse_common = kwargs.get("reuse_common_meta")
        try:
            cache = self._shared_attention.setdefault("prefill_meta_common", {})
            key = self._v41_prefill_meta_cache_key(ratio, args, kwargs)
            if reuse_common is None and key in cache:
                return cache[key]
            common = super()._build_shared_prefill_meta(*args, **kwargs)
            if reuse_common is not None:
                # Parent rechecks this layer's RoPE table but strips SWA Group-2
                # for V4. V4.1 consumes the full same-region broadcast metadata.
                common = common._replace(swa_meta=reuse_common.swa_meta)
            common = common._replace(
                request_row_slices=_prefill_request_row_slices(common)
            )
            cache[key] = common
            return common
        finally:
            self.compress_ratio = ratio

    def _project_prefill_q(self, qr, freqs_cis, workspace):
        rows = qr.shape[0]
        q_out = workspace.prefill_q(rows).view(rows, self.n_heads * self.head_dim)
        if rows == 0:
            return q_out.view(rows, self.n_heads, self.head_dim)
        with record_function_range("dsv41.prefill.q_lora_b_rope"):
            q = self._lin(self.wq_b, qr, out=q_out).view(
                rows, self.n_heads, self.head_dim
            )
            return _prefill_q_rope(q, freqs_cis, self.rope_head_dim)

    def _prefill_sparse_attention(
        self, qkv, common, *, kv, indices, topk_length, profile_name
    ):
        """Project Q into reusable chunk storage immediately before its MLA call."""
        rows = qkv.qr.shape[0]
        if rows == 0:
            out = qkv.qr.new_empty((0, self.dim))
            self._prefill_output_all_reduce(out)
            return out
        from flash_mla import flash_mla_sparse_fwd

        out = torch.empty((rows, self.dim), dtype=torch.bfloat16, device=qkv.qr.device)
        for start in range(0, rows, _FLASH_MLA_SPARSE_Q_CHUNK):
            end = min(start + _FLASH_MLA_SPARSE_Q_CHUNK, rows)
            freqs = common.freqs_cis[start:end]
            q = self._project_prefill_q(qkv.qr[start:end], freqs, common.workspace)
            with record_function_range(profile_name):
                o, _, _ = flash_mla_sparse_fwd(
                    q=q,
                    kv=kv,
                    indices=indices[start:end],
                    sm_scale=self.softmax_scale,
                    attn_sink=self.attn_sink,
                    topk_length=topk_length[start:end],
                )
            with record_function_range("dsv4.fp8.attn.prefill.output_proj"):
                self._prefill_output_proj_into(o, freqs, out=out[start:end])
            dispose_tensor(o)
        self._prefill_output_all_reduce(out)
        return out

    def _begin_forward(self):
        if self.layer_id == min(self._shared_attention["layers"]):
            self._shared_attention["global"] = {}
            self._shared_attention["topk"] = {}
            self._shared_attention["candidates"] = None
            self._shared_attention.pop("candidate_mask", None)
            self._shared_attention.pop("prefill_candidate_mask", None)
            self._shared_attention.pop("prefill_sparse_candidates", None)
            self._shared_attention.pop("prefill_sparse_plans", None)
            self._shared_attention.pop("prefill_score_bounds", None)
            # ``_prefill_chunk_meta`` caches per-source-group chunk offsets for
            # the duration of one forward; drop them with the rest of the
            # per-forward shared state.
            self._shared_attention.pop("prefill_chunk_meta", None)
            self._shared_attention.pop("prefill_kv_workspace", None)
            # The layer-invariant sparse index plan (global/SWA gather indices
            # + per-row lengths) is cached per index-source group and dropped
            # with the rest of the per-forward shared state.
            self._shared_attention.pop("prefill_index_plan", None)
            # Per-forward broadcast prefill-meta cache (see
            # ``_build_shared_prefill_meta``): dropped with the rest of the
            # per-forward shared state so a later forward can never observe a
            # stale entry.
            self._shared_attention.pop("prefill_meta_common", None)

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
        slots = prefill_metadata.try_slot_mapping(
            positions,
            req_ids,
            bt,
            eb,
            tpb,
            self.compress_ratio,
            cp.cp_size if sharded else 1,
            cp.cp_rank if sharded else 0,
            owner_tokens_per_block=(
                self._kv_cache.seq_size_per_block if sharded else tpb
            ),
            state=region == CSA_STATE,
            seq_ends=state_end,
        )
        if slots is not None:
            return slots
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

    def _read_state(self, positions, req_ids, *, slots=None):
        if slots is None:
            slots = self._slots(CSA_STATE, positions, req_ids)
        if slots is None:
            return torch.zeros(
                len(positions), 2 * self.head_dim, device=positions.device
            )
        pool = self._source_pool(CSA_STATE)
        state = pool[slots.clamp_min(0)].clone()
        state.masked_fill_((slots < 0).unsqueeze(-1), 0)
        return self._gather_shards(state)

    def _write_states(
        self, values, scores, positions, req_ids, seq_ends, *, slots=None
    ):
        if slots is None:
            slots = self._slots(CSA_STATE, positions, req_ids, state_end=seq_ends)
        if slots is not None:
            pool = self._source_pool(CSA_STATE)
            if prefill_global.store_states(values, scores, slots, pool):
                return
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
        raw_tiles=None,
        grouped_projection=False,
        batched_groups=False,
    ):
        """Publish global pools in sequence order, carrying pairs across tiles."""
        if projected_tiles is not None and raw_tiles is not None:
            raise ValueError("Global producer accepts either raw or projected tiles")
        if batched_groups and (not grouped_projection or projected_tiles is None):
            raise ValueError("Batched producer requires complete projected groups")
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
        # Batch raw gather uses fewer, bounded GEMMs; retain original owner and
        # single-request partitioning, including unbound startup materialization.
        tile_rows = (
            65536
            if prefill
            and main_pool is not None
            and index_pool is not None
            and cp is not None
            and cp.cp_size == 4
            and len(lengths_host) > 1
            and raw_tiles is None
            and projected_tiles is None
            else _PRODUCE_GLOBAL_TILE_ROWS
        )
        ends = [s + l for s, l in zip(prefixes_host, lengths_host)]
        pair_ranges = []
        base = 0
        for start, length in zip(prefixes_host, lengths_host):
            pair_ranges.append((base + (1 - start % 2), base + length))
            base += length
        producer_meta = None
        state_pool = None
        if (
            prefill
            and main_pool is not None
            and index_pool is not None
            and cp is not None
            and cp.cp_size == 4
            and getattr(cp, "kv_cache_sharded", False)
            and len(lengths_host) >= 2
            and raw_tiles is None
            and projected_tiles is None
        ):
            from . import _v41_producer_metadata

            regions = (self._global_region(), INDEXER_KV)
            if ratio == 2:
                regions += (CSA_STATE,)
            sharded = cp.kv_cache_sharded
            layouts = []
            for region in regions:
                pool = self._source_pool(region)
                if pool is None or self._block_tables_by_type is None:
                    break
                tpb = require_pool_tokens_per_block(self._kv_cache, region=region)
                layouts.append(
                    _v41_producer_metadata.SlotLayout(
                        self._block_tables_by_type[region],
                        self._source_entries(region, pool),
                        tpb,
                        self._kv_cache.seq_size_per_block if sharded else tpb,
                        cp.cp_size if sharded else 1,
                        cp.cp_rank if sharded else 0,
                    )
                )
            producer_meta = _v41_producer_metadata.prepare_raw(
                cp,
                positions,
                req_ids,
                starts,
                lengths,
                ratio,
                self._slots,
                (self._global_region(), INDEXER_KV, CSA_STATE),
                tile_rows=tile_rows,
                slot_layouts=tuple(layouts),
            )
            if producer_meta is not None:
                starts, lengths = producer_meta.starts, producer_meta.lengths
                score_cache = self._shared_attention.setdefault(
                    "prefill_score_bounds", {}
                )
                counts_key = ("producer_key_counts", ratio)
                cached = score_cache.get(
                    ("batch_key_counts", ratio), score_cache.get(counts_key)
                )
                if not (
                    isinstance(cached, tuple)
                    and len(cached) == 3
                    and cached[0] is cp.prefix_lengths
                    and cached[1] is cp.input_lengths_global
                ):
                    cached = (
                        cp.prefix_lengths,
                        cp.input_lengths_global,
                        producer_meta.key_counts,
                    )
                score_cache[counts_key] = cached
                state_pool = self._source_pool(CSA_STATE) if ratio == 2 else None
        if producer_meta is None:
            # Unsupported descriptors preserve the old dtype and callback path.
            starts, lengths = starts.long(), lengths.long()
        if (
            producer_meta is None
            and prefill
            and main_pool is not None
            and index_pool is not None
            and (
                raw_tiles is not None
                or grouped_projection
                or (
                    ratio == 2
                    and projected_tiles is None
                    and cp is not None
                    and cp.cp_size == 4
                )
            )
        ):
            from rtp_llm.models_py.modules.dsv4.fp8 import _v41_producer_metadata

            state_pool = self._source_pool(CSA_STATE) if ratio == 2 else None
            if ratio == 1 or state_pool is not None:
                regions = (self._global_region(), INDEXER_KV, CSA_STATE)
                if raw_tiles is None and projected_tiles is None:
                    producer_meta = _v41_producer_metadata.prepare_raw(
                        cp,
                        positions,
                        req_ids,
                        starts,
                        lengths,
                        ratio,
                        self._slots,
                        regions,
                        tile_rows=tile_rows,
                    )
                else:
                    producer_meta = _v41_producer_metadata.prepare(
                        cp,
                        positions,
                        req_ids,
                        starts,
                        lengths,
                        ratio,
                        _prefill_x_tile_plan(cp),
                        self._slots,
                        regions,
                    )
        if batched_groups:
            from . import _v41_batched_producer as batched_producer

            if producer_meta is None:
                raise ValueError(
                    "Batched producer requires the original segment metadata"
                )
        previous = None
        producer_seq_ends = (
            producer_meta.seq_ends if producer_meta is not None else None
        )
        if ratio == 2:
            # Read the previous token before tail writes can reuse its ring slot.
            if producer_meta is not None and producer_meta.previous_slots is not None:
                previous = self._read_state(
                    None, None, slots=producer_meta.previous_slots
                )
            else:
                previous = self._read_state(
                    (starts - 1).clamp_min(0),
                    torch.arange(len(starts), device=x_full.device),
                )
        # Warmup (pool unbound) still needs the full keys for the shared
        # materialization, so it keeps every tile's keys; the pool path drops
        # them right after quantization.
        warm_keys = [] if main_pool is None else None
        carry = None
        tiles = raw_tiles if raw_tiles is not None else projected_tiles
        if tiles is None:
            tiles = (
                (t0, x_full[t0 : t0 + tile_rows])
                for t0 in range(0, max(rows, 1), tile_rows)
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
            scores = None
            if ratio == 2:
                scores = (
                    x_tile[:, self.head_dim :]
                    if projected_tiles is not None
                    else _linear_bf16_bf16_fp32(x_tile, owner.global_wgate)
                )
            group_plan = None
            if batched_groups:
                group_plan = batched_producer.make_plan(
                    producer_meta, t0, t1, ratio=ratio, request_lengths=lengths_host
                )
                prepared = batched_producer.prepare(group_plan, x_full.device)
                if prepared is None:
                    raise ValueError("Unsupported batched producer group")
                compact = slice(group_plan.first, group_plan.stop)
                boundary_idx = prepared.boundaries
                boundary_pos = producer_meta.positions[compact]
                boundary_req = producer_meta.requests[compact]
                main_slots = producer_meta.main_slots[compact]
                index_slots = producer_meta.index_slots[compact]
                state_slots = producer_meta.state_slots[t0:t1] if ratio == 2 else None
            elif producer_meta is not None:
                (
                    boundary_idx,
                    boundary_pos,
                    boundary_req,
                    main_slots,
                    index_slots,
                    state_slots,
                ) = producer_meta.tile(t0, t1)
            elif ratio == 2:
                # Host lengths retain compact completed-pair rows without
                # nonzero or doubling the index projection's GEMM row count.
                pair_cache = self._shared_attention.setdefault(
                    "prefill_meta_common", {}
                ).setdefault("global_pairs", {})
                pair_key = (tuple(pair_ranges), t0, t1, x_full.device)
                boundary_idx = pair_cache.get(pair_key)
                if boundary_idx is None:
                    pair_indices = []
                    for first, end in pair_ranges:
                        begin = max(first, t0 + (first - t0) % 2)
                        if begin < min(end, t1):
                            pair_indices.append(
                                torch.arange(
                                    begin - t0,
                                    min(end, t1) - t0,
                                    2,
                                    device=x_full.device,
                                )
                            )
                    boundary_idx = (
                        torch.cat(pair_indices)
                        if pair_indices
                        else torch.empty(0, dtype=torch.long, device=x_full.device)
                    )
                    pair_cache[pair_key] = boundary_idx
                boundary_pos, boundary_req = (
                    pos_tile[boundary_idx],
                    req_tile[boundary_idx],
                )
            else:
                boundary_idx = torch.arange(t1 - t0, device=x_full.device)
                boundary_pos, boundary_req = pos_tile, req_tile
            if producer_meta is None:
                main_slots = (
                    self._slots(self._global_region(), boundary_pos, boundary_req)
                    if main_pool is not None
                    else None
                )
                index_slots = (
                    self._slots(INDEXER_KV, boundary_pos, boundary_req)
                    if index_pool is not None
                    else None
                )
            latent = (
                batched_producer.compress_main(
                    values,
                    scores,
                    owner.global_norm,
                    self.eps,
                    pos_tile,
                    req_tile,
                    starts,
                    previous,
                    prepared,
                    self.freqs_cis,
                    main_pool,
                    main_slots,
                )
                if batched_groups
                else (
                    prefill_global.compress_main(
                        values,
                        scores,
                        owner.global_norm,
                        self.eps,
                        pos_tile,
                        req_tile,
                        starts,
                        previous,
                        boundary_idx,
                        self.freqs_cis,
                        main_pool,
                        main_slots,
                        ratio,
                        carry,
                    )
                    if prefill and main_pool is not None
                    else None
                )
            )
            if batched_groups and latent is None:
                raise ValueError("Batched compressor rejected the projected group")
            main_stored = latent is not None
            if latent is None:
                if ratio == 2:
                    head_values, head_scores = (
                        carry if carry is not None else (values[:1], scores[:1])
                    )
                    value_prev = torch.cat((head_values, values[:-1]), 0)
                    score_prev = torch.cat((head_scores, scores[:-1]), 0)
                    first = pos_tile == starts[req_tile]
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
                    latent = compress_pairs(
                        torch.stack(
                            (value_prev[boundary_idx], values[boundary_idx]), 1
                        ),
                        torch.stack(
                            (score_prev[boundary_idx], scores[boundary_idx]), 1
                        ),
                        owner.global_norm,
                        self.eps,
                    )
                else:
                    latent = rms_norm(values, owner.global_norm, self.eps).to(
                        torch.bfloat16
                    )
            if ratio == 2:
                # The immutable predecessor snapshot and old carry are consumed
                # before any ring writes or publication of the next tile carry.
                if batched_groups:
                    if not batched_producer.store_states(
                        values, scores, state_slots, state_pool, slots_are_unique=True
                    ):
                        raise ValueError("Batched producer rejected state publication")
                elif producer_meta is None:
                    self._write_states(
                        values,
                        scores,
                        pos_tile,
                        req_tile,
                        starts + lengths,
                    )
                else:
                    self._write_states(
                        values,
                        scores,
                        pos_tile,
                        req_tile,
                        producer_meta.seq_ends,
                        slots=state_slots,
                    )
                if not batched_groups:
                    carry = (values[-1:].clone(), scores[-1:].clone())
            if batched_groups:
                from ._v41_grouped_gemm import try_grouped_index_gemm

                projected_index = try_grouped_index_gemm(
                    latent,
                    owner.index_wk,
                    tuple(stop - first for _, _, first, stop, _ in group_plan.segments),
                )
                if projected_index is None:
                    pieces = [
                        F.linear(latent[first:stop], owner.index_wk)
                        for _, _, first, stop, _ in group_plan.segments
                        if stop > first
                    ]
                    projected_index = (
                        torch.cat(pieces, dim=0)
                        if pieces
                        else latent.new_empty((0, 128))
                    )
                    del pieces
            else:
                projected_index = F.linear(latent, owner.index_wk)
            index_stored = (
                prefill
                and main_pool is not None
                and index_pool is not None
                and (
                    batched_producer.store_index
                    if batched_groups
                    else prefill_global.store_index
                )(
                    projected_index,
                    owner.index_k_norm,
                    self.eps,
                    boundary_pos,
                    self.freqs_cis,
                    index_pool,
                    index_slots,
                    ratio,
                )
            )
            if not main_stored or not index_stored:
                freqs = self.freqs_cis[(boundary_pos // ratio) * ratio]
            if not main_stored:
                global_keys = rope_only(latent.clone(), freqs, self.rope_head_dim)
            if not index_stored:
                index_keys = rope_only(
                    rms_norm(projected_index, owner.index_k_norm, self.eps),
                    freqs,
                    self.rope_head_dim,
                )
            if main_pool is not None:
                from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_triton import (
                    quantize_and_insert_k_cache_fp4,
                    quantize_indexer_k_fp4,
                )

                if not main_stored:
                    quantize_and_insert_k_cache_fp4(
                        global_keys.contiguous(), main_pool, main_slots
                    )
                if not index_stored:
                    quantize_indexer_k_fp4(
                        index_keys.contiguous(), index_slots, index_pool
                    )
            else:
                warm_keys.append((boundary_pos, boundary_req, global_keys, index_keys))
            # Release all consumer views before the next group is allocated.
            # The independent carry clones survive this storage's lifetime.
            if raw_tiles is not None or projected_tiles is not None:
                del x_tile, values, scores
            if batched_groups:
                # These group-sized outputs are dead after publication. Release
                # them before the next group and the final full-cache gather.
                del latent, projected_index, prepared, boundary_idx
        if producer_meta is not None and producer_meta.key_counts is not None:
            # Row views must not keep the fused slab alive through full-pool readback.
            producer_meta = None
            main_slots = index_slots = state_slots = None
            boundary_idx = boundary_pos = boundary_req = None
        if warm_keys:
            boundary_pos = torch.cat([tile[0] for tile in warm_keys])
            boundary_req = torch.cat([tile[1] for tile in warm_keys])
            global_keys = torch.cat([tile[2] for tile in warm_keys])
            index_keys = torch.cat([tile[3] for tile in warm_keys])
        result = []
        fused_indexer = prefill and prefill_indexer.is_supported(
            x_full.device, getattr(self, "index_n_heads", 0), owner.index_wk.shape[0]
        )
        if (
            fused_indexer
            and main_pool is not None
            and index_pool is not None
            and len(ends) > 1
        ):
            from rtp_llm.models_py.modules.dsv4.fp8._v41_prefill_pools import (
                try_gather_prefill_pools,
            )

            batched = try_gather_prefill_pools(
                self,
                main_pool,
                index_pool,
                ends,
                (
                    producer_seq_ends
                    if producer_seq_ends is not None
                    else starts + lengths
                ),
            )
            if batched is not None:
                self._shared_attention["global"] = {self.layer_id: batched}
                self._shared_attention.pop("prefill_chunk_meta", None)
                return
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

    def _select_indices(self, x, qr, positions, req_ids, *, request_row_slices=None):
        shared = self._shared_attention
        if not self.is_index_source:
            return shared["topk"][self.index_source_layer_id]
        globals_by_req = shared["global"][self.kv_source_layer_id]
        if request_row_slices is not None:
            if len(request_row_slices) != len(globals_by_req):
                raise ValueError("V4.1 query slices do not match the request count")
            end = 0
            for rows in request_row_slices:
                if (
                    not isinstance(rows, slice)
                    or rows.step not in (None, 1)
                    or rows.start != end
                    or rows.stop is None
                    or rows.stop < end
                ):
                    raise ValueError("V4.1 query slices must form contiguous ranges")
                end = rows.stop
            if end != x.shape[0]:
                raise ValueError("V4.1 query slices do not cover the local queries")
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
                self.index_topk == 512
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
            ced_projection = shared.get("ced_indexer_projection")
            if ced_projection is None:
                raw_weights = F.linear(x, self.index_weights)
            else:
                raw_weights = ced_projection(x, self.index_weights)
            fused_indexer = isinstance(
                globals_by_req[0][1], prefill_indexer.PrefillIndexerKeys
            )
            prepared = (
                indexer_q_fusion.try_fused_indexer_q(
                    q.unsqueeze(1),
                    raw_weights.unsqueeze(1),
                    self.freqs_cis,
                    positions,
                    self.rope_head_dim,
                )
                if fused_indexer
                else None
            )
            if prepared is not None:
                q_fp4, q_sf, weights = (value.squeeze(1) for value in prepared)
            else:
                q = rope_only(q, self.freqs_cis[positions], self.rope_head_dim)
                weights = (
                    raw_weights.float()
                    * (self.index_head_dim * self.index_n_heads) ** -0.5
                )
                if fused_indexer:
                    q_fp4, q_sf = prefill_indexer.quantize_indexer_q(q)
                else:
                    q = fp8_roundtrip(q)
            if fused_indexer and len(globals_by_req) > 1:
                from ._v41_batched_prefill_select import try_select_batched

                if try_select_batched(
                    self,
                    q_fp4,
                    q_sf,
                    weights,
                    globals_by_req,
                    request_row_slices,
                    positions,
                    out,
                    candidate_source=candidate_source,
                    publish_candidates=publish_candidates,
                    candidate_size=candidate_size,
                    candidate_blocks=candidate_blocks,
                    req_ids=req_ids,
                ):
                    shared["topk"] = {self.layer_id: out}
                    return out
            single_request = len(globals_by_req) == 1
            for b, (_, keys) in enumerate(globals_by_req):
                # Production queries are packed in request order. Host slices
                # avoid nonzero's D2H sizing sync and all row gather/scatter.
                # Keep indexed selection for callers without that contract.
                contiguous = request_row_slices is not None or single_request
                if request_row_slices is not None:
                    rows = request_row_slices[b]
                elif single_request:
                    rows = slice(0, x.shape[0])
                else:
                    rows = torch.where(req_ids == b)[0]
                row_count = rows.stop - rows.start if contiguous else rows.numel()
                if row_count == 0:
                    continue
                key_count = len(keys)
                candidates = shared.get("candidates")
                probe = slice(rows.start, rows.start + 1) if contiguous else rows[:1]
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
                request_bounds = None
                if contiguous and fused_indexer:
                    cache = shared.setdefault("prefill_score_bounds", {})
                    bounds_key = (
                        b,
                        rows.start,
                        rows.stop,
                        key_count,
                        self.compress_ratio,
                    )
                    request_bounds = cache.get(bounds_key)
                    if request_bounds is None:
                        request_bounds = prefill_metadata.try_score_bounds(
                            positions[rows], key_count, self.compress_ratio
                        )
                        if request_bounds is not None:
                            cache[bounds_key] = request_bounds
                for chunk_index, start in enumerate(range(0, row_count, chunk_rows)):
                    stop = min(start + chunk_rows, row_count)
                    output_rows = (
                        slice(rows.start + start, rows.start + stop)
                        if contiguous
                        else rows[start:stop]
                    )
                    chunk = output_rows
                    bounds = (
                        tuple(t[start:stop] for t in request_bounds)
                        if request_bounds is not None
                        else None
                    )
                    visible = (
                        bounds[1]
                        if bounds is not None
                        else (positions[chunk] + 1) // self.compress_ratio
                    )
                    if sparse_request:
                        with record_function_range(
                            "dsv41.prefill.indexer.sparse_logits"
                        ):
                            plan = _prefill_sparse_plan(
                                shared,
                                (
                                    b,
                                    chunk_index,
                                    stop - start,
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
                                out=out[output_rows] if contiguous else None,
                            )
                            if mapped is None:
                                raise RuntimeError(
                                    "V4.1 sparse prefill remap layout is unsupported"
                                )
                            if not contiguous:
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
                                **({"bounds": bounds} if bounds is not None else {}),
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
                    target = out[output_rows, :k] if contiguous else None
                    selected = prefill_topk.try_select_tokens(
                        logits,
                        visible,
                        k,
                        bounds=bounds,
                        out=target,
                    )
                    if selected is None:
                        scores, selected = logits.topk(k, dim=-1)
                        selected = torch.where(scores.isfinite(), selected, -1).int()
                    if selected is not target:
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

    def _can_fuse_swa_fresh(self, qkv, common):
        meta = common.swa_meta
        if (
            self.compress_ratio not in (1, 2)
            or self.swa_bounded_replay
            or not common.any_cont
            or not common.cp_on
            or getattr(common.cp_ctx, "swa_replay_start", None) is not None
            or getattr(common.cp_ctx, "swa_replay_starts_host", None) is not None
            or meta is None
            or meta.slot_mapping is None
            or meta.cache_slot_mapping is None
            or meta.cache_gather_lens is None
            or not self._swa_cp_byte_sliced()
        ):
            return False
        return swa_codec.is_supported_fresh_store(
            qkv.kv_full,
            self._pool_raw_u8(SWA_KV),
            meta.slot_mapping,
            self._swa_entries_per_block(),
            common.cp_ctx.cp_rank,
            common.cp_ctx.cp_size,
            meta.slot_compaction,
            meta.slot_in_flat,
        )

    def _swa_prefill_workspace(self, qkv, common, *, fuse_cache_write=False):
        """Return BF16 [request, prefix tail + new tokens] for sparse prefill."""
        from rtp_llm.models_py.modules.dsv4.cp import cp_swa_replay_starts

        lengths_host = self._host_prefill_lengths(common)
        if self.swa_bounded_replay:
            prefixes = self._host_prefill_prefixes(common)
            if any(
                p > 0 and n < self.window_size for p, n in zip(prefixes, lengths_host)
            ):
                raise ValueError(
                    "bounded replay cache reuse must leave 128 fresh tokens"
                )
            replay_starts = cp_swa_replay_starts(common.cp_ctx)
            if replay_starts is not None:
                sizes = [min(n, self.window_size) for n in lengths_host]
                if (
                    len(replay_starts) != len(sizes)
                    or len(prefixes) != len(sizes)
                    or any(
                        s != n - size
                        for n, size, s in zip(lengths_host, sizes, replay_starts)
                    )
                    or qkv.kv_full.shape[0] != sum(sizes)
                ):
                    raise ValueError(
                        "bounded decoder KV must match per-request replay rows"
                    )
                return list(qkv.kv_full.split(sizes)), [
                    p + s for p, s in zip(prefixes, replay_starts)
                ]
            # Non-compacted paths recompute the fresh suffix. They must never
            # consume decoder state from a previous request's approximate cache.
            # Native prefix matching leaves at least one complete SWA window.
            return list(qkv.kv_full.split(lengths_host)), prefixes
        if not common.any_cont:
            return list(qkv.kv_full.split(lengths_host)), [0] * len(lengths_host)
        buf = self._swa_prefill_concat(qkv, common, fuse_cache_write=fuse_cache_write)
        prefixes_host = self._host_prefill_prefixes(common)
        tails = [min(int(p), self.window_size - 1) for p in prefixes_host]
        return [
            buf[b, : tails[b] + int(lengths_host[b])] for b in range(common.batch_size)
        ], [int(p) - t for p, t in zip(prefixes_host, tails)]

    def _swa_prefill_concat(self, qkv, common, *, fuse_cache_write=False):
        """Read the 528B prefix before a new chunk overwrites the SWA ring."""
        meta = common.swa_meta
        B, D = common.batch_size, self.head_dim
        allocate = torch.empty if fuse_cache_write else torch.zeros
        buf = allocate(B, meta.M, D, dtype=torch.bfloat16, device=qkv.kv_full.device)
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
        if fuse_cache_write:
            # Prefix reads finish first; all visible fresh rows are then stored
            # even when the SWA ring masks their cache slots. Padding is unused.
            self._prefill_write_swa_fp8_paged(common, qkv.kv_full, fresh_out=buf)
        else:
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
                from rtp_llm.models_py.modules.dsv4.cp import cp_swa_replay_starts

                from ._v41_prefill_metadata import try_chunk_metadata

                compact_replay = cp_swa_replay_starts(common.cp_ctx) is not None
                meta = (
                    None
                    if compact_replay
                    else try_chunk_metadata(
                        common.prefix_lengths,
                        (
                            common.cp_ctx.input_lengths_global
                            if common.cp_on
                            else common.input_lengths
                        ),
                        req_ids,
                        self.compress_ratio,
                        self.window_size,
                        self.swa_bounded_replay,
                    )
                )
                if meta is not None:
                    shared["prefill_chunk_meta"] = meta
                    return meta
                lengths = (
                    common.cp_ctx.input_lengths_global
                    if common.cp_on
                    else common.input_lengths
                ).long()
                prefixes = common.prefix_lengths.long()
                tails = (
                    torch.zeros_like(prefixes)
                    if self.swa_bounded_replay
                    else prefixes.clamp(max=self.window_size - 1)
                )
                sizes = (prefixes + lengths) // self.compress_ratio
                swa_lengths = (
                    lengths.clamp(max=self.window_size) if compact_replay else lengths
                )
                chunk_sizes = sizes + swa_lengths + tails
                offsets_d = chunk_sizes.cumsum(0) - chunk_sizes
                starts_d = (
                    prefixes + lengths - swa_lengths
                    if compact_replay
                    else prefixes - tails
                )
            meta = (
                offsets_d[req_ids, None],
                sizes[req_ids, None],
                starts_d[req_ids, None],
            )
        shared["prefill_chunk_meta"] = meta
        return meta

    def _forward_prefill(self, x, positions, shared_input_quant=None):
        prepared = self._prefill_produce(x, positions, shared_input_quant)
        return self._prefill_query(x, *prepared)

    def _prefill_produce(self, x, positions, shared_input_quant=None):
        """Finish all source writes and prefix snapshots in the original CP domain."""
        self._begin_forward()
        if self.swa_bounded_replay and self.layer_id == 21:
            # L20 has exact cached SWA; decoder replay starts a new SWA domain.
            self._shared_attention.pop("prefill_chunk_meta", None)
            self._shared_attention.pop("prefill_index_plan", None)
        common = self._prefill_common_setup(x, positions)
        qkv = self._prefill_compute_qkv(
            x, common, shared_input_quant=shared_input_quant
        )
        # Launch the first bounded hidden-state transfer after the QKV gather.
        # It overlaps SWA work; subsequent transfers overlap global projection.
        x_gather = None
        small_cp = common.cp_on and _use_small_cp_x_gather(common.cp_ctx)
        x_groups = (
            _prefill_x_group_plan(x, common.cp_ctx)
            if self.is_kv_source and common.cp_on and not small_cp
            else None
        )
        if self.is_kv_source and common.cp_on and x_groups is None:
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
        fuse_swa_fresh = self._can_fuse_swa_fresh(qkv, common)
        swa, swa_starts = (
            self._swa_prefill_workspace(qkv, common, fuse_cache_write=fuse_swa_fresh)
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
        if not fuse_swa_fresh:
            self._prefill_write_swa_fp8_paged(common, qkv.kv_full)
        if not self.compress_ratio or not self.is_kv_source:
            return common, qkv, swa, swa_starts, swa_only_workspace
        starts = common.prefix_lengths
        lengths = (
            common.cp_ctx.input_lengths_global if common.cp_on else common.input_lengths
        )
        if not (
            small_cp
            and common.cp_ctx.cp_size == 4
            and common.cp_ctx.kv_cache_sharded
            and common.batch_size >= 2
        ):
            starts, lengths = starts.long(), lengths.long()
        if self.is_kv_source:
            rows = common.cp_ctx.seq_len_full if common.cp_on else x.shape[0]
            x_full, projected_tiles, raw_tiles = x, None, None
            batched_groups = False
            if small_cp:
                x_full = _cp_restore_gathered_full_2d(
                    _wait_prefill_x_gather(x_gather), common.cp_ctx
                )
            elif x_groups is not None:
                if os.environ.get("RTP_V41_BATCHED_PRODUCER", "0") == "1":
                    from . import _v41_batched_producer

                    batched_groups = _v41_batched_producer.can_batch_groups(
                        self, x, common.cp_ctx
                    )
                owner = self._owner()
                weights = (
                    (owner.global_wkv, owner.global_wgate)
                    if self.compress_ratio == 2
                    else (owner.global_wkv,)
                )
                projected_tiles = _prefill_projected_x_groups(
                    x, common.cp_ctx, x_groups, weights, whole_groups=batched_groups
                )
            elif common.cp_on:
                projected_tiles = _prefill_x_tiles(
                    x, common.cp_ctx, x_plan, x_gather, x_buffers, weights
                )
            x_gather = None
            if common.cp_on and x_groups is None:
                del x_buffers
            row_metadata = None
            if (
                common.cp_on
                and common.batch_size > 1
                and x.is_cuda
                and starts.is_contiguous()
                and lengths.is_contiguous()
            ):
                from rtp_llm.models_py.modules.dsv4._cp_metadata_triton import (
                    try_build_cp_full_prefill_positions,
                )

                row_metadata = try_build_cp_full_prefill_positions(
                    lengths, starts, total_tokens=rows
                )
            if row_metadata is not None:
                pos_full, ids_full = row_metadata
                del row_metadata
            else:
                ids_full = torch.repeat_interleave(
                    torch.arange(common.batch_size, device=x.device),
                    lengths,
                    output_size=rows,
                )
                cu = torch.cat(
                    (
                        torch.zeros(1, device=x.device, dtype=torch.long),
                        lengths.cumsum(0),
                    )
                )
                pos_full = (
                    torch.arange(rows, device=x.device)
                    - cu[ids_full]
                    + starts[ids_full]
                )
            try:
                self._produce_global(
                    x_full,
                    pos_full,
                    ids_full,
                    starts,
                    lengths,
                    prefill=True,
                    projected_tiles=projected_tiles,
                    raw_tiles=raw_tiles,
                    grouped_projection=x_groups is not None,
                    batched_groups=batched_groups,
                )
            finally:
                if raw_tiles is not None:
                    raw_tiles.close()
                if projected_tiles is not None:
                    projected_tiles.close()
            del x_full
        return common, qkv, swa, swa_starts, swa_only_workspace

    def _prefill_query(self, x, common, qkv, swa, swa_starts, swa_only_workspace=None):
        """Consume query rows without regenerating or writing source KV."""
        if not self.compress_ratio:
            if swa_only_workspace is not None:
                dispose_tensor(qkv.kv_full)
                meta = common.swa_meta
                return self._prefill_sparse_attention(
                    qkv,
                    common,
                    kv=swa_only_workspace.view(-1, 1, self.head_dim),
                    indices=meta.combined_indices.unsqueeze(1),
                    topk_length=meta.combined_lens,
                    profile_name="dsv41.prefill.swa_concat.flash_mla",
                )
            topk = common.topk_idxs
            if topk.dim() == 3:
                topk = topk.squeeze(0)
            out = self._prefill_sparse_attention(
                qkv,
                common,
                kv=qkv.kv_full.unsqueeze(1),
                indices=topk.unsqueeze(1).to(torch.int32),
                topk_length=common.swa_meta.topk_length_kv_full,
                profile_name="dsv41.prefill.swa.flash_mla_kv_full",
            )
            dispose_tensor(qkv.kv_full)
            return out
        positions = (
            common.cp_ctx.global_positions if common.cp_on else common.position_ids
        )
        req_ids = common.req_id_per_token
        cache = self._shared_attention.setdefault("prefill_meta_common", {})
        vectors = cache.get("query_vectors")
        if vectors is None or vectors[0] is not positions or vectors[1] is not req_ids:
            vectors = (positions, req_ids, positions.long(), req_ids.long())
            cache["query_vectors"] = vectors
        positions, req_ids = vectors[2:]
        selected = self._select_indices(
            x,
            qkv.qr,
            positions,
            req_ids,
            request_row_slices=getattr(common, "request_row_slices", None),
        )
        globals_by_req = self._shared_attention["global"][self.kv_source_layer_id]
        offsets, ns, swstart = self._prefill_chunk_meta(
            globals_by_req, swa, swa_starts, req_ids, x.device, common=common
        )
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
            fused_plan = prefill_index_plan.try_build_index_plan(
                selected, positions, offsets, ns, swstart, self.window_size
            )
            if fused_plan is not None:
                indices, lens = fused_plan
            else:
                swpos = (
                    positions[:, None]
                    - self.window_size
                    + 1
                    + torch.arange(self.window_size, device=x.device)[None]
                )
                swidx = torch.where(
                    swpos >= swstart, offsets + ns + swpos - swstart, -1
                )
                global_idx = torch.where(selected >= 0, offsets + selected, -1)
                indices = torch.cat((global_idx, swidx), -1).int()
                # FlashMLA's length bounds count a compact valid prefix.
                indices = indices.gather(
                    1, torch.argsort(indices < 0, dim=-1, stable=True)
                )
                lens = (indices >= 0).sum(-1).int()
                if indices.shape[1] % 64:
                    indices = F.pad(indices, (0, 64 - indices.shape[1] % 64), value=-1)
            self._shared_attention["prefill_index_plan"] = (selected, indices, lens)
        return self._prefill_sparse_attention(
            qkv,
            common,
            kv=prefill_kv_workspace.combine_kv(
                self._shared_attention, globals_by_req, swa
            ).unsqueeze(1),
            indices=indices.unsqueeze(1),
            topk_length=lens,
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
            selected.is_cuda
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
