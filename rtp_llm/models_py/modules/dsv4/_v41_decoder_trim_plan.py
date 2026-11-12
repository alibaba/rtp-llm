"""R48d — decoder trim via a compact consumer view on the FINAL chunk (precision class).

Flag ``DSV41_DECODER_TRIM_KEEP`` (int; absent/0 = OFF = today's exact path;
140 = the DeepInfer arm -> CP-legal compact tail L=144). Default OFF. Non-exact
by design (precision class): the decoder layers L21-39 run only on the final
chunk's last ``decoder_trim_tail()`` tokens instead of all 4096. The inexactness
is the SWA-window cascade (the tail's early tokens' windows extend into the
final chunk's skipped pre-tail region), which the ``ced_swa_start`` fresh-only
view replicates exactly — so mask-mode (full compute + visibility limited) and
compact-mode (truncated compute) produce the same final-token output.

This module mirrors the shipped, smoke-tested CED machinery
(`prefill/_v41_ced.py`) with three deltas: (a) TAIL=144 instead of 3072; (b)
engaged only on the final chunk (CED compacts every stage-1 chunk); (c)
precision-class, so CED's byte-exactness geometry guard
``(n-TAIL) % (rows//2) == 0`` is relaxed (the all-gather + re-zigzag in
``_gather_canonical``/``restore_rows`` is tail-alignment-agnostic). The R48c
"CP4 zigzag divisibility" blocker was that exactness guard, not a geometric
constraint — see PROGRESS.md R48d for the alignment math.

Byte-exactness contract: flag OFF => the module is inert (the plan is None and
the forward path is byte-identical to today). Flag ON changes the output
(precision class), so it is gated by the accuracy bar, NOT the byte-exact sha.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.modules.dsv4 import _v41_decoder_trim as _geom
from rtp_llm.models_py.modules.dsv4.cp import CPContext

_CP = _geom._CP
_ANNOUNCED: set = set()
# R48d overhead gate probe: cumulative wall-clock us spent in the trim path
# (the per-chunk vote + the final-chunk compact + restore), printed once per
# request. The added Python/launch overhead must be < 15 ms (the brief's gate).
import time as _time

_OVERHEAD_US = [0.0]
_OVERHEAD_REQ = [None]


def _announce(tag: str, detail: str, *, once_key: Optional[str] = None) -> None:
    if once_key is not None:
        if once_key in _ANNOUNCED:
            return
        _ANNOUNCED.add(once_key)
    print(f"[{tag}] {detail}", file=sys.stderr, flush=True)


def _agree(values, device, group) -> bool:
    """Bounded control exchange: all ranks proceed or all reject together
    (byte-identical to _v41_ced._agree)."""
    local = torch.tensor([values], dtype=torch.int64, device=device)
    received = torch.empty((_CP, len(values)), dtype=torch.int64, device=device)
    torch.distributed.all_gather_into_tensor(received, local, group=group)
    rows = received.cpu().tolist()
    return rows[0][0] == 1 and all(row == rows[0] for row in rows[1:])


@dataclass(frozen=True)
class _TrimCPInfo:
    """Raw compact-tail metadata; never replaces the engine's
    PyContextParallelParams. Field set = every cp_info attribute consumed
    in-tree (mirrors _v41_ced._TailCPInfo)."""

    prefill_cp_padding_lengths: torch.Tensor
    prefill_cp_chunk_lengths: torch.Tensor
    prefill_shuffle_indices: torch.Tensor
    prefill_qkv_restore_indice: torch.Tensor
    prefill_qkv_padding_mask: torch.Tensor
    prefill_actual_input_lengths_cpu: torch.Tensor
    prefill_prefix_lengths_cpu: torch.Tensor
    prefill_mm_spans: torch.Tensor


def _trim_tail_context(
    original: CPContext, tail_start_global: int, tail: int
) -> CPContext:
    """Standalone CP context for the compact tail (mirrors _v41_ced._tail_context).
    ``tail`` = the compact length L (=144); local rows = L//4 (=36)."""
    local_rows = _geom.compact_local_rows(tail)
    device = original.global_positions.device
    rank = original.cp_rank
    relative = torch.tensor(
        _geom.zigzag_relative(rank, local_rows), dtype=torch.int64, device=device
    )
    restore = torch.tensor(
        _geom.zigzag_unpad_restore(local_rows, tail),
        dtype=torch.int64,
        device=device,
    )
    info = _TrimCPInfo(
        prefill_cp_padding_lengths=torch.zeros(1, dtype=torch.int32, device=device),
        prefill_cp_chunk_lengths=torch.tensor(
            [local_rows], dtype=torch.int32, device=device
        ),
        prefill_shuffle_indices=relative.to(torch.int32),
        prefill_qkv_restore_indice=restore.to(torch.int32),
        prefill_qkv_padding_mask=torch.ones(tail, dtype=torch.int32, device=device),
        prefill_actual_input_lengths_cpu=torch.tensor([tail], dtype=torch.int32),
        prefill_prefix_lengths_cpu=torch.tensor([tail_start_global], dtype=torch.int32),
        prefill_mm_spans=torch.empty((0, 3), dtype=torch.int32, device=device),
    )
    return replace(
        original,
        cp_size=_CP,
        cp_rank=rank,
        chunk_length=local_rows,
        padded_seq_len=tail,
        seq_len_full=tail,
        relative_positions=relative,
        prefix_length=tail_start_global,
        global_positions=(relative + tail_start_global).contiguous(),
        local_is_real=torch.ones(local_rows, dtype=torch.bool, device=device),
        unpad_restore=restore,
        seq_len_total=original.seq_len_total,
        cp_info=info,
        req_id_per_token=torch.zeros(local_rows, dtype=torch.int32, device=device),
        prefix_lengths=torch.tensor(
            [tail_start_global], dtype=torch.int64, device=device
        ),
        input_lengths_global=torch.tensor([tail], dtype=torch.int32, device=device),
        cu_seqlens_global=torch.tensor([0, tail], dtype=torch.int32, device=device),
        unpad_restore_is_prefix=False,
        chunk_lengths_per_req=(local_rows,),
        kv_cache_sharded=original.kv_cache_sharded,
        input_lengths_global_host=(tail,),
        prefix_lengths_host=(tail_start_global,),
        compact_geometry_verified=False,
    )


@dataclass
class DecoderTrimPlan:
    """One final-chunk transport plan. Never stored on a model/cache object.
    Mirrors _v41_ced.CEDChunkPlan; TAIL=144, no exactness halo (precision class)."""

    tail: int
    original_context: CPContext
    context: CPContext
    cu_seqlens: torch.Tensor
    _group: Any = field(repr=False, default=None)
    _restore_dst: Optional[torch.Tensor] = field(repr=False, default=None)
    _restore_src: Optional[torch.Tensor] = field(repr=False, default=None)
    _compacted: bool = field(default=False, init=False, repr=False)

    @property
    def original_rows(self) -> int:
        return self.original_context.chunk_length

    @property
    def local_rows(self) -> int:
        return _geom.compact_local_rows(self.tail)

    @property
    def compacted(self) -> bool:
        return self._compacted

    @property
    def indexer_projection(self):
        """Full-geometry BF16 head-weight GEMM (mirrors CED's: GEMM rows are
        independent, only cuBLAS's M-dependent algorithm choice matters, so
        zero-padding to the original per-rank M keeps the layout identical)."""
        rows = self.original_rows

        def project(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            full = x.new_zeros((rows, x.shape[-1]))
            full[: x.shape[0]] = x
            return F.linear(full, weights)[: x.shape[0]]

        return project

    def _require_all(self, valid: bool, *signature: int) -> None:
        if not _agree(
            [int(valid), *signature],
            self.original_context.global_positions.device,
            self._group,
        ):
            raise ValueError(
                "decoder-trim tail state/geometry disagrees across CP ranks"
            )

    def _gather_canonical(self, value: torch.Tensor) -> torch.Tensor:
        """[rows, ...] per rank -> canonical [n, ...] via the ORIGINAL ctx inverse
        map, then take the tail slice and this rank's compact zigzag share
        (byte-identical to _v41_ced._gather_canonical)."""
        ctx = self.original_context
        gathered = torch.empty(
            (ctx.padded_seq_len, *value.shape[1:]),
            dtype=value.dtype,
            device=value.device,
        )
        torch.distributed.all_gather_into_tensor(
            gathered, value.contiguous(), group=self._group
        )
        canonical = gathered.index_select(0, ctx.unpad_restore)
        tail = canonical[ctx.seq_len_full - self.tail : ctx.seq_len_full]
        return tail.index_select(0, self.context.relative_positions).contiguous()

    @torch.inference_mode()
    def compact(
        self, v4, hidden: torch.Tensor, input_ids: torch.Tensor, shared: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """After L20: gather+compact hidden/ids/pre-mix/topk/candidates to the
        tail. Mirrors CEDChunkPlan.compact; the caller installs context/meta/
        swa_start before L21. Raises consistently (via the vote) on any
        cross-rank mismatch — never diverges silently."""
        if self._compacted:
            raise ValueError("decoder-trim plan compacted twice")
        _t0 = _time.perf_counter()
        ctx = self.original_context
        device = ctx.global_positions.device
        rows = ctx.chunk_length
        pre_mix = getattr(getattr(v4.layers[0], "ffn_hc", None), "pre_mix_out", None)
        topk = shared.get("topk", {}).get(20)
        candidates = shared.get("candidates")
        sources = shared.get("global", {}).get(20)
        index_topk = int(getattr(v4.layers[0].attn, "index_topk", -1))
        source_ready = (
            isinstance(sources, (list, tuple))
            and len(sources) == 1
            and isinstance(sources[0], (list, tuple))
            and len(sources[0]) == 2
        )

        def _rows_tensor(value, dtypes) -> bool:
            return (
                torch.is_tensor(value)
                and value.ndim >= 1
                and value.shape[0] == rows
                and value.device == device
                and value.dtype in dtypes
            )

        valid = (
            source_ready
            and shared is v4.layers[0].attn._shared_attention
            and _rows_tensor(hidden, (torch.bfloat16,))
            and _rows_tensor(input_ids, (torch.int32, torch.int64))
            and (
                pre_mix is None
                or _rows_tensor(pre_mix, (torch.float32, torch.bfloat16))
            )
            and _rows_tensor(topk, (torch.int32,))
            and int(topk.shape[-1]) == index_topk
            and (candidates is None or _rows_tensor(candidates, (torch.int32,)))
        )
        if not valid:
            _announce(
                "DTRIMDBG",
                "compact-invalid: "
                f"source_ready={source_ready} "
                f"hidden={tuple(hidden.shape) if torch.is_tensor(hidden) else None} "
                f"ids={tuple(input_ids.shape) if torch.is_tensor(input_ids) else None} "
                f"pre_mix={tuple(pre_mix.shape) if torch.is_tensor(pre_mix) else None} "
                f"topk={tuple(topk.shape) if torch.is_tensor(topk) else None} "
                f"candidates={tuple(candidates.shape) if torch.is_tensor(candidates) else None}",
                once_key="dbg-compact",
            )
        ids_dtype = _DTYPES.index(input_ids.dtype) if input_ids.dtype in _DTYPES else -1
        signature = (
            ids_dtype,
            int(hidden.shape[1]) * 100000
            + (int(hidden.shape[2]) if hidden.ndim > 2 else 0),
            int(candidates.shape[1]) if torch.is_tensor(candidates) else -1,
            int(topk.shape[-1]) if torch.is_tensor(topk) else -1,
            (
                int(pre_mix.shape[1])
                if torch.is_tensor(pre_mix) and pre_mix.ndim > 1
                else -1
            ),
        )
        self._require_all(bool(valid), *signature)
        if not valid:
            raise ValueError("decoder-trim compact inputs invalid on this rank")

        compact_hidden = self._gather_canonical(hidden)
        compact_ids = self._gather_canonical(input_ids)
        compact_topk = self._gather_canonical(topk)
        if candidates is not None:
            shared["candidates"] = self._gather_canonical(candidates)
        if pre_mix is not None:
            # R59: publish the compacted premix through a STABLE per-unit
            # buffer (was: fresh gathered tensor every compacted forward).
            # The successor boundary probes key on the predecessor premix's
            # (shape, dtype, ptr); a recurring address lets them replay-hit,
            # while any shape/address change now safely recaptures/falls back
            # (address-safe keying) instead of reading stale rows.
            unit = v4.layers[0].ffn_hc
            gathered = self._gather_canonical(pre_mix)
            buf = getattr(unit, "_trim_compact_premix", None)
            if (
                buf is None
                or buf.shape != gathered.shape
                or buf.dtype != gathered.dtype
                or buf.device != gathered.device
            ):
                buf = torch.empty_like(gathered)
                unit._trim_compact_premix = buf
            buf.copy_(gathered)
            unit.pre_mix_out = buf
        shared["topk"] = {20: compact_topk}
        for key in _DERIVED_KEYS:
            shared.pop(key, None)
        self._compacted = True
        _OVERHEAD_US[0] += (_time.perf_counter() - _t0) * 1e6
        # R58c replay: publish this rank's compact-row -> canonical-tail index
        # map for the MoE/sparse routing override (no-op unless the replay env
        # is armed). Cleared in restore_rows at the end of this forward.
        from rtp_llm.models_py.modules.dsv4 import _v41_r58c_replay as _replay

        _replay.publish_row_index(
            torch.tensor(
                _geom.zigzag_relative(int(ctx.cp_rank), self.local_rows),
                dtype=torch.int64,
            )
        )
        _announce(
            "DTRIMTAIL",
            f"rows {rows}->{self.local_rows}/rank tail={self.tail} "
            f"start={self.context.prefix_length}",
            once_key=f"compact-{self.context.prefix_length}",
        )
        return compact_hidden, compact_ids

    @torch.inference_mode()
    def restore_rows(self, tensor: torch.Tensor) -> torch.Tensor:
        """After head reduction + norm on the compact rows: gather the canonical
        tail and redistribute each rank's zero-filled original layout. Rows
        outside the tail are zeros — discarded downstream (the engine reads only
        this chunk's last-row logits/hidden, which sit inside the tail).
        Mirrors CEDChunkPlan.restore_rows."""
        if not self._compacted:
            raise ValueError("decoder-trim restore before compact")
        _t0 = _time.perf_counter()
        device = tensor.device
        valid = (
            tensor.ndim == 2
            and tensor.shape[0] == self.local_rows
            and tensor.dtype in (torch.bfloat16, torch.float32)
        )
        width = int(tensor.shape[1]) if tensor.ndim == 2 else -1
        self._require_all(bool(valid), width, int(tensor.dtype == torch.bfloat16))
        if not valid:
            raise ValueError("decoder-trim restore input invalid on this rank")
        gathered = torch.empty((self.tail, width), dtype=tensor.dtype, device=device)
        torch.distributed.all_gather_into_tensor(
            gathered, tensor.contiguous(), group=self._group
        )
        canonical = gathered.index_select(0, self.context.unpad_restore)
        output = tensor.new_zeros((self.original_rows, width))
        if self._restore_dst is not None and self._restore_dst.numel() > 0:
            output.index_copy_(
                0, self._restore_dst, canonical.index_select(0, self._restore_src)
            )
        # R48d overhead gate probe: report the cumulative trim-path wall time.
        _OVERHEAD_US[0] += (_time.perf_counter() - _t0) * 1e6
        total_ms = _OVERHEAD_US[0] / 1e3
        _announce(
            "DTRIMOH",
            f"trim-path overhead ~{total_ms:.2f} ms (vote+compact+restore, "
            f"host wall time; gate < 15 ms)",
            once_key="dtrim-oh",
        )
        _OVERHEAD_US[0] = 0.0
        # R58c replay: the decoder layers are done; stop the routing override.
        from rtp_llm.models_py.modules.dsv4 import _v41_r58c_replay as _replay

        _replay.publish_row_index(None)
        return output


_DTYPES = (torch.bfloat16, torch.float32, torch.int32, torch.int64)
_DERIVED_KEYS = (
    "prefill_chunk_meta",
    "prefill_index_plan",
    "prefill_candidate_mask",
    "candidate_mask",
    "prefill_sparse_plans",
    "prefill_score_bounds",
    "prefill_meta_common",
)


def _env_gates_ok() -> bool:
    """Mirror the CED hard requirements: no persistent prefix reuse of any kind
    (stale skipped SWA slots must never be re-admitted)."""

    def _flag(name: str) -> bool:
        return os.environ.get(name, "0").strip().lower() in ("1", "true", "on", "yes")

    return (
        os.environ.get("REUSE_CACHE", "0") in ("0", "")
        and not _flag("ENABLE_MEMORY_CACHE")
        and not _flag("ENABLE_GPU_PREFIX_TREE")
    )


def _supported_model(v4) -> bool:
    """Stage-1 shape + per-layer source/window/ratio invariants (mirrors
    _v41_ced._supported_model re-scoped to the 20/20 stage: locals 0-19 == L20-39)."""
    args = getattr(v4, "args", None)
    layers = getattr(v4, "layers", ())
    ids = tuple(getattr(v4, "pp_global_layer_ids", ()))
    captures = tuple(getattr(v4, "capture_aux_hidden_layer_ids", ()))
    if (
        args is None
        or getattr(args, "v41_config", None) is None
        or not getattr(v4, "fp8_kv_cache", False)
        or getattr(v4, "norm", None) is None  # final stage only
        or getattr(v4, "_mtp_hidden_buffer", None) is not None
        or captures
        or ids != tuple(range(20, 40))
        or len(layers) != 20
    ):
        return False
    for local in range(20):
        index = 20 + local
        layer = layers[local]
        attn = getattr(layer, "attn", None)
        if (
            attn is None
            or getattr(attn, "layer_id", -1) != index
            or getattr(attn, "kv_source_layer_id", None) != 20
            or getattr(attn, "index_source_layer_id", None)
            != 20 + (index - 20) // 4 * 4
            or getattr(attn, "compress_ratio", 0) != 1
            or getattr(attn, "window_size", 0) != 128
            or bool(getattr(attn, "is_kv_source", False)) != (index == 20)
            or bool(getattr(attn, "is_index_source", False)) != (index % 4 == 0)
            or getattr(layer, "engram", None) is not None
        ):
            return False
    return True


def _evaluate_trim(
    v4,
    cp_ctx: Optional[CPContext],
    attn_inputs,
    *,
    keep: int,
    tail: int,
    mode: str,
    capture_aux: bool,
    cache_store_active: bool,
    after_layers_fn=None,
) -> Optional[Tuple[int, int]]:
    """Shared per-chunk engagement decision for BOTH decoder-trim arms
    (compact and mask). Returns (tail_start_global, tail) when this chunk
    engages, else None.

    Both arms call this with identical inputs (keep/tail differ only via the
    arm's own flag, and the harness drives them with the same KEEP target), so
    they engage on exactly the same chunks — the structural property the R58
    equivalence gate relies on. The decision is voted through a bounded int64
    all_gather after only rank-symmetric preconditions, so one rejecting rank
    rejects on all ranks and no rank can enter a collective alone.
    """
    if keep == 0 or tail == 0:
        return None
    # Rank-symmetric preconditions (identical by construction on every rank of
    # the stage; returning None here cannot diverge into a hung vote).
    if (
        not isinstance(cp_ctx, CPContext)
        or cp_ctx.global_positions is None
        or not torch.distributed.is_initialized()
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    group = collective_torch._get_group(Group.TP)
    if torch.distributed.get_world_size(group) != _CP:
        return None

    eligible = not (
        capture_aux
        or cache_store_active
        or after_layers_fn is not None
        or getattr(attn_inputs, "cache_store_inputs", None) is not None
        or not bool(getattr(attn_inputs, "is_prefill", False))
        or bool(getattr(attn_inputs, "is_target_verify", False))
        or bool(getattr(attn_inputs, "is_cuda_graph", False))
        or not _env_gates_ok()
        or not _supported_model(v4)
    )
    reject = None if eligible else "model-or-env-gate"
    n = int(cp_ctx.seq_len_full)
    rows = int(cp_ctx.chunk_length)
    padded = int(cp_ctx.padded_seq_len)
    prefix = int(cp_ctx.prefix_length)
    total = int(cp_ctx.seq_len_total)
    host_lengths = cp_ctx.input_lengths_global_host or ()
    if eligible:
        eligible = (
            cp_ctx.cp_size == _CP
            and len(host_lengths) == 1
            and padded == rows * _CP
            and n == padded
            and rows % 2 == 0
            and total == prefix + n
            # FINAL chunk only — the decoder trim compacts only the request's
            # last chunk (R48d); earlier chunks run the decoder layers full.
            and _geom.is_final_chunk(prefix, n, total)
            # R58b: reject the degenerate tail == n geometry. The trim exists
            # to SKIP decoder rows; tail == n skips nothing, and the compacted
            # ctx (prefix collapses to 0) leaves the SWA concat meta with an
            # unpopulated slot_in_flat -> index_copy_(0, None, ...) crashes the
            # engine (observed in the full-cluster repro). Reject cleanly instead of engaging.
            and tail < n
            and tail % (2 * _CP) == 0
            and (tail // _CP) % 2 == 0
            # NOTE: CED's byte-exactness guard (n-TAIL)%(rows//2)==0 is
            # deliberately RELAXED here — R48d is precision-class and the
            # all-gather + re-zigzag is tail-alignment-agnostic (the whole
            # 144-tail sits on rank 0's back half; trim_tail_source_map and
            # _gather_canonical handle that). This is the R48c blocker resolved.
        )
        if not eligible:
            reject = "geometry"
    device = cp_ctx.global_positions.device
    # Vote unconditionally (eligibility encoded as the first element): a
    # rank-divergent gate must reject everywhere, never hang.
    if not _agree([int(eligible), keep, tail, prefix, n, rows, total], device, group):
        if reject is None:
            reject = "vote-disagreement"
        _announce(
            "DTRIM",
            f"state=rejected reason={reject} mode={mode}",
            once_key=f"reject-{mode}-{reject}",
        )
        return None
    if not eligible:
        _announce(
            "DTRIM",
            f"state=rejected reason={reject} mode={mode}",
            once_key=f"reject-{mode}-{reject}",
        )
        return None
    return prefix + n - tail, tail


def plan_decoder_trim_mask(
    v4,
    cp_ctx: Optional[CPContext],
    attn_inputs,
    *,
    capture_aux: bool,
    cache_store_active: bool,
    after_layers_fn=None,
) -> Optional[int]:
    """R58 mask-mode reference arm (DSV41_DECODER_TRIM_MASK > 0). Same
    engagement decision as the compact arm (shared ``_evaluate_trim``), but
    NOTHING is compacted: the decoder layers run the full chunk and the SWA
    visibility is masked to positions >= the returned floor. Returns the floor
    (absolute position) for the forward loop to publish as
    ``dtrim_mask_swa_start`` after L20, else None.

    Floor choice: mask-matched (default) = the compact arm's tail start
    (prefix + n - tail); mask-exact (DSV41_DECODER_TRIM_MASK_EXACT=1) =
    prefix + n - keep (DeepInfer's exact keep-140 rectangle)."""
    keep = _geom.decoder_trim_mask_keep()
    tail = _geom.mask_tail()
    got = _evaluate_trim(
        v4,
        cp_ctx,
        attn_inputs,
        keep=keep,
        tail=tail,
        mode="mask",
        capture_aux=capture_aux,
        cache_store_active=cache_store_active,
        after_layers_fn=after_layers_fn,
    )
    if got is None:
        return None
    tail_start_global, _ = got
    exact = _geom.decoder_trim_mask_exact()
    floor = _geom.mask_floor_global(
        int(cp_ctx.prefix_length), int(cp_ctx.seq_len_full), tail, keep, exact
    )
    _announce(
        "DTRIMMASK",
        f"planned final-chunk mask floor={floor} tail_start={tail_start_global} "
        f"tail={tail} keep={keep} exact={int(exact)}",
        once_key=f"mask-{floor}",
    )
    return floor


def plan_decoder_trim_chunk(
    v4,
    cp_ctx: Optional[CPContext],
    attn_inputs,
    *,
    capture_aux: bool,
    cache_store_active: bool,
    after_layers_fn=None,
) -> Optional[DecoderTrimPlan]:
    """Per-chunk decision (called on every stage-1 rank before the layer loop).

    Returns a compacting plan ONLY for the request's final chunk, else None
    (full untouched path). The decision is voted through a bounded int64
    all_gather after only rank-symmetric preconditions, so one rejecting rank
    rejects on all ranks and no rank can enter a collective alone.
    """
    keep = _geom.decoder_trim_keep()
    tail = _geom.decoder_trim_tail()
    got = _evaluate_trim(
        v4,
        cp_ctx,
        attn_inputs,
        keep=keep,
        tail=tail,
        mode="compact",
        capture_aux=capture_aux,
        cache_store_active=cache_store_active,
        after_layers_fn=after_layers_fn,
    )
    if got is None:
        return None
    tail_start_global, tail = got
    # _evaluate_trim has voted the engagement; recompute the (rank-symmetric)
    # dims for the plan construction.
    n = int(cp_ctx.seq_len_full)
    rows = int(cp_ctx.chunk_length)
    prefix = int(cp_ctx.prefix_length)
    device = cp_ctx.global_positions.device
    group = collective_torch._get_group(Group.TP)
    context = _trim_tail_context(cp_ctx, tail_start_global, tail)
    dst, src = _geom.trim_tail_source_map(int(cp_ctx.cp_rank), rows, n, tail)
    plan = DecoderTrimPlan(
        tail=tail,
        original_context=cp_ctx,
        context=context,
        cu_seqlens=torch.tensor(
            [0, _geom.compact_local_rows(tail)], dtype=torch.int32, device=device
        ),
        _group=group,
        _restore_dst=torch.tensor(dst, dtype=torch.int64, device=device),
        _restore_src=torch.tensor(src, dtype=torch.int64, device=device),
    )
    _announce(
        "DTRIMPLAN",
        f"planned final-chunk prefix={prefix} rows={rows}->{plan.local_rows}/rank "
        f"start={tail_start_global} tail={tail} keep={keep}",
        once_key=f"plan-{prefix}",
    )
    return plan


__all__ = [
    "DecoderTrimPlan",
    "plan_decoder_trim_chunk",
    "plan_decoder_trim_mask",
]
