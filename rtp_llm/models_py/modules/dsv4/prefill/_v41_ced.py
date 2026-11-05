"""#19 CED — exact upper-layer tail compaction for chunked 20/20 V4.1 prefill.

Port of the gated single-pass experiment (a3808cf66, DeepInfer CED; the
checkpoint-aware follow-up 511b9e0a8 is C++ and is NOT needed here),
re-scoped for the adopted 20/20 PP cut and 4096x8 chunked 32K prefill.

Flag ``DSV41_CED`` (default off): ``1`` compacts EVERY stage-1 chunk; any
other non-zero value is rejected loudly (same contract as DSV41_PP_CUT).

Mechanism (stage 1 = L20-39, CP4 packed zigzag, 1024 rows/rank/chunk):
L20 always runs on ALL rows — it is the only upper-stage kv/index source
(kv_source_layer_ids=[2,8,14,20]) and its compact-CP producer fires during
its own pass. Afterwards the in-flight hidden/ids/pre-mix/topk/candidates
are compacted to the chunk's last TAIL_TOKENS=3072 tokens, the tail CP
context + prefill meta are rebuilt, and L21-39 run on 768 rows/rank.
Unlike the single-pass reference (whole tail on CP rank 0 -> broadcast),
the chunked zigzag spreads the tail over all four ranks (chunk pairs 2-7
-> ranks 2,3,3,2,1,0), so transport is an all_gather + index_select. The
output is restored to the original [rows, dim] layout, zero-filled outside
the tail (never exposed: the engine only reads the final chunk's last-row
logits/hidden — exactly what the parity sha hashes).

Exactness (why output AND effective cache state stay identical):
* L21-39 write only their own rolling SWA-window(128) pool entries
  (``_source_layer_cache``: source = self.layer_id for SWA_KV, else the
  kv_source layer). ``ced_swa_start`` disables prefix reads, so a chunk's
  exact region is self-contained: each of L21-39 eats a 127-row halo ->
  3072 - 19*127 = 659 exact L39 rows per chunk.
* The end-of-prefill SWA window (rows 32640-32767) lies inside the final
  chunk's 659-row exact region at EVERY layer, so the surviving pool state
  is written exactly; skipped/garbage rows occupy only slots the final
  window overwrites (rolling 128-window) and are never read again (decode
  reads the last-128 window only; all prefix reuse is hard-gated off).
* Global/index reads come from the CACHED L20 pools (complete for every
  row) and per-forward shared state; shared["topk"] is scratch, not cache.
* The final row (logits/hidden) is inside the exact region.
* Byte-exactness of the compaction itself: the indexer head-weight BF16
  GEMM keeps its original per-rank M via shared["ced_indexer_projection"]
  (cuBLAS M-dependent rounding can flip Top-K membership); FP8/FP4 GEMM
  rows are independent (reference-validated).

Wire/engine safety: stage 1 has after_layers_fn=None (frame consumer), so
no frame packing observes the compacted T; the per-chunk return keeps its
[rows, dim] shape; the (20,1) producer fires during L20 before compaction;
execution_scope exit drains unconsumed gathers. Skipping the L21-39
consumers cannot hang NCCL because all four CP ranks take the identical
voted decision (bounded int64 all_gather — the reference ``_agree``
pattern); any rejection falls back to the untouched full path on all
ranks. A "skip non-final chunks entirely" variant would additionally need
engine-side final-chunk exposure (C++ — out of scope per the R19 rule)
and saves ~0 TTFT anyway: the traces show stage 1 runs ~30-40 ms/chunk
behind stage 0's cadence, so only the serial final-chunk tail is on the
critical path. Python-only: no C++, no rebuild.
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
from rtp_llm.models_py.modules.dsv4.cp import CPContext

CED_FLAG = "DSV41_CED"

_CP_SIZE = 4
# DSV41_CED_TAIL is a DIAGNOSTIC knob (default 3072): TAIL=4096 combined
# with DSV41_CED_NO_SWA_OVERRIDE=1 makes the compaction an identity
# transform (same geometry, same GEMM M, same prefix reads) to bisect
# mechanism vs row-count-sensitive kernels. Only 3072 is the real
# exactness-margin configuration (659 exact L39 rows >= 128 window).
TAIL_TOKENS = int(os.environ.get("DSV41_CED_TAIL", "3072"))
_LOCAL_ROWS = TAIL_TOKENS // _CP_SIZE  # 768 compact rows per rank
_UPPER_LAYERS = 19  # L21..L39
_HALO_PER_LAYER = 127  # window(128) cascade: exact rows shrink by 127/layer
_WINDOW = 128
EXACT_TAIL_ROWS = TAIL_TOKENS - _UPPER_LAYERS * _HALO_PER_LAYER  # 659
assert EXACT_TAIL_ROWS >= _WINDOW, "tail must cover the final SWA window"
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

CED_STATS: Dict[str, int] = {
    "compacted_chunks": 0,
    "compacted_rows": 0,
    "skipped_layer_rows": 0,
}

_ANNOUNCED: set = set()


def ced_mode() -> int:
    """DSV41_CED: absent/0 = off, 1 = compact every chunk's upper layers to
    the trailing 3072 tokens. Any other value is rejected loudly."""
    value = os.environ.get(CED_FLAG, "0")
    if value not in ("0", "1"):
        raise ValueError("DSV41_CED must be 0 or 1")
    return int(value)


def _announce(tag: str, detail: str, *, once_key: Optional[str] = None) -> None:
    if once_key is not None:
        if once_key in _ANNOUNCED:
            return
        _ANNOUNCED.add(once_key)
    print(f"[{tag}] {detail}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Pure zigzag geometry (unit-tested on CPU; mirrors cp.py's ZigZagProcessor
# split and the reference _tail_context arithmetic).
# --------------------------------------------------------------------------


def zigzag_owner_local(p: int, half: int, cp: int = _CP_SIZE) -> Tuple[int, int]:
    """Canonical position -> (owner rank, rank-local row) for a zigzag split
    into 2*cp halves of ``half`` rows (front half pair p, back half mirrored)."""
    pair = p // half
    owner = pair if pair < cp else 2 * cp - 1 - pair
    local = p % half + (0 if pair < cp else half)
    return owner, local


def zigzag_relative(rank: int, local_rows: int, cp: int = _CP_SIZE) -> list:
    """Rank-local row idx -> canonical position (front half + mirrored back)."""
    half = local_rows // 2
    total = local_rows * cp
    return [rank * half + i for i in range(half)] + [
        total - (rank + 1) * half + i for i in range(half)
    ]


def zigzag_unpad_restore(local_rows: int, total: int, cp: int = _CP_SIZE) -> list:
    """Canonical position -> index into the rank-concatenated gather flat."""
    half = local_rows // 2
    restore = [0] * total
    for p in range(total):
        owner, local = zigzag_owner_local(p, half, cp)
        restore[p] = owner * local_rows + local
    return restore


def tail_source_map(
    rank: int, rows: int, n: int, tail: int, cp: int = _CP_SIZE
) -> Tuple[list, list]:
    """Restore path: this rank's ORIGINAL local rows that fall inside the
    chunk tail -> (dst local idx, src canonical tail idx)."""
    half = rows // 2
    start = n - tail
    dst, src = [], []
    for i in range(rows):
        pair = rank if i < half else 2 * cp - 1 - rank
        p = pair * half + i % half
        if p >= start:
            dst.append(i)
            src.append(p - start)
    return dst, src


@dataclass(frozen=True)
class _TailCPInfo:
    """Raw tail metadata; never replaces the engine's PyContextParallelParams.
    Field set = every cp_info attribute consumed in-tree (cp.py restore /
    padding mask, forward.py cpu lengths) — mirrors the reference."""

    prefill_cp_padding_lengths: torch.Tensor
    prefill_cp_chunk_lengths: torch.Tensor
    prefill_shuffle_indices: torch.Tensor
    prefill_qkv_restore_indice: torch.Tensor
    prefill_qkv_padding_mask: torch.Tensor
    prefill_actual_input_lengths_cpu: torch.Tensor
    prefill_prefix_lengths_cpu: torch.Tensor
    prefill_mm_spans: torch.Tensor


def _tensor(value, shape, device=None, dtypes=None) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and tuple(value.shape) == tuple(shape)
        and (device is None or value.device == device)
        and (dtypes is None or value.dtype in dtypes)
    )


def _agree(values, device, group) -> bool:
    """Bounded control exchange: all ranks proceed or all reject together."""
    local = torch.tensor([values], dtype=torch.int64, device=device)
    received = torch.empty((_CP_SIZE, len(values)), dtype=torch.int64, device=device)
    torch.distributed.all_gather_into_tensor(received, local, group=group)
    rows = received.cpu().tolist()
    return rows[0][0] == 1 and all(row == rows[0] for row in rows[1:])


def _supported_model(v4) -> bool:
    """Stage-1 shape + per-layer source/window/ratio invariants (reference
    _supported_model re-scoped to the 20/20 stage: locals 0-19 == L20-39)."""
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
            or getattr(attn, "window_size", 0) != _WINDOW
            or bool(getattr(attn, "is_kv_source", False)) != (index == 20)
            or bool(getattr(attn, "is_index_source", False)) != (index % 4 == 0)
            or getattr(layer, "engram", None) is not None
        ):
            return False
    return True


def _env_gates_ok() -> bool:
    """Mirror the reference hard requirements: no persistent prefix reuse of
    any kind (stale skipped SWA slots must never be re-admitted)."""

    def _flag(name: str) -> bool:
        return os.environ.get(name, "0").strip().lower() in ("1", "true", "on", "yes")

    return (
        os.environ.get("REUSE_CACHE", "0") in ("0", "")
        and not _flag("ENABLE_MEMORY_CACHE")
        and not _flag("ENABLE_GPU_PREFIX_TREE")
    )


def _tail_context(original: CPContext, tail_start_global: int) -> CPContext:
    """Explicitly replace every CP field for the compact tail; no original
    restore/position cache survives (replace() resets init=False fields)."""
    device = original.global_positions.device
    rank = original.cp_rank
    relative = torch.tensor(
        zigzag_relative(rank, _LOCAL_ROWS), dtype=torch.int64, device=device
    )
    restore = torch.tensor(
        zigzag_unpad_restore(_LOCAL_ROWS, TAIL_TOKENS),
        dtype=torch.int64,
        device=device,
    )
    info = _TailCPInfo(
        prefill_cp_padding_lengths=torch.zeros(1, dtype=torch.int32, device=device),
        prefill_cp_chunk_lengths=torch.tensor(
            [_LOCAL_ROWS], dtype=torch.int32, device=device
        ),
        prefill_shuffle_indices=relative.to(torch.int32),
        prefill_qkv_restore_indice=restore.to(torch.int32),
        prefill_qkv_padding_mask=torch.ones(
            TAIL_TOKENS, dtype=torch.int32, device=device
        ),
        prefill_actual_input_lengths_cpu=torch.tensor([TAIL_TOKENS], dtype=torch.int32),
        prefill_prefix_lengths_cpu=torch.tensor([tail_start_global], dtype=torch.int32),
        prefill_mm_spans=torch.empty((0, 3), dtype=torch.int32, device=device),
    )
    return replace(
        original,
        cp_size=_CP_SIZE,
        cp_rank=rank,
        chunk_length=_LOCAL_ROWS,
        padded_seq_len=TAIL_TOKENS,
        seq_len_full=TAIL_TOKENS,
        relative_positions=relative,
        prefix_length=tail_start_global,
        global_positions=(relative + tail_start_global).contiguous(),
        local_is_real=torch.ones(_LOCAL_ROWS, dtype=torch.bool, device=device),
        unpad_restore=restore,
        seq_len_total=original.seq_len_total,
        cp_info=info,
        req_id_per_token=torch.zeros(_LOCAL_ROWS, dtype=torch.int32, device=device),
        prefix_lengths=torch.tensor(
            [tail_start_global], dtype=torch.int64, device=device
        ),
        input_lengths_global=torch.tensor(
            [TAIL_TOKENS], dtype=torch.int32, device=device
        ),
        cu_seqlens_global=torch.tensor(
            [0, TAIL_TOKENS], dtype=torch.int32, device=device
        ),
        unpad_restore_is_prefix=False,
        chunk_lengths_per_req=(_LOCAL_ROWS,),
        kv_cache_sharded=original.kv_cache_sharded,
        input_lengths_global_host=(TAIL_TOKENS,),
        prefix_lengths_host=(tail_start_global,),
        compact_geometry_verified=False,
    )


@dataclass
class CEDChunkPlan:
    """One chunk's transport plan. Never stored on a model/cache object."""

    mode: int
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
    def compacted(self) -> bool:
        return self._compacted

    @property
    def indexer_projection(self):
        """Full-geometry BF16 head-weight GEMM (the reference
        ced_indexer_layout trick through our callable hook): GEMM rows are
        independent, only cuBLAS's M-dependent algorithm choice matters, so
        zero-padding to the original per-rank M keeps rounding identical."""
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
            raise ValueError("CED tail state/geometry disagrees across CP ranks")

    def _gather_canonical(self, value: torch.Tensor) -> torch.Tensor:
        """[rows, ...] per rank -> canonical [n, ...] via the ORIGINAL ctx
        inverse map (verified, not assumed), then take the tail slice and
        this rank's compact zigzag share."""
        ctx = self.original_context
        gathered = torch.empty(
            (ctx.padded_seq_len, *value.shape[1:]),
            dtype=value.dtype,
            device=value.device,
        )
        # Ordinary ephemeral allocation — never the persistent symmetric-memory
        # fast path (reference contract).
        torch.distributed.all_gather_into_tensor(
            gathered, value.contiguous(), group=self._group
        )
        canonical = gathered.index_select(0, ctx.unpad_restore)
        tail = canonical[ctx.seq_len_full - TAIL_TOKENS : ctx.seq_len_full]
        return tail.index_select(0, self.context.relative_positions).contiguous()

    @torch.inference_mode()
    def compact(
        self, v4, hidden: torch.Tensor, input_ids: torch.Tensor, shared: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """After L20: gather+compact hidden/ids/pre-mix/topk/candidates to
        the tail. The caller installs context/meta/ced_swa_start/
        ced_indexer_projection before L21. Raises consistently (via the
        vote) on any cross-rank mismatch — never diverges silently."""
        if self._compacted:
            raise ValueError("CED plan compacted twice")
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
            # candidates may legitimately be absent: the bitmap is allocated
            # only once history exceeds the topk candidate capacity (chunk 0
            # of a 32K prompt), and later layers allocate it fresh at the
            # compact row count when missing.
            and (candidates is None or _rows_tensor(candidates, (torch.int32,)))
        )
        if not valid:
            _announce(
                "CEDDBG",
                "compact-invalid: "
                f"source_ready={source_ready} "
                f"shared_ok={shared is v4.layers[0].attn._shared_attention} "
                f"hidden={tuple(hidden.shape) if torch.is_tensor(hidden) else None}"
                f"/{hidden.dtype if torch.is_tensor(hidden) else None} "
                f"ids={tuple(input_ids.shape) if torch.is_tensor(input_ids) else None}"
                f"/{input_ids.dtype if torch.is_tensor(input_ids) else None} "
                f"pre_mix={tuple(pre_mix.shape) if torch.is_tensor(pre_mix) else None} "
                f"topk={tuple(topk.shape) if torch.is_tensor(topk) else None} "
                f"index_topk={index_topk} "
                f"candidates={tuple(candidates.shape) if torch.is_tensor(candidates) else None}",
                once_key="dbg-compact",
            )
        # Vote the exact payload geometry so any cross-rank divergence fails
        # consistently instead of corrupting the gather.
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
            raise ValueError("CED compact inputs invalid on this rank")

        compact_hidden = self._gather_canonical(hidden)
        compact_ids = self._gather_canonical(input_ids)
        compact_topk = self._gather_canonical(topk)
        if candidates is not None:
            shared["candidates"] = self._gather_canonical(candidates)
        if pre_mix is not None:
            v4.layers[0].ffn_hc.pre_mix_out = self._gather_canonical(pre_mix)
        shared["topk"] = {20: compact_topk}
        for key in _DERIVED_KEYS:
            shared.pop(key, None)
        self._compacted = True
        CED_STATS["compacted_chunks"] += 1
        CED_STATS["compacted_rows"] += _LOCAL_ROWS
        CED_STATS["skipped_layer_rows"] += (rows - _LOCAL_ROWS) * _UPPER_LAYERS
        _announce(
            "CEDTAIL",
            f"rows {rows}->{_LOCAL_ROWS}/rank tail={TAIL_TOKENS} "
            f"start={self.context.prefix_length} exact_l39_rows={EXACT_TAIL_ROWS} "
            f"stats={CED_STATS}",
        )
        return compact_hidden, compact_ids

    @torch.inference_mode()
    def restore_rows(self, tensor: torch.Tensor) -> torch.Tensor:
        """After head reduction + norm on the compact rows: gather the
        canonical tail and redistribute each rank's zero-filled original
        layout. Rows outside the tail are zeros — discarded downstream (the
        final-row logits/hidden sit inside the exact 659)."""
        if not self._compacted:
            raise ValueError("CED restore before compact")
        device = tensor.device
        valid = (
            tensor.ndim == 2
            and tensor.shape[0] == _LOCAL_ROWS
            and tensor.dtype in (torch.bfloat16, torch.float32)
        )
        width = int(tensor.shape[1]) if tensor.ndim == 2 else -1
        self._require_all(bool(valid), width, int(tensor.dtype == torch.bfloat16))
        if not valid:
            _announce(
                "CEDDBG",
                f"restore-invalid: shape={tuple(tensor.shape)} dtype={tensor.dtype}",
                once_key="dbg-restore",
            )
            raise ValueError("CED restore input invalid on this rank")
        gathered = torch.empty((TAIL_TOKENS, width), dtype=tensor.dtype, device=device)
        torch.distributed.all_gather_into_tensor(
            gathered, tensor.contiguous(), group=self._group
        )
        canonical = gathered.index_select(0, self.context.unpad_restore)
        output = tensor.new_zeros((self.original_rows, width))
        if self._restore_dst is not None and self._restore_dst.numel() > 0:
            output.index_copy_(
                0, self._restore_dst, canonical.index_select(0, self._restore_src)
            )
        return output


def _geometry_verified(cp_ctx: CPContext) -> bool:
    """Check the ACTUAL inverse maps, not the assumed zigzag arithmetic
    (reference _original_tail_is_rank_zero philosophy): reject if the CP
    layout differs from the packed zigzag this port is derived for."""
    rows = int(cp_ctx.chunk_length)
    n = int(cp_ctx.seq_len_full)
    device = cp_ctx.global_positions.device
    expected_restore = torch.tensor(
        zigzag_unpad_restore(rows, n), dtype=torch.int64, device=device
    )
    expected_relative = torch.tensor(
        zigzag_relative(int(cp_ctx.cp_rank), rows), dtype=torch.int64, device=device
    )
    checks = [
        (cp_ctx.unpad_restore.to(device) == expected_restore).all(),
        (cp_ctx.relative_positions.to(device) == expected_relative).all(),
        (
            cp_ctx.global_positions.to(device)
            == expected_relative + int(cp_ctx.prefix_length)
        ).all(),
        cp_ctx.local_is_real.all(),
    ]
    return bool(torch.stack(checks).all().item())


def plan_ced_chunk(
    v4,
    cp_ctx: Optional[CPContext],
    attn_inputs,
    *,
    capture_aux: bool,
    cache_store_active: bool,
    after_layers_fn=None,
) -> Optional[CEDChunkPlan]:
    """Per-chunk decision (called on every stage rank before the layer loop).

    Returns a compacting plan or None (full untouched path). The decision is
    voted through a bounded int64 all_gather AFTER only rank-symmetric
    preconditions, so one rejecting rank rejects on all ranks and no rank
    can enter a collective alone.
    """
    mode = ced_mode()
    if mode == 0:
        return None
    # Rank-symmetric preconditions (identical by construction on every rank
    # of the stage; returning None here cannot diverge into a hung vote).
    if (
        not isinstance(cp_ctx, CPContext)
        or cp_ctx.global_positions is None
        or not torch.distributed.is_initialized()
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    group = collective_torch._get_group(Group.TP)
    if torch.distributed.get_world_size(group) != _CP_SIZE:
        return None

    from rtp_llm.models_py.modules.dsv4.bounded_replay import (
        enabled as bounded_replay_enabled,
    )
    from rtp_llm.models_py.modules.dsv4.moe.forward_ep_plan import (
        enabled as _ep_count_plan_enabled,
    )

    eligible = not (
        capture_aux
        or cache_store_active
        or after_layers_fn is not None
        or getattr(attn_inputs, "cache_store_inputs", None) is not None
        or not bool(getattr(attn_inputs, "is_prefill", False))
        or bool(getattr(attn_inputs, "is_target_verify", False))
        or bool(getattr(attn_inputs, "is_cuda_graph", False))
        or bounded_replay_enabled()
        or _ep_count_plan_enabled()
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
            cp_ctx.cp_size == _CP_SIZE
            # NOTE: no kv_cache_sharded requirement — this deployment runs
            # the UNSHARDED CP cache (CacheConfigCreator: "V4.1 exact cache
            # requires unsharded CP cache"), unlike the reference's sharded
            # setup. The port is sharding-agnostic: ced_swa_start disables
            # prefix pool reads either way, current-chunk globals arrive via
            # shared["global"][20], and fresh-row SWA is CP-gathered.
            and len(host_lengths) == 1
            and padded == rows * _CP_SIZE
            and n == padded
            and rows % 2 == 0
            and total == prefix + n
            and n >= TAIL_TOKENS
            and (n - TAIL_TOKENS) % (rows // 2) == 0
        )
        if not eligible:
            reject = "geometry"
    device = cp_ctx.global_positions.device
    if eligible and not _geometry_verified(cp_ctx):
        eligible = False
        reject = "cp-layout"
    # Vote unconditionally (eligibility encoded as the first element): a
    # rank-divergent gate must reject everywhere, never hang.
    if not _agree([int(eligible), mode, prefix, n, rows, total], device, group):
        if reject is None:
            reject = "vote-disagreement"
        _announce("CED", f"state=rejected reason={reject}", once_key=f"reject-{reject}")
        return None
    if not eligible:
        _announce("CED", f"state=rejected reason={reject}", once_key=f"reject-{reject}")
        return None
    _announce(
        "CED",
        f"mode={mode} tail={TAIL_TOKENS} exact_l39_rows={EXACT_TAIL_ROWS} "
        f"pid={os.getpid()}",
        once_key="mode",
    )
    tail_start_global = prefix + n - TAIL_TOKENS
    context = _tail_context(cp_ctx, tail_start_global)
    dst, src = tail_source_map(int(cp_ctx.cp_rank), rows, n, TAIL_TOKENS)
    plan = CEDChunkPlan(
        mode=mode,
        original_context=cp_ctx,
        context=context,
        cu_seqlens=torch.tensor([0, _LOCAL_ROWS], dtype=torch.int32, device=device),
        _group=group,
        _restore_dst=torch.tensor(dst, dtype=torch.int64, device=device),
        _restore_src=torch.tensor(src, dtype=torch.int64, device=device),
    )
    _announce(
        "CEDPLAN",
        f"planned prefix={prefix} rows={rows}->{_LOCAL_ROWS}/rank "
        f"start={tail_start_global}",
        once_key=f"plan-{prefix}",
    )
    return plan


__all__ = [
    "CEDChunkPlan",
    "CED_FLAG",
    "CED_STATS",
    "EXACT_TAIL_ROWS",
    "TAIL_TOKENS",
    "ced_mode",
    "plan_ced_chunk",
    "tail_source_map",
    "zigzag_owner_local",
    "zigzag_relative",
    "zigzag_unpad_restore",
]
