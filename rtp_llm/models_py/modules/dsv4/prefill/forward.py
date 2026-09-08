"""DSV4 prefill forward helpers — extracted from ``DeepSeekV4Model``.

Exposes qwen3-style prefill primitives as free functions so the Model
class stays thin:

* ``set_cp_info``                    — bind/clear Context-Parallel metadata on ``v4``
* ``forward_layers``                 — per-layer loop body (embed → layers → reduce → norm)
* unified cache-store registration via
  :func:`rtp_llm.models_py.modules.factory.attention.common.create_write_cache_store_impl`
* ``forward_prefill``                — full prefill arm (per-request loop over flat 1D input_ids)

Generic KV-cache tag constants and lookup helpers
(``build_block_tables_batched``) live in
:mod:`rtp_llm.models_py.modules.dsv4.kv_cache_utils`.

Nothing in here holds state. ``DeepSeekV4Model.forward`` feeds in
``self.v4`` / ``self.kv_cache`` / ``self.parallelism_config`` explicitly.

Paired with :mod:`rtp_llm.models_py.modules.dsv4.decode.forward`, which
does the same job for the decode path.

----------------------------------------------------------------------
Context-Parallel (CP) prefill data flow
----------------------------------------------------------------------

CP repurposes the TP process group as the CP group (see
``ParallelismConfig::get_attn_tp_size`` — returns 1 when CP enabled).
The C++ ``ZigZagProcessor`` splits each request's padded prefill tokens
across the CP group with a zigzag layout. ``forward_layers`` consumes
the resulting per-rank metadata and builds a ``CPContext`` (in
``cp.py``) bound onto every Attention / Compressor / Indexer module
via ``v4._propagate_cp_ctx`` before the per-layer loop runs.

Per-rank inputs (rank-local, shaped for ``T_local = chunk_length``):
  * ``input_ids``                — token slice owned by this rank
  * ``attn.combo_position_ids``  — framework-provided positions; under CP,
                                   ``forward_layers`` replaces these with
                                   CPContext's per-token request-absolute
                                   positions
  * ``attn.cu_seqlens``          — rank-local request boundaries
  * ``attn.input_lengths``       — rank-local per-req token count

Rank-invariant inputs:
  * ``attn.prefix_lengths``      — global per-req KV prefix length

Global view (held on ``CPContext``, derived once in
``build_cp_context``):
  * ``cp_ctx.input_lengths_global`` — full per-req length, =
    ``cp_info.prefill_actual_input_lengths_cpu``
  * ``cp_ctx.cu_seqlens_global``    — cumsum, used as ``query_start_loc``
    for SWA-pool write meta
  * ``cp_ctx.global_positions``     — GLOBAL absolute pos per rank-local
    token (zigzag-derived, per-request for B>=1)
  * ``cp_ctx.seq_len_full``         — total real prefill length

Per-layer pipeline under CP (compress_ratio == 0, SWA-only):
  1. ``_prefill_compute_qkv``: rank-local Q + KV → KV all-gathered to
     ``kv_full[seq_len_full, D]`` in GLOBAL request order
  2. ``_prefill_write_swa_fp8_paged``: every rank writes the GATHER'd
     KV to its own paged pool. ``slot_mapping`` is built from the
     global write trio (cu_seqlens_global / combined_seq_lens_global /
     seq_len_full) so all ranks' pools end up bit-identical.
  3. ``_attn_fp8_swa_via_kv_full`` (fresh, sp==0): rank-local Q over
     gathered KV. The varlen topk builder uses CP global positions plus
     global per-request cu_seqlens, so topk indices address rows in
     ``kv_full`` for B>=1.
  4. ``_attn_fp8_swa_via_concat`` (cont, sp>0): workspace ``[B, M, D]``
     with per-request prefix tails and new-K slots. ``combined_indices`` /
     ``combined_lens`` are built in Python because the Triton helper's
     ``pos = start_pos + token_idx_in_query`` formula assumes contiguous Q,
     which zigzag CP breaks.

CSA / HCA layers (compress_ratio == 4 / 128) add:
  * ``CompressorFP8.forward`` all-gathers KV/score then drops the
    rank-local ``meta`` and rebuilds ``state_slots`` / ``kv_slots`` from
    CPContext's full per-request positions.
  * ``IndexerFP8.prepare`` swaps ``input_lengths`` →
    ``cp_ctx.input_lengths_global`` for ``T_per_req`` so ks / ke /
    cu_kv_seqlens index into the per-rank pool's GLOBAL compressed-K
    extent. Nested compressor_meta is nulled for the same rebuild path.

Output: each layer's hidden state ``h`` is rank-local
``[T_local, hc, dim]`` — the framework's exit all-gather + strip-pad
gather (driven by ``cp_info.prefill_qkv_restore_indice`` /
``prefill_qkv_padding_mask``) reassembles the full sequence for the
next-layer / lm-head step.

Decode does NOT all-gather. Each rank's pool already holds the full
sequence's compressed entries (each rank wrote the gather'd new K
during prefill), so per-rank decode reads remain self-contained.

Padding-token slots are nulled via ``cp_info.prefill_qkv_padding_mask``.
"""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import torch

from rtp_llm.models_py.modules.dsv4 import _forward_tensor_debug as _fwd_dbg
from rtp_llm.models_py.modules.dsv4 import _profiler
from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt
from rtp_llm.models_py.modules.dsv4.cp import (
    build_cp_context_for_forward,
    cp_gather_last_by_request,
)
from rtp_llm.models_py.modules.dsv4.fp8.prefill_meta import (
    build_and_propagate_prefill_meta_fp8,
    clear_prefill_meta_shared_fp8,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    build_block_tables_batched,
    primary_attention_inputs,
)
from rtp_llm.models_py.modules.dsv4.prefill_workspace import (
    PrefillWorkspace,
    resolve_prefill_workspace_rows,
)
from rtp_llm.models_py.modules.factory.attention.common import (
    create_write_cache_store_impl,
)
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)

if TYPE_CHECKING:
    # Kept behind TYPE_CHECKING to avoid an import cycle — ``transformer``
    # doesn't depend on ``prefill`` today but this guard makes that
    # non-load-bearing (module loads fine even if the cycle reappears).
    from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer


_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_PREFILL_FAST_LAYER_CALLS_ATTR = "_dsv4_prefill_fast_layer_calls"

_PrefillFastLayerCall = Callable[..., torch.Tensor]


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUE_ENV_VALUES


# ---------------------------------------------------------------------------
# Env-gated per-forward CPU-wall accounting (``DSV4_FWD_STATS=1``).
#
# Step 15's ``t_stage(C) = 799 + C*26.0`` fit leaves ~15 ms/round unexplained:
# the timed CP gathers are only ~7-10 ms/round and the removed D2H syncs ~0.6.
# The remainder has to be host-side work the layer loop cannot overlap — D2H
# ``.item()`` syncs, the per-forward ``PrefillWorkspace`` union allocation, and
# the meta build. All three are CPU costs, so they show up as CPU wall time, and
# a real cudaMalloc/cudaFree shows up in the allocator counters. Neither needs a
# profiler or a rebuild. Free when the flag is off (one bool read per forward).
# ---------------------------------------------------------------------------
_FWD_STATS = _env_flag("DSV4_FWD_STATS")
_FWD_STATS_MAX = int(os.environ.get("DSV4_FWD_STATS_MAX", "0") or 0)
_FWD_STATS_ROWS: list = []
_FWD_STATS_N = [0]

# GPU wall time per forward, bracketed by a CUDA event pair and drained lazily
# with the non-blocking Event.query() on later forwards (the same trick the CP
# gather stats use) so measuring does not serialise the thing being measured.
#
# Needed because pH2/pH3/pH4 showed that removing BOTH per-round host barriers in
# PPExecutor changes nothing, which leaves one question CPU wall cannot answer and
# CUPTI perturbs: is the ~126 ms/round stage cost inside forward_layers, or
# outside it in the PP activation transfer (67.1 MB/round) and tpSync?
_FWD_GPU = _env_flag("DSV4_FWD_GPU")
if _FWD_GPU:
    _FWD_STATS = True
_FWD_GPU_PENDING: list = []
_FWD_GPU_ROWS: list = []

# Allocator counters that only move on a real driver call. A per-forward union
# buffer that the caching allocator recycles leaves all three at zero; one that
# does not is paying cudaMalloc/cudaFree (a cudaFree is a device sync).
_FWD_STATS_MEM_KEYS = (
    "num_device_alloc",
    "num_device_free",
    "num_alloc_retries",
)


def _fwd_stats_snap() -> Optional[tuple]:
    if not _FWD_STATS:
        return None
    ms = torch.cuda.memory_stats()
    return tuple(int(ms.get(k, 0)) for k in _FWD_STATS_MEM_KEYS)


def _fwd_gpu_drain(force: bool = False) -> None:
    """Pop completed event pairs and report their GPU wall time.

    ``Event.query()`` is non-blocking, so this never waits on the GPU; pairs
    still in flight stay queued for a later forward. ``force`` synchronises the
    tail event first and is only for the atexit flush.
    """
    while _FWD_GPU_PENDING:
        n_tokens, cp_size, _ev0, ev1 = _FWD_GPU_PENDING[0]
        if force:
            ev1.synchronize()
        elif not ev1.query():
            return
        _FWD_GPU_PENDING.pop(0)
        gpu_ms = None
        try:
            gpu_ms = float(_ev0.elapsed_time(ev1))
        except RuntimeError:
            gpu_ms = None
        if gpu_ms is None:
            continue
        _FWD_GPU_ROWS.append((n_tokens, cp_size, gpu_ms))
        import sys

        print(
            "[FWDTG] rank=%d T=%d cp=%d gpu_ms=%.2f"
            % (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else -1,
                n_tokens,
                cp_size,
                gpu_ms,
            ),
            file=sys.stderr,
            flush=True,
        )


def _fwd_stats_report_row(
    *,
    n_tokens: int,
    cp_size: int,
    marks: Dict[str, float],
    mem_before: Optional[tuple],
    mem_after: Optional[tuple],
    ev0=None,
) -> None:
    """Emit one ``[FWDT]`` line of per-phase CPU wall times (ms)."""
    _FWD_STATS_N[0] += 1
    if _FWD_STATS_MAX and _FWD_STATS_N[0] > _FWD_STATS_MAX:
        return
    if _FWD_GPU and ev0 is not None:
        ev1 = torch.cuda.Event(enable_timing=True)
        ev1.record()
        _FWD_GPU_PENDING.append((n_tokens, cp_size, ev0, ev1))
        _fwd_gpu_drain()
    order = ("cpctx", "pos", "embed", "meta", "loop", "tail")
    parts = []
    prev = marks.get("entry")
    for name in order:
        cur = marks.get(name)
        if prev is None or cur is None:
            parts.append("%s=NA" % name)
            prev = cur if cur is not None else prev
            continue
        parts.append("%s=%.2f" % (name, (cur - prev) * 1e3))
        prev = cur
    total = 0.0
    if marks.get("entry") is not None and marks.get("tail") is not None:
        total = (marks["tail"] - marks["entry"]) * 1e3
    deltas = ""
    if mem_before is not None and mem_after is not None:
        deltas = " " + " ".join(
            "d_%s=%d" % (k, a - b)
            for k, b, a in zip(_FWD_STATS_MEM_KEYS, mem_before, mem_after)
        )
    import sys

    print(
        "[FWDT] rank=%d n=%d T=%d cp=%d total=%.2f %s%s"
        % (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else -1,
            _FWD_STATS_N[0],
            n_tokens,
            cp_size,
            total,
            " ".join(parts),
            deltas,
        ),
        file=sys.stderr,
        flush=True,
    )
    _FWD_STATS_ROWS.append((n_tokens, cp_size, total, tuple(parts)))


def _fwd_stats_flush() -> None:
    if _FWD_GPU:
        _fwd_gpu_drain(force=True)
    if not _FWD_STATS or not _FWD_STATS_ROWS:
        return
    import sys

    by_shape: Dict[tuple, list] = {}
    for n_tokens, cp_size, total, _parts in _FWD_STATS_ROWS:
        by_shape.setdefault((n_tokens, cp_size), []).append(total)
    print("[FWDT] ---- summary over %d forwards ----" % len(_FWD_STATS_ROWS),
          file=sys.stderr, flush=True)
    for (n_tokens, cp_size), totals in sorted(by_shape.items()):
        print(
            "[FWDT] T=%d cp=%d calls=%d mean_total=%.2f ms sum_total=%.1f ms"
            % (
                n_tokens,
                cp_size,
                len(totals),
                sum(totals) / len(totals),
                sum(totals),
            ),
            file=sys.stderr,
            flush=True,
        )
    if _FWD_GPU_ROWS:
        gpu_by_shape: Dict[tuple, list] = {}
        for n_tokens, cp_size, gpu_ms in _FWD_GPU_ROWS:
            gpu_by_shape.setdefault((n_tokens, cp_size), []).append(gpu_ms)
        for (n_tokens, cp_size), vals in sorted(gpu_by_shape.items()):
            print(
                "[FWDTG] T=%d cp=%d calls=%d mean_gpu=%.2f ms sum_gpu=%.1f ms"
                % (n_tokens, cp_size, len(vals), sum(vals) / len(vals), sum(vals)),
                file=sys.stderr,
                flush=True,
            )


if _FWD_STATS:
    import atexit

    atexit.register(_fwd_stats_flush)


# ---------------------------------------------------------------------------
# Env-gated single-forward GPU kernel breakdown (``DSV4_FWD_PROFILE=1``).
#
# pH0's CPU-wall accounting put a whole forward at ~65 ms of host time while the
# fitted stage cost is ~126 ms per round, so the per-round term ``o`` is GPU-side
# and host timings cannot localise it further. This captures ONE forward per rank
# with CUPTI and prints self-CUDA time per kernel — one profiled forward inflates
# that request's TTFT, so a profiling leg's mean is not a result.
# ---------------------------------------------------------------------------
_FWD_PROFILE = _env_flag("DSV4_FWD_PROFILE")
_FWD_PROFILE_IDX = int(os.environ.get("DSV4_FWD_PROFILE_IDX", "20") or 20)
_FWD_PROFILE_ROWS = int(os.environ.get("DSV4_FWD_PROFILE_ROWS", "35") or 35)
# -1 = every rank profiles. CUPTI on all 8 ranks at once perturbs the pipeline
# and the CP collectives, so measurement legs normally pin one world rank.
_FWD_PROFILE_RANK = int(os.environ.get("DSV4_FWD_PROFILE_RANK", "-1") or -1)
_FWD_PROFILE_CT = [0]


def _fwd_profile_rank_ok() -> bool:
    if _FWD_PROFILE_RANK < 0:
        return True
    if not torch.distributed.is_initialized():
        return _FWD_PROFILE_RANK == 0
    return torch.distributed.get_rank() == _FWD_PROFILE_RANK


def _fwd_profile_start():
    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    )
    prof.__enter__()
    return prof


def _fwd_profile_dump(prof) -> None:
    import sys

    rank = (
        torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    )
    ka = prof.key_averages()
    total_us = sum(float(e.self_device_time_total) for e in ka)
    print(
        "[FWDP] rank=%d ---- one forward: total self GPU %.2f ms ----"
        % (rank, total_us / 1e3),
        file=sys.stderr,
        flush=True,
    )
    print(
        ka.table(
            sort_by="self_device_time_total",
            row_limit=_FWD_PROFILE_ROWS,
            max_name_column_width=78,
        ),
        file=sys.stderr,
        flush=True,
    )


def _prefill_fast_path_layer_calls(
    v4: "V4Transformer",
) -> Optional[Tuple[_PrefillFastLayerCall, ...]]:
    if hasattr(v4, _PREFILL_FAST_LAYER_CALLS_ATTR):
        return getattr(v4, _PREFILL_FAST_LAYER_CALLS_ATTR)

    layer_calls: Optional[Tuple[_PrefillFastLayerCall, ...]] = None
    layers = getattr(v4, "layers", None)
    if getattr(v4, "fp8_kv_cache", False) and layers:
        from rtp_llm.models_py.modules.dsv4.block import Block
        from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8

        calls = []
        for layer in layers:
            if not isinstance(layer, Block):
                calls = []
                break
            if not isinstance(getattr(layer, "attn", None), AttentionFP8):
                calls = []
                break
            fast_callable_fn = getattr(layer, "prefill_fast_callable", None)
            if fast_callable_fn is None:
                calls = []
                break
            fast_call = fast_callable_fn()
            if fast_call is None:
                calls = []
                break
            calls.append(fast_call)
        if calls:
            layer_calls = tuple(calls)

    # The layer stack is static after model construction. Cache both the
    # supported and unsupported outcomes so the hot path does not rescan every
    # Block. If a future maintainer mutates ``v4.layers`` at runtime, this cache
    # must be invalidated together with that mutation.
    setattr(v4, _PREFILL_FAST_LAYER_CALLS_ATTR, layer_calls)
    return layer_calls


def _prefill_fast_path_enabled(
    v4: "V4Transformer",
    prepare_hidden_fn: Optional[Any],
    layer_calls: Optional[Tuple[_PrefillFastLayerCall, ...]] = None,
) -> bool:
    # Default-on production fast path. The normal path remains the source of
    # truth for debug/recording, BF16, custom-hidden, and unsupported modules.
    if not _env_flag("DSV4_PREFILL_FAST_PATH", "1"):
        return False
    if prepare_hidden_fn is not None:
        return False
    if _rt.ENABLED or _fwd_dbg.enabled():
        return False
    if layer_calls is None:
        layer_calls = _prefill_fast_path_layer_calls(v4)
    return layer_calls is not None


def _build_positions_from_lengths(
    input_lengths: torch.Tensor,  # [B] int
    prefix_lengths: torch.Tensor,  # [B] int
    device: torch.device,
    total_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Synthesize per-token global positions ``[T_total]`` int64 when the
    framework didn't populate ``attn.combo_position_ids`` (warmup / cudagraph
    capture path).

    For each request ``b`` with prefix ``sp[b]`` and input length ``L[b]``,
    emit ``sp[b], sp[b]+1, ..., sp[b]+L[b]-1``; concatenated across the batch.

    Must be CUDA-graph-capture-safe: callers pass GPU-resident tensors
    (``input_lengths`` / ``prefix_lengths``) during capture. Keep the
    body tensor-only so capture does not synchronize on scalar reads.
    """
    input_lengths = input_lengths.to(device=device, dtype=torch.int64)
    prefix_lengths = prefix_lengths.to(device=device, dtype=torch.int64)
    batch_size = int(input_lengths.numel())
    if total_tokens is None:
        total_tokens = int(input_lengths.sum().item())
    if batch_size == 0 or total_tokens == 0:
        return torch.zeros(0, dtype=torch.int64, device=device)

    token_offsets = torch.arange(total_tokens, dtype=torch.int64, device=device)
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=device),
            input_lengths.cumsum(0),
        ],
        dim=0,
    )
    req_ids = torch.searchsorted(cu_seqlens[1:], token_offsets, right=True)
    req_ids = req_ids.clamp(max=batch_size - 1)
    local_offsets = token_offsets - cu_seqlens.gather(0, req_ids)
    return prefix_lengths.gather(0, req_ids) + local_offsets


def _last_hidden_by_request(
    flat: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
    cp_ctx: Optional[Any],
) -> torch.Tensor:
    if cp_ctx is not None and cp_ctx.cp_size > 1:
        return cp_gather_last_by_request(flat, cp_ctx)
    if cu_seqlens is not None and cu_seqlens.numel() >= 2:
        last_indices = cu_seqlens[1:].to(device=flat.device, dtype=torch.long) - 1
        return flat.index_select(0, last_indices).contiguous()
    return flat[-1:].contiguous()


def set_cp_info(
    v4: V4Transformer,
    parallelism_config: Optional[ParallelismConfig],
    attn: Optional[PyAttentionInputs],
    is_prefill: bool,
) -> None:
    """Stash per-forward Context-Parallel metadata on ``v4`` so
    :func:`forward_layers` can build + propagate the derived
    ``CPContext`` when it enters the per-layer loop.

    Clears with ``(None, 1, 0)`` when CP is off so no stale ctx leaks
    from a prior request (warmup, etc.).
    """
    cp_enabled = (
        parallelism_config is not None
        and getattr(parallelism_config, "prefill_cp_config", None) is not None
        # is_enabled() deliberately EXCLUDES CPRotateMethod.PREFILL_CP, which has
        # its own is_prefill_enabled() predicate. Gating on is_enabled() alone
        # means DSV4 can never build a CPContext in PREFILL_CP mode: the C++
        # ContextParallelProcessor splits the batch to rank-local tokens while
        # this side still assumes the full sequence, which trips a device-side
        # assert in the SWA Triton kernel. The python config layer already treats
        # the two as alternatives (backend_rpc_server_visitor.py:131).
        and (
            parallelism_config.prefill_cp_config.is_enabled()
            or parallelism_config.prefill_cp_config.is_prefill_enabled()
        )
        and is_prefill
        and attn is not None
        and getattr(attn, "context_parallel_info", None) is not None
    )
    if cp_enabled:
        v4.set_cp_info(
            cp_info=attn.context_parallel_info,
            cp_size=int(parallelism_config.tp_size),
            cp_rank=int(parallelism_config.tp_rank),
            kv_cache_sharded=bool(
                getattr(parallelism_config.prefill_cp_config, "kv_cache_sharded", False)
            ),
        )
    else:
        v4.set_cp_info(None, 1, 0)


def forward_layers(
    v4: V4Transformer,
    kv_cache: Optional[KVCache],
    input_ids: torch.Tensor,  # [T_total] flat 1D
    positions: torch.Tensor,  # [T_total] int64 — per-token global absolute pos
    cu_seqlens: torch.Tensor,  # [B+1] int64 — request boundaries
    block_tables_by_type: Optional[Dict[str, torch.Tensor]],
    attn_inputs: Optional[PyAttentionInputs] = None,
    prepare_hidden_fn: Optional[Any] = None,
) -> torch.Tensor:
    """Flat per-layer loop — vLLM-aligned layout.

    Shapes:
      * ``input_ids``   ``[T_total]``    — flat tokens across the forward's requests
      * ``positions``   ``[T_total]``    — per-token global absolute position (RoPE)
      * ``cu_seqlens``  ``[B+1]``        — per-request cumulative-token prefix sum
      * ``hidden``      ``[T_total, hc, dim]`` — internal, flat in the token axis
      * returns         ``[T_total, dim]`` — pre-lm-head, engine applies lm_head.
        On a non-last PP stage (``v4.norm is None``) it instead returns the
        pre-reduce ``[T_total, hc, dim]`` boundary tensor for the next stage.

    The ``B`` axis is collapsed out of ``input_ids`` / ``hidden`` entirely,
    matching vLLM's ``DeepseekV4`` (``deepseek_v4.py:1310-1317``). Per-request
    bookkeeping that still needs request boundaries (block-table lookups,
    compressor/indexer per-row state) is carried by ``cu_seqlens``.

    **Stage-2 compat shim**: ``Block.forward`` is now flat-native (accepts
    ``[T, hc, dim]`` + 1D ``input_ids`` / ``positions`` / ``cu_seqlens``)
    so the layer call site has no unsqueeze/squeeze. ``attention.py`` /
    ``compressor.py`` / ``indexer.py`` still consume ``[B=1, T, hc, dim]``
    internally — ``Block.forward`` re-wraps them. ``_hc_head_reduce`` +
    ``norm`` also still assume a 4D input (``dim=2`` for the hc reduction),
    so the reduce + norm pair here is still wrapped until ``transformer.py``
    is flattened.

    When ``attn_inputs`` is provided AND cache_store is active, each layer's
    owned KV regions are registered with the PD-disagg cache_store immediately
    after that layer's forward.
    """
    _fs_marks: Optional[Dict[str, float]] = None
    _fs_mem0 = None
    _fs_ev0 = None
    if _FWD_STATS:
        _fs_marks = {"entry": time.perf_counter()}
        _fs_mem0 = _fwd_stats_snap()
    if _FWD_GPU:
        _fs_ev0 = torch.cuda.Event(enable_timing=True)
        _fs_ev0.record()
    _fs_prof = None
    if _FWD_PROFILE and _fwd_profile_rank_ok() and int(input_ids.size(0)) >= 1024:
        _FWD_PROFILE_CT[0] += 1
        if _FWD_PROFILE_CT[0] == _FWD_PROFILE_IDX:
            _fs_prof = _fwd_profile_start()

    # Build + propagate CP context once per prefill step. Under CP the
    # caller hands us a per-rank chunk slice (T_local = chunk_length),
    # and each attn / compressor / indexer reads ``cp_ctx`` off the
    # module to compute its own per-token positions. Without CP we pass
    # None to clear any stale context from a prior forward (warmup).
    cp_info = getattr(v4, "_cp_info", None)
    cp_size = getattr(v4, "_cp_size", 1)
    cp_rank = getattr(v4, "_cp_rank", 0)
    cp_ctx = None
    if cp_info is not None and cp_size > 1:
        cp_ctx = build_cp_context_for_forward(
            cp_info,
            cp_size,
            cp_rank,
            int(input_ids.size(0)),
            input_ids.device,
            prefix_lengths=getattr(attn_inputs, "prefix_lengths", None),
            kv_cache_sharded=bool(getattr(v4, "_kv_cache_sharded", False)),
        )
    v4._propagate_cp_ctx(cp_ctx)
    if _fs_marks is not None:
        _fs_marks["cpctx"] = time.perf_counter()
    if os.environ.get("DSV4_CP_PROBE"):
        # Report the first few DISTINCT large token counts. A one-shot probe is
        # useless here: the 5-token warm-up request legitimately has CP cleared
        # (set_cp_info falls through to (None, 1, 0) when
        # attn.context_parallel_info is None), so it says nothing about whether
        # CP splits a real 32K chunk.
        _seen = getattr(forward_layers, "_cp_probe_seen", None)
        if _seen is None:
            _seen = forward_layers._cp_probe_seen = set()
        _T = int(input_ids.size(0))
        if _T >= 1024 and len(_seen) < 4 and _T not in _seen:
            _seen.add(_T)
            import sys as _sys

            _il = getattr(attn_inputs, "input_lengths", None)
            _pl = getattr(attn_inputs, "prefix_lengths", None)
            print(
                "[CPPROBE] T={T} cp_info={ci} cp_size={cs} cp_rank={cr} cp_ctx={cc} "
                "chunk_len={cl} seq_len_full={sf} input_lengths={il} prefix_lengths={pl}".format(
                    T=_T,
                    ci=cp_info is not None,
                    cs=cp_size,
                    cr=cp_rank,
                    cc="None(CP CLEARED)" if cp_ctx is None else "set",
                    cl=getattr(cp_ctx, "chunk_length", None),
                    sf=getattr(cp_ctx, "seq_len_full", None),
                    il=_il.flatten()[:4].tolist() if _il is not None else None,
                    pl=_pl.flatten()[:4].tolist() if _pl is not None else None,
                ),
                file=_sys.stderr,
                flush=True,
            )
    if cp_ctx is not None:
        # The framework's fallback position_ids are rank-local contiguous
        # after ZigZagProcessor rewrites input_lengths to CP chunk lengths.
        # DSV4 attention/indexer/compressor need the per-token absolute
        # request positions carried by CPContext.
        positions = cp_ctx.global_positions.to(
            device=positions.device, dtype=torch.long
        )
    positions = positions.reshape(-1).contiguous()
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.reshape(-1).contiguous()
    if _fs_marks is not None:
        _fs_marks["pos"] = time.perf_counter()

    # MOEDBG hook (mirrors V4Transformer.forward standalone path so the
    # smoke / production prefill path produces the same per-layer dump
    # consumed by /tmp/moedbg_runs diff scripts).  Read once per forward.
    _rt_on = _rt.ENABLED
    if _rt_on:
        _rt.begin(seqlen=int(input_ids.size(0)))
        if _rt._get_buf() is None:
            _rt_on = False

    # Build the per-layer cache_store writer once per forward. Active
    # only on prefill calls with cache_store_inputs bound; otherwise
    # ``write_cache_store_impl`` is None and the per-layer call site is
    # a cheap None check.
    write_cache_store_impl = None
    if kv_cache is not None and attn_inputs is not None:
        write_cache_store_impl = create_write_cache_store_impl(attn_inputs, kv_cache)

    if prepare_hidden_fn is None:
        h = v4.embed_full(input_ids)  # [T_total, dim]
        if _rt_on:
            _rt.record("prefill_embed_out", h)
        h = h.unsqueeze(-2).repeat(1, v4.hc_mult, 1)  # [T_total, hc, dim]
    else:
        h = prepare_hidden_fn(input_ids=input_ids, positions=positions)
    if _rt_on:
        _rt.record("prefill_embed_hc_expanded", h)

    capture_ids = frozenset(v4.capture_aux_hidden_layer_ids)
    capture_aux = bool(capture_ids)
    if _fs_marks is not None:
        _fs_marks["embed"] = time.perf_counter()

    prefill_fast_layer_calls = _prefill_fast_path_layer_calls(v4)
    use_prefill_fast_path = _prefill_fast_path_enabled(
        v4, prepare_hidden_fn, prefill_fast_layer_calls
    )
    if not use_prefill_fast_path:
        prefill_fast_layer_calls = None
    record_range_ctx = (
        _profiler.disable_record_function_ranges
        if use_prefill_fast_path
        else nullcontext
    )
    # FP8 KV-cache: hoist host-side prefill metadata once per ratio bucket
    # and broadcast to every layer's ``AttentionFP8._prefill_meta_shared``.
    # BF16 path doesn't need this; ``Attention`` rebuilds meta inside its
    # own forward.
    with record_range_ctx():
        if v4.fp8_kv_cache:
            sp_int_for_meta = int(positions[0].item())
            sp_per_req: Optional[torch.Tensor] = None
            req_id_per_token: Optional[torch.Tensor] = None
            if cp_ctx is not None:
                # Under CP, rank-local token order is zigzagged. The first token of
                # each rank-local request chunk is therefore not necessarily the
                # request's absolute start position. Use CP metadata instead of
                # deriving request ids from rank-local cu_seqlens.
                sp_per_req = cp_ctx.prefix_lengths.to(
                    device=positions.device, dtype=torch.int64
                ).contiguous()
                req_id_per_token = cp_ctx.req_id_per_token.to(
                    device=positions.device, dtype=torch.int32
                ).contiguous()
            elif cu_seqlens is not None and cu_seqlens.numel() >= 2:
                starts = cu_seqlens[:-1].to(device=positions.device, dtype=torch.int64)
                sp_per_req = (
                    positions.index_select(0, starts).to(torch.int64).contiguous()
                )
                req_id_per_token = (
                    torch.searchsorted(
                        cu_seqlens.to(device=positions.device, dtype=torch.int64),
                        torch.arange(
                            int(cu_seqlens[-1].item()),
                            device=positions.device,
                            dtype=torch.int64,
                        ),
                        right=True,
                    )
                    .sub_(1)
                    .to(torch.int32)
                    .contiguous()
                )
            batch_size = 1
            if cu_seqlens is not None and cu_seqlens.numel() >= 2:
                batch_size = int(cu_seqlens.numel() - 1)
            input_lengths: Optional[torch.Tensor] = None
            prefix_lengths: Optional[torch.Tensor] = None
            max_seqlen_q = 0
            if attn_inputs is not None:
                il = getattr(attn_inputs, "input_lengths", None)
                if il is not None and il.numel() > 0:
                    input_lengths = il.to(
                        device=positions.device, dtype=torch.int32
                    ).contiguous()
                    max_seqlen_q = int(input_lengths.max().item())
                pl = getattr(attn_inputs, "prefix_lengths", None)
                if pl is not None and pl.numel() > 0:
                    prefix_lengths = pl.to(
                        device=positions.device, dtype=torch.int32
                    ).contiguous()
            # Per-forward prefill workspace: one runtime buffer allocated at the
            # top of the forward, freed when ``forward_layers`` returns (so the
            # MTP draft forward, which runs right after on a near-full card, can
            # borrow it). Holds the prefill-Q output (eager) and — whenever CP is
            # active — the main + indexer compressor CP gather/restore scratch
            # (dedicated buffer pairs per role, used by BOTH the serial and
            # overlap paths for the workspace-backed roles). Sizing is MAX
            # (capacity-bound, runtime-length-independent) so every forward
            # allocates the same-sized block → zero allocator fragmentation,
            # IDENTICAL across main and MTP-draft forwards (the draft overrides
            # ``_resolve_prefill_ws_gather_widths`` to size off the main model's
            # ratios — see ``deepseek_v4_mtp_model``). Current-layer SWA ``kv_full``
            # all-gather is intentionally not workspace-backed.
            #
            # ``reserve_cp`` gates the CP region; we cannot derive it from
            # ``compress_ratio != 0`` on the layers because the workspace is bound
            # once for the whole prefill forward. The bound ``_prefill_ws_full_rows>0``
            # is the canonical signal that CP is active at workspace bind time.
            reserve_cp = (cp_ctx is not None) and int(v4._prefill_ws_full_rows) > 0
            q_rows, full_rows = resolve_prefill_workspace_rows(
                v4._prefill_ws_q_rows,
                v4._prefill_ws_full_rows,
                int(input_ids.numel()),
                int(cp_ctx.cp_size) if cp_ctx is not None else 1,
                allow_dynamic_growth=os.environ.get(
                    "DSV4_SM120_DYNAMIC_PREFILL_WORKSPACE", "0"
                )
                == "1",
            )
            ws = PrefillWorkspace(
                input_ids.device,
                q_rows=q_rows,
                q_dim=v4._prefill_ws_q_dim,
                reserve_cp=reserve_cp,
                cp_rows=full_rows,
                main_w=v4._prefill_ws_main_w,
                idx_w=v4._prefill_ws_idx_w,
            )
            build_and_propagate_prefill_meta_fp8(
                v4,
                h,
                sp_int_for_meta,
                kv_cache,
                block_tables_by_type,
                sp_per_req=sp_per_req,
                cu_seqlens=cu_seqlens,
                batch_size=batch_size,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                position_ids=positions,
                req_id_per_token=req_id_per_token,
                max_seqlen_q=max_seqlen_q,
                workspace=ws,
            )

    if _fs_marks is not None:
        _fs_marks["meta"] = time.perf_counter()

    try:
        with record_range_ctx():
            # Two callable chains intentionally coexist:
            #   * normal ``Block.forward`` keeps debug checks and fallback layouts;
            #   * cached fast callables are validated once for the FP8 production
            #     matrix, then reused for every request. Keep both signatures in
            #     sync when changing prefill inputs, including B>1/reuse metadata.
            layer_calls = (
                prefill_fast_layer_calls
                if prefill_fast_layer_calls is not None
                else v4.layers
            )
            for layer_idx, layer_call in enumerate(layer_calls):
                h = layer_call(
                    h,  # [T, hc, dim]
                    input_ids,  # [T]
                    positions,  # [T]
                    cu_seqlens,  # [B+1]
                    kv_cache=kv_cache,
                    block_tables_by_type=block_tables_by_type,
                )  # [T, hc, dim]
                # ``capture_ids`` are GLOBAL layer ids while ``layer_idx`` is
                # this stage's local position, so translate before matching.
                if capture_aux:
                    global_layer_id = v4.pp_global_layer_ids[layer_idx]
                    if global_layer_id in capture_ids:
                        v4.capture_aux_hidden(global_layer_id, h)
                if _rt_on:
                    _rt.record(f"prefill_layer{layer_idx:02d}_out", h)
                if write_cache_store_impl is not None:
                    # Cache surfaces stay LOCAL on purpose: the C++ layout is
                    # projected to this stage and numbered 0..len(layers)-1.
                    write_cache_store_impl(kv_cache.get_layer_cache_groups(layer_idx))
                if _rt_on:
                    _rt.record(f"layer{layer_idx:02d}_out", h)
                    if cp_ctx is None:
                        layer_last = h[-1:].contiguous()
                    else:
                        layer_last_pos = cp_ctx.seq_len_total - 1
                        layer_last_mask = (
                            cp_ctx.global_positions == layer_last_pos
                        ) & cp_ctx.local_is_real
                        layer_last = h[layer_last_mask].contiguous()
                        dbg_pos = getattr(_rt, "_DBG_GLOBAL_POS", -1)
                        if dbg_pos >= 0:
                            layer_pos_mask = (
                                cp_ctx.global_positions == dbg_pos
                            ) & cp_ctx.local_is_real
                            _rt.record(
                                f"layer{layer_idx:02d}_pos{dbg_pos}",
                                h[layer_pos_mask].contiguous(),
                            )
                        layer_tail_mask = (
                            (
                                cp_ctx.global_positions
                                >= max(cp_ctx.seq_len_total - 128, 0)
                            )
                            & (cp_ctx.global_positions < cp_ctx.seq_len_total)
                            & cp_ctx.local_is_real
                        )
                        _rt.record(
                            f"layer{layer_idx:02d}_tail128",
                            h[layer_tail_mask].contiguous(),
                        )
                    _rt.record(f"layer{layer_idx:02d}_last", layer_last)
    finally:
        # Always drop the per-layer ``common.workspace`` references, even if a
        # layer raises mid-prefill (e.g. a CUDA OOM under memory pressure —
        # the exact case this per-forward workspace exists to relieve). The
        # ref lives on each layer's ``_prefill_meta_shared`` (a persistent
        # module attr), so without this the ~16 GiB workspace would stay
        # pinned past the failing forward and starve the retry / next request
        # on a near-full card. ``clear`` is idempotent (sets None per layer).
        if v4.fp8_kv_cache:
            clear_prefill_meta_shared_fp8(v4)
        if _fs_prof is not None:
            # __exit__ synchronises, so every kernel the loop launched is
            # captured even though the launches themselves are async.
            _fs_prof.__exit__(None, None, None)
            _fwd_profile_dump(_fs_prof)
            _fs_prof = None

    if _fs_marks is not None:
        _fs_marks["loop"] = time.perf_counter()

    if v4._mtp_hidden_buffer is not None:
        if capture_aux:
            # DSpARK mode: the buffer already holds this forward's aux rows
            # (written per selected layer above); only account for them.
            v4._note_aux_hidden_rows(h.size(0), is_cuda_graph=False)
        else:
            _pre_hc_flat = h.flatten(-2)
            v4._write_mtp_hidden_buffer(_pre_hc_flat, is_cuda_graph=False)
            if v4._mtp_last_hidden_buffer is not None:
                _last_pre_hc = _last_hidden_by_request(_pre_hc_flat, cu_seqlens, cp_ctx)
                v4._write_mtp_last_hidden_buffer(_last_pre_hc)

    # PP: the mHC head reduce and the final norm belong to the LAST stage only —
    # that is the only stage that loads ``head_hc`` / ``norm`` / ``head_weight``.
    # A non-last stage must hand downstream the PRE-reduce ``[T, hc, dim]``
    # tensor: reducing here would collapse the hyper-connection lanes the next
    # stage continues from.  Returning before the reduce also skips the debug
    # blocks below, which dereference ``v4.head_weight``.
    if v4.norm is None:
        if _fs_marks is not None:
            _fs_marks["tail"] = time.perf_counter()
            _fwd_stats_report_row(
                n_tokens=int(input_ids.size(0)),
                cp_size=int(cp_ctx.cp_size) if cp_ctx is not None else 1,
                marks=_fs_marks,
                mem_before=_fs_mem0,
                mem_after=_fwd_stats_snap(),
                ev0=_fs_ev0,
            )
        return h  # [T, hc, dim]

    # _hc_head_reduce is flat-native: [T, hc, dim] -> [T, dim].
    # Framework ``RMSNorm`` expects 2D, which matches the [T, dim] shape here.
    with record_range_ctx():
        h = v4._hc_head_reduce(h)  # [T, dim]
        if _rt_on:
            _rt.record("prefill_hc_reduced", h)
        h = v4.norm(h)  # [T, dim]
    if _rt_on:
        _rt.record("prefill_final_norm", h)
        if cp_ctx is None:
            last_h = h[-1:].contiguous()
        else:
            last_pos = cp_ctx.seq_len_total - 1
            last_mask = (cp_ctx.global_positions == last_pos) & cp_ctx.local_is_real
            last_h = h[last_mask].contiguous()
        _rt.record("lm_last_hidden", last_h)
        lm_logits = torch.mm(
            last_h.to(v4.head_weight.dtype), v4.head_weight.t()
        ).float()
        _rt.record("lm_logits_last", lm_logits)
        top_k = min(16, lm_logits.size(-1))
        lm_top_values, lm_top_indices = torch.topk(lm_logits, k=top_k, dim=-1)
        _rt.record("lm_top_values", lm_top_values)
        _rt.record("lm_top_indices", lm_top_indices)

    if _rt_on:
        extra: dict = {
            "input_ids_shape": tuple(input_ids.shape),
            "input_ids": input_ids.detach().cpu(),
            "path": "prefill",
            "positions": positions.detach().cpu(),
            "cu_seqlens": cu_seqlens.detach().cpu(),
        }
        if cp_ctx is not None:
            extra.update(
                {
                    "cp_size": cp_ctx.cp_size,
                    "cp_rank": cp_ctx.cp_rank,
                    "chunk_length": cp_ctx.chunk_length,
                    "padded_seq_len": cp_ctx.padded_seq_len,
                    "seq_len_full": cp_ctx.seq_len_full,
                    "prefix_length": cp_ctx.prefix_length,
                    "seq_len_total": cp_ctx.seq_len_total,
                    "relative_positions": cp_ctx.relative_positions.detach().cpu(),
                    "global_positions": cp_ctx.global_positions.detach().cpu(),
                    "unpad_restore": cp_ctx.unpad_restore.detach().cpu(),
                    "local_is_real": cp_ctx.local_is_real.detach().cpu(),
                }
            )
        else:
            extra.update(
                {
                    "cp_size": 1,
                    "cp_rank": 0,
                    "seq_len_full": int(input_ids.size(0)),
                    "prefix_length": 0,
                    "seq_len_total": int(input_ids.size(0)),
                }
            )
        if attn_inputs is not None:
            for name in ("input_lengths", "prefix_lengths", "sequence_lengths"):
                value = getattr(attn_inputs, name, None)
                if value is not None and value.numel() > 0:
                    extra[name] = value.detach().cpu()
        step = getattr(v4, "_dbg_step", 0)
        _rt.dump(step=step, extra=extra)
        v4._dbg_step = step + 1
    if _fwd_dbg.enabled():
        _fwd_dbg.print_prefill(
            hidden=h,
            input_ids=input_ids,
            positions=positions,
            cu_seqlens=cu_seqlens,
            attn_inputs=attn_inputs,
            cp_ctx=cp_ctx,
            head_weight=getattr(v4, "head_weight", None),
            step=int(getattr(v4, "_dbg_step", 0)),
        )
    # The per-forward ``PrefillWorkspace`` (prefill-Q + optional CP
    # gather/restore scratch) is a local of this function: it drops here on
    # return, returning ~16 GiB to the caching allocator so the MTP draft
    # forward (which runs right after the main model on a near-full card) can
    # borrow it. No explicit reset needed — the per-layer ``common.workspace``
    # references were cleared by ``clear_prefill_meta_shared_fp8`` above.
    if _fs_marks is not None:
        _fs_marks["tail"] = time.perf_counter()
        _fwd_stats_report_row(
            n_tokens=int(input_ids.size(0)),
            cp_size=int(cp_ctx.cp_size) if cp_ctx is not None else 1,
            marks=_fs_marks,
            mem_before=_fs_mem0,
            mem_after=_fwd_stats_snap(),
            ev0=_fs_ev0,
        )
    return h  # [T, dim]


def forward_prefill(
    v4: V4Transformer,
    kv_cache: Optional[KVCache],
    parallelism_config: Optional[ParallelismConfig],
    inputs: PyModelInputs,
    prepare_hidden_fn: Optional[Any] = None,
) -> PyModelOutputs:
    """Prefill dispatcher — single :func:`forward_layers` call on the full
    flat ``[T_total]`` batch (vLLM-aligned).

    Pulls flat metadata off :attr:`PyModelInputs.attention_inputs`. For the
    multi-group DSV4 cache that attribute is a ``{tag: PyAttentionInputs}``
    mapping, so group-invariant fields are read off the primary entry while
    block tables are collected per tag:

    * ``positions``  = ``attn.combo_position_ids`` — ``[T_total]`` global pos
    * ``cu_seqlens`` = ``attn.cu_seqlens``         — ``[B+1]`` int32 prefix sum
    * ``block_tables_by_type`` = Dict[cache tag, [B, max_blocks]] — full-batch
      block tables (B axis = request axis), built via
      :func:`build_block_tables_batched`.

    Downstream (``block.py`` / ``attention.py`` / ``compressor.py`` /
    ``indexer.py``) consumes the cu_seqlens-aware metadata directly; under CP
    the per-layer setup swaps in CPContext's request-absolute positions and
    full-length write-side view.

    Returns ``PyModelOutputs`` with ``[T_total, dim]`` pre-lm-head hidden.  On a
    non-last PP stage it instead carries the pre-reduce ``[T_total, hc, dim]``
    boundary tensor, published on ``pp_intermediates`` for the next stage.
    """
    attn_inputs = inputs.attention_inputs
    attn = primary_attention_inputs(attn_inputs, kv_cache)
    if attn is None:
        raise RuntimeError("DSV4 prefill: PyModelInputs carries no attention inputs")

    # Context-Parallel setup must precede the per-layer loop because
    # forward_layers reads v4._cp_info to build the CP context.
    set_cp_info(v4, parallelism_config, attn, is_prefill=True)

    input_ids: torch.Tensor = inputs.input_ids  # [T_total] flat 1D

    # Framework already populates these — don't recompute.
    #  * ``attn.cu_seqlens``          : [B+1]     per-request cumulative prefix sum
    #  * ``attn.combo_position_ids``  : [T_total] per-token global absolute pos
    #    (the field the dev branch called ``position_ids``; it is only populated
    #    when the model declares a position-id length factor, so the synthesize
    #    branch below stays the live path for DSV4).
    cu_seqlens = getattr(attn, "cu_seqlens", None)
    # The startup real-warmup request is a valid single-request prefill, but
    # some framework paths leave the optional cu_seqlens field as an empty
    # tensor.  The FP8 metadata builder requires the vLLM-style [B+1] prefix
    # sum, so reconstruct it from the authoritative input_lengths when
    # available (and fall back to the complete flat input for a single request).
    if cu_seqlens is None or cu_seqlens.numel() < 2:
        input_lengths_for_cu = getattr(attn, "input_lengths", None)
        if input_lengths_for_cu is not None and input_lengths_for_cu.numel() > 0:
            lengths = input_lengths_for_cu.reshape(-1).to(
                device=input_ids.device, dtype=torch.int32
            )
            if int(lengths.sum().item()) == int(inputs.input_ids.numel()):
                cu_seqlens = torch.cat(
                    [
                        torch.zeros(1, dtype=torch.int32, device=input_ids.device),
                        torch.cumsum(lengths, dim=0),
                    ]
                )
        if cu_seqlens is None or cu_seqlens.numel() < 2:
            cu_seqlens = torch.tensor(
                [0, int(input_ids.numel())],
                dtype=torch.int32,
                device=input_ids.device,
            )
    positions = getattr(attn, "combo_position_ids", None)
    # warmup / cudagraph capture path doesn't populate combo_position_ids —
    # synthesize from (prefix_lengths, input_lengths). Prefer ``_d`` (GPU)
    # variants when available: during cudagraph capture, the host-side
    # ``input_lengths`` / ``prefix_lengths`` are pinned int32 CPU tensors,
    # but a dtype-converting ``.to(device=..., dtype=int64)`` on a pinned
    # tensor produces an unpinned intermediate which capture rejects.
    if positions is None or positions.numel() == 0:
        il_d = attn.input_lengths
        pl_d = attn.prefix_lengths
        input_lens = il_d if il_d.numel() > 0 else attn.input_lengths
        prefix_lens = pl_d if pl_d.numel() > 0 else attn.prefix_lengths
        positions = _build_positions_from_lengths(
            input_lens,
            prefix_lens,
            input_ids.device,
            total_tokens=int(input_ids.numel()),
        )

    block_tables_by_type = build_block_tables_batched(kv_cache, attn_inputs)

    # PP: a non-first stage owns no embedding and resumes the upstream stage's
    # activations instead.  ``forward_layers`` reads a non-None
    # ``prepare_hidden_fn`` as "hidden is already ``[T, hc, dim]``", which is
    # exactly the shape the upstream stage publishes.  Under CP the boundary
    # tensor is this rank's own CP chunk: PP pairs equal ``cp_rank`` across
    # stages (``PPLayout::rankOfStage`` preserves ``tp_rank``), so the token
    # widths already match and no gather is needed here.
    if v4.embed is None:
        upstream_hidden = (
            inputs.pp_intermediates.get("hidden_states")
            if inputs.pp_intermediates
            else None
        )
        if upstream_hidden is None:
            raise RuntimeError(
                "DSV4 prefill on a non-first PP stage received no upstream "
                "hidden_states in pp_intermediates"
            )

        def _resume_upstream_hidden(input_ids, positions):
            return upstream_hidden

        prepare_hidden_fn = _resume_upstream_hidden

    hidden = forward_layers(
        v4,
        kv_cache,
        input_ids,
        positions,
        cu_seqlens,
        block_tables_by_type,
        attn_inputs=attn,
        prepare_hidden_fn=prepare_hidden_fn,
    )  # [T_total, dim], or [T_total, hc, dim] on a non-last PP stage
    outputs = PyModelOutputs(hidden)
    if v4.norm is None:
        # Non-last stage: hand the pre-reduce mHC lanes downstream.  Dropping
        # back to [T, dim] here would corrupt the hyper-connection stream.
        outputs.pp_intermediates = {"hidden_states": hidden}
    return outputs
