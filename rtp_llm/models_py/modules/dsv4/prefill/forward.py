"""DSV4 prefill forward helpers — extracted from ``DeepSeekV4Model``.

Exposes qwen3-style prefill primitives as free functions so the Model
class stays thin:

* ``set_cp_info``                    — bind/clear Context-Parallel metadata on ``v4``
* ``forward_layers``                 — per-layer loop body (embed → layers → reduce → norm)
* unified cache-store registration via
  :func:`rtp_llm.models_py.modules.factory.attention.common.create_write_cache_store_impl`
* ``forward_prefill``                — full prefill arm (per-request loop over flat 1D input_ids)

Generic KV-cache lookup helpers (``build_block_tables_batched``) live
in :mod:`rtp_llm.models_py.modules.dsv4.kv_cache_utils`.

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
  * ``attn.position_ids``        — framework-provided positions; under CP,
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
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from rtp_llm.models_py.modules.dsv4 import _forward_tensor_debug as _fwd_dbg
from rtp_llm.models_py.modules.dsv4 import _profiler
from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt
from rtp_llm.models_py.modules.dsv4.cp import (
    build_cp_context,
    cp_gather_last_by_request,
)
from rtp_llm.models_py.modules.dsv4.fp8.prefill_meta import (
    build_and_propagate_prefill_meta_fp8,
    clear_prefill_meta_shared_fp8,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import build_block_tables_batched
from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace
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


def _log_prefill_graph_line(line: str) -> None:
    try:
        graph_log = os.path.join(
            os.environ.get("HIPPO_APP_WORKDIR", "/tmp"),
            "graph_decision.log",
        )
        os.makedirs(os.path.dirname(graph_log), exist_ok=True)
        with open(graph_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUE_ENV_VALUES


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_int_fail_closed(name: str, default: int) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return None


def _prefill_request_token_count(
    input_ids: torch.Tensor,
    cp_ctx: Optional[Any],
) -> int:
    if cp_ctx is not None and hasattr(cp_ctx, "seq_len_full"):
        return int(getattr(cp_ctx, "seq_len_full"))
    return int(input_ids.numel())


class _MetadataOnlyKVCache:
    """KV-cache metadata shim for graph diagnostics that must not touch pools."""

    def __init__(self, source: Any):
        for name in (
            "seq_size_per_block",
            "kernel_seq_size_per_block",
            "num_kv_heads",
            "head_dim",
            "use_mla",
            "kv_lora_rank",
            "rope_head_dim",
            "layer_group_types",
            "group_region_names",
            "group_seq_size_per_block",
            "layer_region_to_group_id",
        ):
            setattr(self, name, getattr(source, name, None))

    def get_layer_cache(self, *args, **kwargs):
        raise RuntimeError("metadata-only graph KV cache has no pool tensors")

    def get_layer_caches(self, *args, **kwargs):
        return []

    def get_raw_pool_tensor(self, *args, **kwargs):
        raise RuntimeError("metadata-only graph KV cache has no raw pool tensor")


def _small_token_bypass_enabled(
    v4: "V4Transformer",
    input_ids: torch.Tensor,
    prepare_hidden_fn: Optional[Any],
    cp_ctx: Optional[Any],
) -> bool:
    if not _env_flag("DSV4_PREFILL_SMALL_TOKEN_BYPASS", "0"):
        return False
    max_tokens = _env_int_fail_closed(
        "DSV4_PREFILL_SMALL_TOKEN_BYPASS_MAX_TOKENS", 8192
    )
    token_count = _prefill_request_token_count(input_ids, cp_ctx)
    if max_tokens is None or max_tokens <= 0 or token_count > max_tokens:
        return False
    if prepare_hidden_fn is not None:
        return False
    if _rt.ENABLED or _fwd_dbg.enabled():
        return False
    return hasattr(v4, "layers")


def _prefill_graph_static_eager_run_allowed(
    v4: "V4Transformer",
    graph_decision: Optional[Any],
    *,
    kv_cache: Optional[Any],
    static_state_updated_this_forward: bool,
    graph_static_bind_allowed: bool,
    graph_replay_requested: bool,
    use_small_token_bypass: bool,
    write_cache_store_impl: Optional[Any],
    rt_on: bool,
) -> bool:
    """Diagnostic-only gate for running the layer loop on static buffers eagerly."""

    if not _env_flag("DSV4_PREFILL_GRAPH_STATIC_EAGER_RUN", "0"):
        return False
    if not static_state_updated_this_forward:
        return False
    if graph_replay_requested:
        return False
    if not graph_static_bind_allowed or not use_small_token_bypass:
        return False
    if write_cache_store_impl is not None or rt_on:
        return False
    if kv_cache is not None and not _env_flag(
        "DSV4_PREFILL_GRAPH_STATIC_EAGER_ALLOW_LIVE_KV", "0"
    ):
        return False
    if graph_decision is None or not getattr(graph_decision, "enabled", False):
        return False
    if not _env_flag("DSV4_PREFILL_GRAPH_BIND_STATIC_INPUTS", "0"):
        return False
    if getattr(v4, "fp8_kv_cache", False) and not _env_flag(
        "DSV4_PREFILL_GRAPH_BIND_STATIC_META", "0"
    ):
        return False
    state = getattr(v4, "_last_prefill_graph_state", None)
    if state is None or not getattr(state, "valid", False):
        return False
    if getattr(v4, "_last_prefill_graph_bind_static_error", None) is not None:
        return False
    if getattr(v4, "_last_prefill_graph_bind_static_meta_error", None) is not None:
        return False
    report = getattr(v4, "_last_prefill_graph_bound_capture_surface", None)
    if report is None or not getattr(report, "static_bound", False):
        return False
    return True


def _prefill_graph_copy_shadow_allowed(
    v4: "V4Transformer",
    graph_decision: Optional[Any],
    *,
    static_args_bound_this_forward: bool,
    static_state_updated_this_forward: bool,
    use_small_token_bypass: bool,
    write_cache_store_impl: Optional[Any],
    rt_on: bool,
) -> bool:
    """Diagnostic-only shadow graph that validates static graph copy mechanics."""

    if not _env_flag("DSV4_PREFILL_GRAPH_COPY_SHADOW", "0"):
        return False
    if not static_args_bound_this_forward or not static_state_updated_this_forward:
        return False
    if not use_small_token_bypass:
        return False
    if write_cache_store_impl is not None or rt_on:
        return False
    if graph_decision is None or not getattr(graph_decision, "enabled", False):
        return False
    state = getattr(v4, "_last_prefill_graph_state", None)
    if state is None or not getattr(state, "valid", False):
        return False
    if getattr(v4, "_last_prefill_graph_bind_static_error", None) is not None:
        return False
    if getattr(v4, "_last_prefill_graph_bind_static_meta_error", None) is not None:
        return False
    report = getattr(v4, "_last_prefill_graph_bound_capture_surface", None)
    if report is None or not getattr(report, "static_bound", False):
        return False
    return True


def _prefill_graph_needs_bound_capture_surface() -> bool:
    return (
        _env_flag("DSV4_PREFILL_GRAPH_STATIC_EAGER_RUN", "0")
        or _env_flag("DSV4_PREFILL_GRAPH_COPY_SHADOW", "0")
        or _env_flag("DSV4_PREFILL_GRAPH_CAPTURE_SURFACE_LOG", "0")
    )


def _clear_prefill_graph_copy_shadow_result(v4: "V4Transformer") -> None:
    setattr(v4, "_last_prefill_graph_copy_shadow_stats", None)
    setattr(v4, "_last_prefill_graph_copy_shadow_error", None)


def _set_prefill_graph_copy_shadow_stats(
    v4: "V4Transformer",
    *,
    mode: str,
    exact: bool,
    max_abs: float,
    mean_abs: float,
) -> None:
    setattr(
        v4,
        "_last_prefill_graph_copy_shadow_stats",
        {
            "mode": mode,
            "exact": exact,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
        },
    )
    setattr(v4, "_last_prefill_graph_copy_shadow_error", None)


def _set_prefill_graph_copy_shadow_error(
    v4: "V4Transformer",
    error: str,
) -> None:
    setattr(v4, "_last_prefill_graph_copy_shadow_stats", None)
    setattr(v4, "_last_prefill_graph_copy_shadow_error", error)


def _prefill_graph_attn_body_shadow_layer() -> Optional[int]:
    if not _env_flag("DSV4_PREFILL_GRAPH_ATTN_BODY_SHADOW", "0"):
        return None
    layer = _env_int_fail_closed("DSV4_PREFILL_GRAPH_ATTN_BODY_SHADOW_LAYER", 0)
    if layer is None or layer < 0:
        return None
    return layer


def _prefill_graph_attn_body_shadow_allowed(
    v4: "V4Transformer",
    graph_decision: Optional[Any],
    *,
    layer_idx: int,
    loop_kv_cache: Optional[Any],
    static_state_updated_this_forward: bool,
    graph_replay_requested: bool,
    write_cache_store_impl: Optional[Any],
    rt_on: bool,
) -> bool:
    shadow_layer = _prefill_graph_attn_body_shadow_layer()
    if shadow_layer is None or int(layer_idx) != int(shadow_layer):
        return False
    if graph_replay_requested:
        return False
    if not static_state_updated_this_forward:
        return False
    if not _env_flag("DSV4_PREFILL_GRAPH_ATTN_BODY_SHADOW_ALLOW_GRAPH_KV", "0"):
        return False
    if loop_kv_cache is None:
        return False
    if write_cache_store_impl is not None or rt_on:
        return False
    if graph_decision is None or not getattr(graph_decision, "enabled", False):
        return False
    state = getattr(v4, "_last_prefill_graph_state", None)
    if state is None or not getattr(state, "valid", False):
        return False
    if getattr(v4, "_last_prefill_graph_bind_static_error", None) is not None:
        return False
    if getattr(v4, "_last_prefill_graph_bind_static_meta_error", None) is not None:
        return False
    if getattr(state, "key", None) != getattr(graph_decision, "key", None):
        return False
    key = getattr(graph_decision, "key", None)
    if key is not None and (
        getattr(key, "prefix_bucket", 0) != 0 or getattr(key, "reuse_bucket", 0) != 0
    ):
        return False
    if getattr(state, "cuda_graph", None) is not None:
        return False
    return True


def _clear_prefill_graph_attn_body_shadow_result(v4: "V4Transformer") -> None:
    setattr(v4, "_last_prefill_graph_attn_body_shadow_stats", None)
    setattr(v4, "_last_prefill_graph_attn_body_shadow_error", None)


def _clear_prefill_graph_attn_body_shadow_result_if_enabled(
    v4: "V4Transformer",
) -> bool:
    if not _env_flag("DSV4_PREFILL_GRAPH_ATTN_BODY_SHADOW", "0"):
        return False
    _clear_prefill_graph_attn_body_shadow_result(v4)
    return True


def _set_prefill_graph_attn_body_shadow_stats(
    v4: "V4Transformer",
    *,
    layer_idx: int,
    exact: bool,
    max_abs: float,
    mean_abs: float,
) -> None:
    setattr(
        v4,
        "_last_prefill_graph_attn_body_shadow_stats",
        {
            "layer_idx": int(layer_idx),
            "exact": exact,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
        },
    )
    setattr(v4, "_last_prefill_graph_attn_body_shadow_error", None)


def _set_prefill_graph_attn_body_shadow_error(
    v4: "V4Transformer",
    error: str,
) -> None:
    setattr(v4, "_last_prefill_graph_attn_body_shadow_stats", None)
    setattr(v4, "_last_prefill_graph_attn_body_shadow_error", error)


def _release_prefill_graph_shadow_kv(state: Any) -> None:
    if state is None:
        return
    state.graph_kv_cache = None
    state.graph_kv_block_cap = 0
    state._graph_kv_signature = None


def _compare_prefill_graph_attn_body_shadow_tensors(
    v4: "V4Transformer",
    *,
    layer_idx: int,
    shadow_out: torch.Tensor,
    eager_out: torch.Tensor,
) -> None:
    # If live eager attention failed asynchronously, make that a real forward
    # failure rather than hiding it as a diagnostic compare problem.
    torch.cuda.synchronize(eager_out.device)
    exact = bool(torch.equal(shadow_out, eager_out))
    diff = (shadow_out.detach().float() - eager_out.detach().float()).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    _set_prefill_graph_attn_body_shadow_stats(
        v4,
        layer_idx=layer_idx,
        exact=exact,
        max_abs=max_abs,
        mean_abs=mean_abs,
    )
    if _env_flag("DSV4_PREFILL_GRAPH_LOG_DECISION", "0"):
        _log_prefill_graph_line(
            "[DSV4PrefillGraph] stage=attn_body_shadow "
            f"enabled=True layer={layer_idx} exact={exact} "
            f"max_abs={max_abs:.9g} mean_abs={mean_abs:.9g}"
        )


def _graph_prefix_facts(
    cp_ctx: Optional[Any], attn_inputs: Optional[Any], *, batch_size: int = 1
) -> tuple[int, int, bool]:
    prefix_length = 0
    max_prefix_length = 0
    prefix_unknown = False
    if cp_ctx is not None:
        cp_prefix_lengths = getattr(cp_ctx, "prefix_lengths", None)
        if cp_prefix_lengths is not None and cp_prefix_lengths.numel() > 0:
            if cp_prefix_lengths.device.type == "cpu":
                max_prefix_length = int(cp_prefix_lengths.max().item())
                prefix_length = max_prefix_length
            elif int(batch_size) <= 1:
                prefix_length = int(getattr(cp_ctx, "prefix_length", 0))
                max_prefix_length = prefix_length
            else:
                prefix_unknown = True
        else:
            prefix_length = int(getattr(cp_ctx, "prefix_length", 0))
            max_prefix_length = prefix_length
    elif attn_inputs is not None:
        prefix_lengths = getattr(attn_inputs, "prefix_lengths", None)
        if prefix_lengths is not None and prefix_lengths.numel() > 0:
            if prefix_lengths.device.type == "cpu":
                max_prefix_length = int(prefix_lengths.max().item())
                prefix_length = max_prefix_length
            else:
                prefix_unknown = True
    return prefix_length, max_prefix_length, prefix_unknown


def _build_positions_from_lengths(
    input_lengths: torch.Tensor,  # [B] int
    prefix_lengths: torch.Tensor,  # [B] int
    device: torch.device,
    total_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Synthesize per-token global positions ``[T_total]`` int64 when the
    framework didn't populate ``attn.position_ids`` (warmup / cudagraph
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
        and parallelism_config.prefill_cp_config.is_enabled()
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
    block_tables_by_type: Optional[Dict[int, torch.Tensor]],
    attn_inputs: Optional[PyAttentionInputs] = None,
    prepare_hidden_fn: Optional[Any] = None,
) -> torch.Tensor:
    """Flat per-layer loop — vLLM-aligned layout.

    Shapes:
      * ``input_ids``   ``[T_total]``    — flat tokens across the forward's requests
      * ``positions``   ``[T_total]``    — per-token global absolute position (RoPE)
      * ``cu_seqlens``  ``[B+1]``        — per-request cumulative-token prefix sum
      * ``hidden``      ``[T_total, hc, dim]`` — internal, flat in the token axis
      * returns         ``[T_total, dim]`` — pre-lm-head, engine applies lm_head

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
        T_local = int(input_ids.size(0))
        prefix_offsets = 0
        prefix_lengths = getattr(attn_inputs, "prefix_lengths", None)
        if prefix_lengths is not None and prefix_lengths.numel() > 0:
            prefix_offsets = prefix_lengths.to(
                device=input_ids.device, dtype=torch.long
            )
        cp_ctx = build_cp_context(
            cp_info,
            cp_size,
            cp_rank,
            T_local,
            input_ids.device,
            position_offset=prefix_offsets,
            kv_cache_sharded=bool(getattr(v4, "_kv_cache_sharded", False)),
        )
    v4._propagate_cp_ctx(cp_ctx)
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
        h = v4.embed(input_ids)  # [T_total, dim]
        if _rt_on:
            _rt.record("prefill_embed_out", h)
        h = h.unsqueeze(-2).repeat(1, v4.hc_mult, 1)  # [T_total, hc, dim]
    else:
        h = prepare_hidden_fn(input_ids=input_ids, positions=positions)
    if _rt_on:
        _rt.record("prefill_embed_hc_expanded", h)

    use_small_token_bypass = _small_token_bypass_enabled(
        v4, input_ids, prepare_hidden_fn, cp_ctx
    )
    graph_decision = None
    graph_replay_requested = _env_flag("DSV4_PREFILL_GRAPH_REPLAY", "0")
    graph_drop_kv_for_logits_only = _env_flag(
        "DSV4_PREFILL_GRAPH_DROP_KV_FOR_LOGITS_ONLY", "0"
    )
    graph_owned_kv_requested = _env_flag("DSV4_PREFILL_GRAPH_OWNED_KV", "0")
    graph_owned_kv = graph_owned_kv_requested and _env_flag(
        "DSV4_PREFILL_GRAPH_ALLOW_UNSAFE_OWNED_KV", "0"
    )
    graph_replay_block_reason = None
    if graph_replay_requested and graph_owned_kv_requested and not graph_owned_kv:
        graph_replay_block_reason = "owned_kv_requires_unsafe_ack"
        setattr(v4, "_last_prefill_graph_replay_error", graph_replay_block_reason)
        if _env_flag("DSV4_PREFILL_GRAPH_LOG_DECISION", "0"):
            _log_prefill_graph_line(
                "[DSV4PrefillGraph] stage=graph_replay "
                "enabled=False reason=owned_kv_requires_unsafe_ack"
            )
    if (
        graph_replay_requested
        and graph_replay_block_reason is None
        and kv_cache is not None
        and not graph_drop_kv_for_logits_only
        and not graph_owned_kv
        and not _env_flag("DSV4_PREFILL_GRAPH_ALLOW_LIVE_KV", "0")
    ):
        graph_replay_block_reason = "live_kv_not_static"
        setattr(v4, "_last_prefill_graph_replay_error", graph_replay_block_reason)
        if _env_flag("DSV4_PREFILL_GRAPH_LOG_DECISION", "0"):
            _log_prefill_graph_line(
                "[DSV4PrefillGraph] stage=graph_replay "
                "enabled=False reason=live_kv_not_static"
            )
    graph_static_bind_allowed = _env_flag(
        "DSV4_PREFILL_GRAPH_ALLOW_STATIC_EAGER", "0"
    ) or (graph_replay_requested and graph_replay_block_reason is None)
    graph_manager_enabled = (
        _env_flag("DSV4_PREFILL_GRAPH_MANAGER", "0")
        or _env_flag("DSV4_PREFILL_GRAPH_UPDATE_STATIC", "0")
        or (graph_replay_requested and graph_replay_block_reason is None)
    )
    if graph_manager_enabled:
        from rtp_llm.models_py.modules.dsv4.prefill_graph import (
            PrefillGraphRequest,
            select_prefill_graph_key,
        )

        token_count_for_graph = (
            int(getattr(cp_ctx, "seq_len_full"))
            if cp_ctx is not None and hasattr(cp_ctx, "seq_len_full")
            else int(input_ids.numel())
        )
        batch_size_for_graph = (
            int(cu_seqlens.numel() - 1)
            if cu_seqlens is not None and cu_seqlens.numel() >= 2
            else 1
        )
        (
            prefix_length_for_graph,
            max_prefix_length_for_graph,
            prefix_unknown_for_graph,
        ) = _graph_prefix_facts(cp_ctx, attn_inputs, batch_size=batch_size_for_graph)
        graph_decision = select_prefill_graph_key(
            PrefillGraphRequest(
                token_count=token_count_for_graph,
                batch_size=batch_size_for_graph,
                cp_size=int(getattr(cp_ctx, "cp_size", getattr(v4, "_cp_size", 1))),
                prefix_length=prefix_length_for_graph,
                max_prefix_length=max_prefix_length_for_graph,
                prefix_unknown=prefix_unknown_for_graph,
                prepare_hidden=prepare_hidden_fn is not None,
                cache_store=write_cache_store_impl is not None,
                mtp_hidden=getattr(v4, "_mtp_hidden_buffer", None) is not None,
            ),
            enabled=hasattr(v4, "layers")
            and not _rt.ENABLED
            and not _fwd_dbg.enabled(),
            token_buckets=os.environ.get("DSV4_PREFILL_GRAPH_BUCKETS", "512"),
            batch_buckets=os.environ.get("DSV4_PREFILL_GRAPH_BATCH_BUCKETS", "1"),
            prefix_buckets=os.environ.get("DSV4_PREFILL_GRAPH_PREFIX_BUCKETS", "0"),
            reuse_buckets=os.environ.get("DSV4_PREFILL_GRAPH_REUSE_BUCKETS", "0"),
            fixed_cp_size=_env_int("DSV4_PREFILL_GRAPH_CP_SIZE", 8),
            allow_prefix_reuse=_env_flag("DSV4_PREFILL_GRAPH_ALLOW_PREFIX_REUSE", "0"),
        )
        setattr(v4, "_last_prefill_graph_decision", graph_decision)
        if _env_flag("DSV4_PREFILL_GRAPH_LOG_DECISION", "0"):
            _log_prefill_graph_line(
                "[DSV4PrefillGraph] "
                "stage=decision "
                f"decision_enabled={getattr(graph_decision, 'enabled', None)} "
                f"reason={getattr(graph_decision, 'reason', None)} "
                f"key={getattr(graph_decision, 'key', None)}"
            )
    record_range_ctx = (
        _profiler.disable_record_function_ranges
        if use_small_token_bypass
        else nullcontext
    )

    # FP8 KV-cache: hoist host-side prefill metadata once per ratio bucket
    # and broadcast to every layer's ``AttentionFP8._prefill_meta_shared``.
    # BF16 path doesn't need this; ``Attention`` rebuilds meta inside its
    # own forward.
    meta_by_ratio = None
    live_meta_by_ratio = None
    live_input_ids = input_ids
    live_h = h
    live_positions = positions
    live_cu_seqlens = cu_seqlens
    live_block_tables_by_type = block_tables_by_type
    static_state_updated_this_forward = False
    static_args_bound_this_forward = False
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
            # active — the main + indexer + SWA CP gather/restore scratch
            # (dedicated buffer pairs per role, used by BOTH the serial and
            # overlap paths for the workspace-backed roles). Sizing is MAX
            # (capacity-bound, runtime-length-independent) so every forward
            # allocates the same-sized block → zero allocator fragmentation,
            # IDENTICAL across main and MTP-draft forwards (the draft overrides
            # ``_resolve_prefill_ws_gather_widths`` to size off the main model's
            # ratios — see ``deepseek_v4_mtp_model``). All three CP roles
            # (main / indexer / swa) are workspace-backed.
            #
            # ``reserve_cp`` gates the CP region; we cannot derive it from
            # ``compress_ratio != 0`` on the layers because the SWA gather runs
            # on EVERY attention layer (including the draft's single SWA layer).
            # The bound ``_prefill_ws_full_rows>0`` is the canonical signal that
            # CP is active at workspace bind time.
            reserve_cp = (cp_ctx is not None) and int(v4._prefill_ws_full_rows) > 0
            ws = PrefillWorkspace(
                input_ids.device,
                q_rows=v4._prefill_ws_q_rows,
                q_dim=v4._prefill_ws_q_dim,
                reserve_cp=reserve_cp,
                cp_rows=v4._prefill_ws_full_rows,
                main_w=v4._prefill_ws_main_w,
                idx_w=v4._prefill_ws_idx_w,
                swa_w=v4._prefill_ws_swa_w,
            )
            meta_by_ratio = build_and_propagate_prefill_meta_fp8(
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
            live_meta_by_ratio = meta_by_ratio
            if graph_decision is not None and (
                _env_flag("DSV4_PREFILL_GRAPH_UPDATE_STATIC", "0")
                or (graph_replay_requested and graph_replay_block_reason is None)
            ):
                from rtp_llm.models_py.modules.dsv4.prefill_graph import (
                    analyze_prefill_capture_surface,
                    try_update_static_prefill_graph_state,
                )

                graph_cu_seqlens = cu_seqlens
                if graph_cu_seqlens is None:
                    graph_cu_seqlens = torch.tensor(
                        [0, int(input_ids.numel())],
                        dtype=torch.int64,
                        device=input_ids.device,
                    )
                graph_input_lengths = input_lengths
                if graph_input_lengths is None:
                    graph_input_lengths = (
                        graph_cu_seqlens[1:].to(torch.int64)
                        - graph_cu_seqlens[:-1].to(torch.int64)
                    ).to(torch.int32)
                graph_prefix_lengths = prefix_lengths
                if graph_prefix_lengths is None:
                    graph_prefix_lengths = torch.zeros(
                        int(graph_input_lengths.numel()),
                        dtype=torch.int32,
                        device=input_ids.device,
                    )
                graph_req_id_per_token = req_id_per_token
                if graph_req_id_per_token is None:
                    graph_req_id_per_token = torch.repeat_interleave(
                        torch.arange(
                            int(graph_input_lengths.numel()),
                            device=input_ids.device,
                            dtype=torch.int32,
                        ),
                        graph_input_lengths.to(device=input_ids.device),
                    )
                block_cap = _env_int("DSV4_PREFILL_GRAPH_BLOCK_CAP", 0)
                if block_cap <= 0:
                    block_cap = 1
                    if block_tables_by_type:
                        observed_block_cols = [
                            int(table.size(1))
                            for table in block_tables_by_type.values()
                            if table.dim() == 2
                        ]
                        if observed_block_cols:
                            block_cap = max(observed_block_cols)
                graph_decision = try_update_static_prefill_graph_state(
                    v4,
                    graph_decision,
                    input_ids=input_ids,
                    hidden=h,
                    position_ids=positions,
                    req_id_per_token=graph_req_id_per_token,
                    cu_seqlens=graph_cu_seqlens,
                    input_lengths=graph_input_lengths,
                    prefix_lengths=graph_prefix_lengths,
                    block_tables_by_type=block_tables_by_type,
                    seq_len_full=token_count_for_graph,
                    prefix_length=prefix_length_for_graph,
                    meta_by_ratio=meta_by_ratio,
                    block_cap=block_cap,
                    workspace_config=dict(
                        q_rows=int(input_ids.numel()),
                        q_dim=v4._prefill_ws_q_dim,
                        reserve_cp=reserve_cp,
                        cp_rows=token_count_for_graph,
                        main_w=v4._prefill_ws_main_w,
                        idx_w=v4._prefill_ws_idx_w,
                        swa_w=v4._prefill_ws_swa_w,
                    ),
                )
                setattr(v4, "_last_prefill_graph_decision", graph_decision)
                state_after_update = getattr(v4, "_last_prefill_graph_state", None)
                static_state_updated_this_forward = bool(
                    graph_decision.enabled
                    and state_after_update is not None
                    and getattr(state_after_update, "valid", False)
                )
                if (
                    graph_decision.enabled
                    and (
                        graph_static_bind_allowed
                        and (
                            _env_flag("DSV4_PREFILL_GRAPH_BIND_STATIC_META", "0")
                            or graph_replay_requested
                        )
                    )
                    and getattr(v4, "_last_prefill_graph_state", None) is not None
                ):
                    state = getattr(v4, "_last_prefill_graph_state")
                    try:
                        meta_by_ratio = state.meta.materialize(
                            meta_by_ratio, workspace=state.workspace
                        )
                        for layer in v4.layers:
                            attn = getattr(layer, "attn", None)
                            if attn is None:
                                continue
                            attn._ensure_freqs_cis_bound()
                            attn._set_prefill_meta_shared(
                                meta_by_ratio.get(int(attn.compress_ratio))
                            )
                        setattr(v4, "_last_prefill_graph_bind_static_meta_error", None)
                        if _env_flag("DSV4_PREFILL_GRAPH_LOG_DECISION", "0"):
                            _log_prefill_graph_line(
                                "[DSV4PrefillGraph] "
                                "stage=bind_static_meta bound=True reason=ok"
                            )
                    except Exception as exc:
                        setattr(
                            v4,
                            "_last_prefill_graph_bind_static_meta_error",
                            str(exc),
                        )
                        if _env_flag("DSV4_PREFILL_GRAPH_LOG_DECISION", "0"):
                            _log_prefill_graph_line(
                                "[DSV4PrefillGraph] "
                                f"stage=bind_static_meta bound=False reason={exc}"
                            )
                if _env_flag("DSV4_PREFILL_GRAPH_CAPTURE_SURFACE_LOG", "0"):
                    state = getattr(v4, "_last_prefill_graph_state", None)
                    if state is not None:
                        report = analyze_prefill_capture_surface(
                            state,
                            input_ids=input_ids,
                            hidden=h,
                            position_ids=positions,
                            req_id_per_token=graph_req_id_per_token,
                            cu_seqlens=graph_cu_seqlens,
                            input_lengths=graph_input_lengths,
                            prefix_lengths=graph_prefix_lengths,
                            block_tables_by_type=block_tables_by_type,
                            meta_by_ratio=meta_by_ratio,
                        )
                        setattr(v4, "_last_prefill_graph_capture_surface", report)
                        _log_prefill_graph_line(
                            "[DSV4PrefillGraph] "
                            "stage=capture_surface "
                            f"static_bound={report.static_bound} "
                            f"live_tensor_count={report.live_tensor_count} "
                            f"static_bound_count={report.static_bound_count} "
                            f"live_not_static_count={len(report.live_not_static)} "
                            f"missing_static_count={len(report.missing_static)} "
                            f"skipped_critical_count={len(report.skipped_critical)} "
                            f"live_not_static_sample={report.live_not_static[:8]} "
                            f"missing_static_sample={report.missing_static[:8]} "
                            f"skipped_critical_sample={report.skipped_critical[:8]}"
                        )
            if _env_flag("DSV4_PREFILL_GRAPH_LOG_DECISION", "0"):
                state = getattr(v4, "_last_prefill_graph_state", None)
                log_line = (
                    "[DSV4PrefillGraph] "
                    "stage=post_meta "
                    f"decision_enabled={getattr(graph_decision, 'enabled', None)} "
                    f"reason={getattr(graph_decision, 'reason', None)} "
                    f"key={getattr(graph_decision, 'key', None)} "
                    f"state_valid={getattr(state, 'valid', None)} "
                    f"pointer_stable={getattr(state, 'pointer_stable', None)} "
                    "inventory_count="
                    f"{len(state.pointer_inventory()) if state is not None else None} "
                    "state_error="
                    f"{getattr(v4, '_last_prefill_graph_state_error', None)}"
                )
                _log_prefill_graph_line(log_line)

    if _prefill_graph_needs_bound_capture_surface():
        setattr(v4, "_last_prefill_graph_bound_capture_surface", None)

    if graph_decision is not None and (
        graph_static_bind_allowed
        and (
            _env_flag("DSV4_PREFILL_GRAPH_BIND_STATIC_INPUTS", "0")
            or graph_replay_requested
        )
    ):
        state = getattr(v4, "_last_prefill_graph_state", None)
        if state is None:
            setattr(v4, "_last_prefill_graph_bind_static_error", "missing_state")
            if _env_flag("DSV4_PREFILL_GRAPH_LOG_DECISION", "0"):
                _log_prefill_graph_line(
                    "[DSV4PrefillGraph] stage=bind_static "
                    "bound=False reason=missing_state"
                )
        else:
            try:
                from rtp_llm.models_py.modules.dsv4.prefill_graph import (
                    analyze_prefill_capture_surface,
                    exact_static_prefill_layer_loop_args,
                )

                static_args = exact_static_prefill_layer_loop_args(
                    state,
                    input_ids=input_ids,
                    hidden=h,
                    position_ids=positions,
                    cu_seqlens=cu_seqlens,
                    block_tables_by_type=block_tables_by_type,
                )
                input_ids = static_args.input_ids
                h = static_args.hidden
                positions = static_args.position_ids
                cu_seqlens = static_args.cu_seqlens
                block_tables_by_type = static_args.block_tables_by_type
                static_args_bound_this_forward = True
                setattr(v4, "_last_prefill_graph_bind_static_error", None)
                should_analyze_bound_surface = (
                    _prefill_graph_needs_bound_capture_surface()
                )
                if should_analyze_bound_surface:
                    report = analyze_prefill_capture_surface(
                        state,
                        input_ids=input_ids,
                        hidden=h,
                        position_ids=positions,
                        req_id_per_token=state.request.req_id_per_token,
                        cu_seqlens=cu_seqlens,
                        input_lengths=state.request.input_lengths,
                        prefix_lengths=state.request.prefix_lengths,
                        block_tables_by_type=block_tables_by_type,
                        meta_by_ratio=meta_by_ratio,
                    )
                    setattr(v4, "_last_prefill_graph_bound_capture_surface", report)
                    if _env_flag("DSV4_PREFILL_GRAPH_CAPTURE_SURFACE_LOG", "0"):
                        _log_prefill_graph_line(
                            "[DSV4PrefillGraph] "
                            "stage=bind_static_capture_surface "
                            f"static_bound={report.static_bound} "
                            f"live_tensor_count={report.live_tensor_count} "
                            f"static_bound_count={report.static_bound_count} "
                            f"live_not_static_count={len(report.live_not_static)} "
                            f"missing_static_count={len(report.missing_static)} "
                            f"skipped_critical_count={len(report.skipped_critical)} "
                            f"live_not_static_sample={report.live_not_static[:8]} "
                            f"missing_static_sample={report.missing_static[:8]} "
                            f"skipped_critical_sample={report.skipped_critical[:8]}"
                        )
                if _env_flag("DSV4_PREFILL_GRAPH_LOG_DECISION", "0"):
                    _log_prefill_graph_line(
                        "[DSV4PrefillGraph] stage=bind_static bound=True reason=ok"
                    )
            except Exception as exc:
                setattr(v4, "_last_prefill_graph_bind_static_error", str(exc))
                if _env_flag("DSV4_PREFILL_GRAPH_LOG_DECISION", "0"):
                    _log_prefill_graph_line(
                        "[DSV4PrefillGraph] stage=bind_static "
                        f"bound=False reason={exc}"
                    )

    fast_block_cls = None
    fast_attn_cls = None
    if use_small_token_bypass:
        from rtp_llm.models_py.modules.dsv4.block import Block
        from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8

        fast_block_cls = Block
        fast_attn_cls = AttentionFP8

    def _run_layer_loop(
        loop_h: torch.Tensor,
        *,
        loop_input_ids: torch.Tensor,
        loop_positions: torch.Tensor,
        loop_cu_seqlens: torch.Tensor,
        loop_kv_cache: Optional[KVCache],
        loop_block_tables_by_type: Optional[Dict[int, torch.Tensor]],
    ) -> torch.Tensor:
        split_fast_loop = _env_flag("DSV4_PREFILL_SPLIT_FAST_LOOP", "0")
        request_token_count = _prefill_request_token_count(loop_input_ids, cp_ctx)
        pre_island_enabled = split_fast_loop and _env_flag(
            "DSV4_PREFILL_ISLAND_GRAPH", "0"
        )
        bridge_island_enabled = (
            split_fast_loop
            and _env_flag("DSV4_PREFILL_ISLAND_BRIDGE", "0")
            and write_cache_store_impl is None
            and not _rt_on
        )
        bridge_qkv_enabled = bridge_island_enabled and _env_flag(
            "DSV4_PREFILL_ISLAND_QKV_BRIDGE", "0"
        )
        bridge_q_enabled = bridge_qkv_enabled and _env_flag(
            "DSV4_PREFILL_ISLAND_Q_BRIDGE", "0"
        )
        if bridge_q_enabled:
            q_bridge_max_tokens = _env_int_fail_closed(
                "DSV4_PREFILL_ISLAND_Q_BRIDGE_MAX_TOKENS", 512
            )
            max_shapes = _env_int_fail_closed("DSV4_PREFILL_ISLAND_GRAPH_MAX_SHAPES", 1)
            bridge_q_enabled = (
                q_bridge_max_tokens is not None
                and q_bridge_max_tokens > 0
                and request_token_count <= q_bridge_max_tokens
                and max_shapes == 1
            )
        attn_ffn_bridge_enabled = bridge_island_enabled and _env_flag(
            "DSV4_PREFILL_ISLAND_ATTN_FFN_BRIDGE", "0"
        )
        if attn_ffn_bridge_enabled:
            attn_ffn_bridge_max_tokens = _env_int_fail_closed(
                "DSV4_PREFILL_ISLAND_ATTN_FFN_BRIDGE_MAX_TOKENS", 512
            )
            max_shapes = _env_int_fail_closed("DSV4_PREFILL_ISLAND_GRAPH_MAX_SHAPES", 1)
            attn_ffn_bridge_enabled = (
                attn_ffn_bridge_max_tokens is not None
                and attn_ffn_bridge_max_tokens > 0
                and request_token_count <= attn_ffn_bridge_max_tokens
                and max_shapes == 1
            )
        island_graph = None
        island_graph_enabled = pre_island_enabled or bridge_island_enabled
        if not island_graph_enabled and hasattr(v4, "_prefill_island_graph_manager"):
            delattr(v4, "_prefill_island_graph_manager")
        if island_graph_enabled:
            island_graph = getattr(v4, "_prefill_island_graph_manager", None)
            if island_graph is None:
                from rtp_llm.models_py.modules.dsv4.prefill_island_graph import (
                    PrefillIslandGraphManager,
                )

                island_graph = PrefillIslandGraphManager()
                setattr(v4, "_prefill_island_graph_manager", island_graph)
        pending_attn_pre = None
        pending_local_qkv = None
        if _env_flag("DSV4_PREFILL_GRAPH_ATTN_BODY_SHADOW", "0"):
            _clear_prefill_graph_attn_body_shadow_result(v4)

        def _run_attn_body_shadow(
            *,
            layer_idx: int,
            layer,
            x_pre: torch.Tensor,
            prefill_local_qkv,
        ):
            target_layer = _prefill_graph_attn_body_shadow_layer()
            if target_layer is None or int(layer_idx) != int(target_layer):
                return None
            _clear_prefill_graph_attn_body_shadow_result(v4)
            if not _prefill_graph_attn_body_shadow_allowed(
                v4,
                graph_decision,
                layer_idx=layer_idx,
                loop_kv_cache=loop_kv_cache,
                static_state_updated_this_forward=static_state_updated_this_forward,
                graph_replay_requested=graph_replay_requested,
                write_cache_store_impl=write_cache_store_impl,
                rt_on=_rt_on,
            ):
                return None
            if x_pre.device.type != "cuda" or not torch.cuda.is_available():
                return None
            if torch.cuda.is_current_stream_capturing():
                return None
            state = getattr(v4, "_last_prefill_graph_state", None)
            entered_graph = False
            try:
                graph_kv_cache = state.ensure_graph_kv_cache(
                    loop_kv_cache,
                    loop_block_tables_by_type,
                    min_block_cap=_env_int(
                        "DSV4_PREFILL_GRAPH_ATTN_BODY_SHADOW_KV_BLOCK_CAP", 64
                    ),
                )
                if not state.graph_kv_fits(loop_block_tables_by_type):
                    raise RuntimeError("attn_body_shadow_graph_kv_block_overflow")
                torch.cuda.synchronize(x_pre.device)
                graph = torch.cuda.CUDAGraph()
                entered_graph = True
                with torch.cuda.graph(graph):
                    shadow_out = layer.prefill_fast_attn_body(
                        x_pre,
                        loop_positions,
                        kv_cache=graph_kv_cache,
                        block_tables_by_type=loop_block_tables_by_type,
                        prefill_local_qkv=prefill_local_qkv,
                    )
                graph.replay()
                torch.cuda.synchronize(x_pre.device)
                return shadow_out
            except Exception as exc:
                _set_prefill_graph_attn_body_shadow_error(v4, str(exc))
                if _env_flag("DSV4_PREFILL_GRAPH_LOG_DECISION", "0"):
                    _log_prefill_graph_line(
                        "[DSV4PrefillGraph] stage=attn_body_shadow "
                        f"enabled=False layer={layer_idx} reason={exc}"
                    )
                if entered_graph:
                    raise RuntimeError(
                        f"attn_body_shadow_graph_failed:{exc}"
                    ) from exc
                return None
            finally:
                _release_prefill_graph_shadow_kv(state)

        def _compare_attn_body_shadow(
            *,
            layer_idx: int,
            shadow_out,
            eager_out: torch.Tensor,
        ) -> None:
            if shadow_out is None:
                return
            _compare_prefill_graph_attn_body_shadow_tensors(
                v4,
                layer_idx=layer_idx,
                shadow_out=shadow_out,
                eager_out=eager_out,
            )

        for layer_idx, layer in enumerate(v4.layers):
            if use_small_token_bypass and isinstance(layer, fast_block_cls):
                if split_fast_loop and isinstance(
                    getattr(layer, "attn", None), fast_attn_cls
                ):
                    attn_pre = None
                    if pending_attn_pre is not None:
                        residual, x_pre, post, comb = pending_attn_pre
                        pending_attn_pre = None
                        prefill_local_qkv = pending_local_qkv
                        pending_local_qkv = None
                    elif pre_island_enabled and island_graph is not None:
                        attn_pre = island_graph.run_pre(
                            layer=layer,
                            kind="attn_pre",
                            x=loop_h,
                            eager_fn=layer.prefill_fast_attn_pre,
                        )
                        residual = loop_h
                        if attn_pre is None:
                            residual, x_pre, post, comb = layer.prefill_fast_attn_pre(
                                loop_h
                            )
                            prefill_local_qkv = None
                        else:
                            x_pre, post, comb = attn_pre
                            prefill_local_qkv = None
                    else:
                        residual, x_pre, post, comb = layer.prefill_fast_attn_pre(
                            loop_h
                        )
                        prefill_local_qkv = None
                    attn_body_shadow_out = _run_attn_body_shadow(
                        layer_idx=layer_idx,
                        layer=layer,
                        x_pre=x_pre,
                        prefill_local_qkv=prefill_local_qkv,
                    )
                    attn_out = layer.prefill_fast_attn_body(
                        x_pre,
                        loop_positions,
                        kv_cache=loop_kv_cache,
                        block_tables_by_type=loop_block_tables_by_type,
                        prefill_local_qkv=prefill_local_qkv,
                    )
                    _compare_attn_body_shadow(
                        layer_idx=layer_idx,
                        shadow_out=attn_body_shadow_out,
                        eager_out=attn_out,
                    )
                    attn_ffn_bridge = None
                    if attn_ffn_bridge_enabled and island_graph is not None:
                        attn_ffn_bridge = island_graph.run_attn_post_ffn_pre_bridge(
                            layer=layer,
                            attn_out=attn_out,
                            residual=residual,
                            post=post,
                            comb=comb,
                        )
                    if attn_ffn_bridge is None:
                        loop_h = layer.prefill_fast_attn_post(
                            attn_out, residual, post, comb
                        )
                        ffn_pre = None
                        if pre_island_enabled and island_graph is not None:
                            ffn_pre = island_graph.run_pre(
                                layer=layer,
                                kind="ffn_pre",
                                x=loop_h,
                                eager_fn=layer.prefill_fast_ffn_pre,
                            )
                        residual = loop_h
                        if ffn_pre is None:
                            residual, x_pre, post, comb = layer.prefill_fast_ffn_pre(
                                loop_h
                            )
                        else:
                            x_pre, post, comb = ffn_pre
                    else:
                        residual, x_pre, post, comb = attn_ffn_bridge
                    ffn_out = layer.prefill_fast_ffn_body(x_pre, loop_input_ids)
                    next_layer = (
                        v4.layers[layer_idx + 1]
                        if layer_idx + 1 < len(v4.layers)
                        else None
                    )
                    bridge = None
                    next_common = None
                    if (
                        bridge_island_enabled
                        and island_graph is not None
                        and next_layer is not None
                        and isinstance(next_layer, fast_block_cls)
                        and isinstance(getattr(next_layer, "attn", None), fast_attn_cls)
                    ):
                        if bridge_qkv_enabled:
                            next_common = getattr(
                                next_layer.attn, "_prefill_meta_shared", None
                            )
                        bridge = island_graph.run_ffn_post_attn_pre_bridge(
                            current_layer=layer,
                            next_layer=next_layer,
                            ffn_out=ffn_out,
                            residual=residual,
                            post=post,
                            comb=comb,
                            common=next_common,
                            include_qkv=bool(next_common is not None),
                            include_q=bool(next_common is not None and bridge_q_enabled),
                        )
                    if bridge is None:
                        loop_h = layer.prefill_fast_ffn_post(
                            ffn_out, residual, post, comb
                        )
                    else:
                        bridge_h, next_x_pre, next_post, next_comb = bridge[:4]
                        loop_h = bridge_h
                        pending_attn_pre = (
                            bridge_h,
                            next_x_pre,
                            next_post,
                            next_comb,
                        )
                        pending_local_qkv = bridge[4:] if len(bridge) in (6, 7) else None
                else:
                    loop_h = layer.forward_prefill_fast(
                        loop_h,  # [T, hc, dim]
                        loop_input_ids,  # [T]
                        loop_positions,  # [T]
                        loop_cu_seqlens,  # [B+1]
                        kv_cache=loop_kv_cache,
                        block_tables_by_type=loop_block_tables_by_type,
                    )  # [T, hc, dim]
            else:
                loop_h = layer(
                    loop_h,  # [T, hc, dim]
                    loop_input_ids,  # [T]
                    loop_positions,  # [T]
                    loop_cu_seqlens,  # [B+1]
                    kv_cache=loop_kv_cache,
                    block_tables_by_type=loop_block_tables_by_type,
                )  # [T, hc, dim]
            if _rt_on:
                _rt.record(f"prefill_layer{layer_idx:02d}_out", loop_h)
            if write_cache_store_impl is not None:
                write_cache_store_impl(kv_cache.get_layer_caches(layer_idx))
            if _rt_on:
                _rt.record(f"layer{layer_idx:02d}_out", loop_h)
                if cp_ctx is None:
                    layer_last = loop_h[-1:].contiguous()
                else:
                    layer_last_pos = cp_ctx.seq_len_total - 1
                    layer_last_mask = (
                        cp_ctx.global_positions == layer_last_pos
                    ) & cp_ctx.local_is_real
                    layer_last = loop_h[layer_last_mask].contiguous()
                    dbg_pos = getattr(_rt, "_DBG_GLOBAL_POS", -1)
                    if dbg_pos >= 0:
                        layer_pos_mask = (
                            cp_ctx.global_positions == dbg_pos
                        ) & cp_ctx.local_is_real
                        _rt.record(
                            f"layer{layer_idx:02d}_pos{dbg_pos}",
                            loop_h[layer_pos_mask].contiguous(),
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
                        loop_h[layer_tail_mask].contiguous(),
                    )
                _rt.record(f"layer{layer_idx:02d}_last", layer_last)
        return loop_h

    def _clear_static_prefill_meta_refs() -> None:
        # Always drop the per-layer ``common.workspace`` references, even if a
        # layer raises mid-prefill (e.g. a CUDA OOM under memory pressure — the
        # exact case this per-forward workspace exists to relieve). The ref lives
        # on each layer's ``_prefill_meta_shared`` (a persistent module attr), so
        # without this the ~16 GiB workspace would stay pinned past the failing
        # forward and starve the retry / next request on a near-full card. ``clear``
        # is idempotent (sets None per layer).
        if v4.fp8_kv_cache:
            clear_prefill_meta_shared_fp8(v4)

    def _bind_prefill_meta_refs(meta) -> None:
        if not v4.fp8_kv_cache or meta is None:
            return
        for layer in v4.layers:
            attn = getattr(layer, "attn", None)
            if attn is None:
                continue
            attn._ensure_freqs_cis_bound()
            attn._set_prefill_meta_shared(meta.get(int(attn.compress_ratio)))

    def _cp_prefill_sync_pending_for_graph() -> bool:
        if os.environ.get("DSV4_CP_SYNC_AFTER_ATTN_ONCE", "1") == "0":
            return False
        if cp_ctx is None:
            return False
        for layer in v4.layers:
            if getattr(layer, "_cp_sync_after_attn_done", True):
                continue
            ffn_strategy = getattr(getattr(layer, "ffn", None), "_strategy", None)
            if getattr(ffn_strategy, "name", "") == "mega":
                return True
        return False

    def _try_run_static_prefill_layer_loop_graph(
        loop_h: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        def _skip(reason: str) -> None:
            setattr(v4, "_last_prefill_graph_replay_error", reason)
            if _env_flag("DSV4_PREFILL_GRAPH_LOG_DECISION", "0"):
                _log_prefill_graph_line(
                    "[DSV4PrefillGraph] stage=graph_replay "
                    f"enabled=False reason={reason}"
                )

        if not _env_flag("DSV4_PREFILL_GRAPH_REPLAY", "0"):
            if _env_flag("DSV4_PREFILL_GRAPH_LOG_DECISION", "0"):
                _log_prefill_graph_line(
                    "[DSV4PrefillGraph] stage=graph_try_entry replay_env=False"
                )
            return None
        if _env_flag("DSV4_PREFILL_GRAPH_LOG_DECISION", "0"):
            _log_prefill_graph_line("[DSV4PrefillGraph] stage=graph_try_entry")
        if graph_replay_block_reason is not None:
            _skip(graph_replay_block_reason)
            return None
        state = getattr(v4, "_last_prefill_graph_state", None)
        if graph_decision is None:
            _skip("missing_decision")
            return None
        if not graph_decision.enabled:
            _skip(f"decision_disabled:{graph_decision.reason}")
            return None
        if state is None:
            _skip("missing_state")
            return None
        if not getattr(state, "valid", False):
            _skip("state_invalid")
            return None
        if not use_small_token_bypass or write_cache_store_impl is not None or _rt_on:
            _skip("unsupported_forward_context")
            return None
        if (
            kv_cache is not None
            and not graph_drop_kv_for_logits_only
            and not graph_owned_kv
            and not _env_flag("DSV4_PREFILL_GRAPH_ALLOW_LIVE_KV", "0")
        ):
            _skip("live_kv_not_static")
            return None
        if graph_owned_kv and kv_cache is not None:
            if getattr(graph_decision.key, "prefix_bucket", 0) != 0 or getattr(
                graph_decision.key, "reuse_bucket", 0
            ) != 0:
                setattr(
                    v4,
                    "_last_prefill_graph_replay_error",
                    "graph_owned_kv_prefix_reuse_unsupported",
                )
                if _env_flag("DSV4_PREFILL_GRAPH_LOG_DECISION", "0"):
                    _log_prefill_graph_line(
                        "[DSV4PrefillGraph] stage=graph_replay "
                        "enabled=False reason=graph_owned_kv_prefix_reuse_unsupported"
                    )
                return None
            try:
                graph_kv_cache = state.ensure_graph_kv_cache(
                    kv_cache,
                    block_tables_by_type,
                    min_block_cap=_env_int("DSV4_PREFILL_GRAPH_KV_BLOCK_CAP", 0),
                )
            except Exception as exc:
                _skip(f"graph_kv_prepare_failed:{exc}")
                return None
            if not state.graph_kv_fits(block_tables_by_type):
                state.reset_cuda_graph("graph_kv_block_overflow")
                _skip("graph_kv_block_overflow")
                return None
            graph_block_tables_by_type = block_tables_by_type
        else:
            graph_kv_cache = (
                _MetadataOnlyKVCache(kv_cache)
                if graph_drop_kv_for_logits_only and kv_cache is not None
                else kv_cache
            )
            graph_block_tables_by_type = (
                None if graph_drop_kv_for_logits_only else block_tables_by_type
            )
        if loop_h is not state.hidden:
            _skip("hidden_not_static")
            return None
        if loop_h.device.type != "cuda" or not torch.cuda.is_available():
            _skip("not_cuda")
            return None
        if torch.cuda.is_current_stream_capturing():
            _skip("already_capturing")
            return None
        if _cp_prefill_sync_pending_for_graph():
            _skip("cp_sync_pending")
            return None

        def _record_graph_output_stats(stage: str) -> None:
            if not _env_flag("DSV4_PREFILL_GRAPH_OUTPUT_STATS", "0"):
                return
            torch.cuda.synchronize(state.output_hidden.device)
            out_f = state.output_hidden.detach().float()
            nonzero = int(torch.count_nonzero(out_f).item())
            max_abs = float(out_f.abs().max().item()) if out_f.numel() else 0.0
            mean_abs = float(out_f.abs().mean().item()) if out_f.numel() else 0.0
            stats = {
                "stage": stage,
                "numel": int(out_f.numel()),
                "nonzero": nonzero,
                "max_abs": max_abs,
                "mean_abs": mean_abs,
            }
            setattr(v4, "_last_prefill_graph_output_stats", stats)
            _log_prefill_graph_line(
                "[DSV4PrefillGraph] stage=output_stats "
                f"mode={stage} numel={stats['numel']} "
                f"nonzero={stats['nonzero']} "
                f"max_abs={stats['max_abs']:.9g} "
                f"mean_abs={stats['mean_abs']:.9g}"
            )

        ran_graph = False
        entered_graph = False
        try:
            from rtp_llm.models_py.modules.dsv4.prefill_graph import (
                analyze_prefill_capture_surface,
            )

            report = analyze_prefill_capture_surface(
                state,
                input_ids=input_ids,
                hidden=loop_h,
                position_ids=positions,
                req_id_per_token=state.request.req_id_per_token,
                cu_seqlens=cu_seqlens,
                input_lengths=state.request.input_lengths,
                prefix_lengths=state.request.prefix_lengths,
                block_tables_by_type=block_tables_by_type,
                meta_by_ratio=meta_by_ratio,
            )
            setattr(v4, "_last_prefill_graph_replay_capture_surface", report)
            if not report.static_bound:
                _skip(
                    "not_static_bound:"
                    f"live_not_static={report.live_not_static[:4]}:"
                    f"missing={report.missing_static[:4]}:"
                    f"skipped={report.skipped_critical[:4]}"
                )
                return None

            graph = getattr(state, "cuda_graph", None)
            if graph is not None:
                entered_graph = True
                graph.replay()
                state.graph_replay_count += 1
                ran_graph = True
                if graph_owned_kv and kv_cache is not None:
                    try:
                        copied = state.copy_graph_kv_to_live(
                            kv_cache, block_tables_by_type
                        )
                        setattr(v4, "_last_prefill_graph_kv_copy_count", copied)
                    except Exception as copy_exc:
                        state.reset_cuda_graph(f"graph_kv_copy_failed:{copy_exc}")
                        _skip(f"graph_kv_copy_failed:{copy_exc}")
                        raise RuntimeError(f"graph_kv_copy_failed:{copy_exc}") from copy_exc
                setattr(v4, "_last_prefill_graph_replay_error", None)
                if _env_flag("DSV4_PREFILL_GRAPH_LOG_DECISION", "0"):
                    _log_prefill_graph_line(
                        "[DSV4PrefillGraph] stage=graph_replay "
                        f"enabled=True mode=replay count={state.graph_replay_count}"
                    )
                _record_graph_output_stats("replay")
                return state.output_hidden

            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            entered_graph = True
            with torch.cuda.graph(graph):
                graph_out = _run_layer_loop(
                    state.hidden,
                    loop_input_ids=input_ids,
                    loop_positions=positions,
                    loop_cu_seqlens=cu_seqlens,
                    loop_kv_cache=graph_kv_cache,
                    loop_block_tables_by_type=graph_block_tables_by_type,
                )
                state.output_hidden.copy_(graph_out)
            torch.cuda.synchronize()
            state.mark_cuda_graph_captured(graph)
            ran_graph = True
            if graph_owned_kv and kv_cache is not None:
                try:
                    copied = state.copy_graph_kv_to_live(kv_cache, block_tables_by_type)
                    setattr(v4, "_last_prefill_graph_kv_copy_count", copied)
                except Exception as copy_exc:
                    state.reset_cuda_graph(f"graph_kv_copy_failed:{copy_exc}")
                    _skip(f"graph_kv_copy_failed:{copy_exc}")
                    raise RuntimeError(f"graph_kv_copy_failed:{copy_exc}") from copy_exc
            setattr(v4, "_last_prefill_graph_replay_error", None)
            if _env_flag("DSV4_PREFILL_GRAPH_LOG_DECISION", "0"):
                _log_prefill_graph_line(
                    "[DSV4PrefillGraph] stage=graph_replay "
                    "enabled=True mode=capture count="
                    f"{state.graph_capture_count}"
                )
            _record_graph_output_stats("capture")
            return state.output_hidden
        except Exception as exc:
            state.reset_cuda_graph(str(exc))
            setattr(v4, "_last_prefill_graph_replay_error", str(exc))
            if _env_flag("DSV4_PREFILL_GRAPH_LOG_DECISION", "0"):
                _log_prefill_graph_line(
                    "[DSV4PrefillGraph] stage=graph_replay "
                    f"enabled=False reason={exc}"
                )
            if entered_graph:
                raise
            return None
        finally:
            if ran_graph:
                _clear_static_prefill_meta_refs()

    def _run_static_prefill_copy_shadow_graph(loop_h: torch.Tensor) -> None:
        _clear_prefill_graph_copy_shadow_result(v4)
        if not _prefill_graph_copy_shadow_allowed(
            v4,
            graph_decision,
            static_args_bound_this_forward=static_args_bound_this_forward,
            static_state_updated_this_forward=static_state_updated_this_forward,
            use_small_token_bypass=use_small_token_bypass,
            write_cache_store_impl=write_cache_store_impl,
            rt_on=_rt_on,
        ):
            return
        state = getattr(v4, "_last_prefill_graph_state", None)
        if state is None or loop_h is not state.hidden:
            return
        if loop_h.device.type != "cuda" or not torch.cuda.is_available():
            return
        if torch.cuda.is_current_stream_capturing():
            return
        try:
            signature = (
                tuple(int(dim) for dim in state.hidden.shape),
                str(state.hidden.dtype),
                str(state.hidden.device),
                int(state.hidden.data_ptr()),
                int(state.output_hidden.data_ptr()),
            )
            graph = getattr(state, "_copy_shadow_graph", None)
            graph_signature = getattr(state, "_copy_shadow_graph_signature", None)
            mode = "replay"
            if graph is None or graph_signature != signature:
                torch.cuda.synchronize(state.hidden.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    state.output_hidden.copy_(state.hidden)
                torch.cuda.synchronize(state.hidden.device)
                setattr(state, "_copy_shadow_graph", graph)
                setattr(state, "_copy_shadow_graph_signature", signature)
                setattr(state, "_copy_shadow_graph_capture_count", 1)
                setattr(state, "_copy_shadow_graph_replay_count", 0)
                graph.replay()
                setattr(state, "_copy_shadow_graph_replay_count", 1)
                mode = "capture_replay"
            else:
                graph.replay()
                replay_count = int(
                    getattr(state, "_copy_shadow_graph_replay_count", 0)
                )
                setattr(state, "_copy_shadow_graph_replay_count", replay_count + 1)
            torch.cuda.synchronize(state.hidden.device)
            exact = bool(torch.equal(state.output_hidden, state.hidden))
            diff = (
                state.output_hidden.detach().float() - state.hidden.detach().float()
            ).abs()
            max_abs = float(diff.max().item()) if diff.numel() else 0.0
            mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
            if _env_flag("DSV4_PREFILL_GRAPH_LOG_DECISION", "0"):
                _log_prefill_graph_line(
                    "[DSV4PrefillGraph] stage=copy_shadow "
                    f"enabled=True mode={mode} exact={exact} "
                    f"max_abs={max_abs:.9g} mean_abs={mean_abs:.9g} "
                    "capture_count="
                    f"{getattr(state, '_copy_shadow_graph_capture_count', 0)} "
                    "replay_count="
                    f"{getattr(state, '_copy_shadow_graph_replay_count', 0)}"
                )
            _set_prefill_graph_copy_shadow_stats(
                v4,
                mode=mode,
                exact=exact,
                max_abs=max_abs,
                mean_abs=mean_abs,
            )
        except Exception as exc:
            _set_prefill_graph_copy_shadow_error(v4, str(exc))
            if _env_flag("DSV4_PREFILL_GRAPH_LOG_DECISION", "0"):
                _log_prefill_graph_line(
                    "[DSV4PrefillGraph] stage=copy_shadow "
                    f"enabled=False reason={exc}"
                )

    if _env_flag("DSV4_PREFILL_GRAPH_LOG_DECISION", "0"):
        _log_prefill_graph_line(
            "[DSV4PrefillGraph] stage=before_graph_try "
            f"replay_env={_env_flag('DSV4_PREFILL_GRAPH_REPLAY', '0')} "
            f"graph_decision={getattr(graph_decision, 'enabled', None)} "
            f"state_valid={getattr(getattr(v4, '_last_prefill_graph_state', None), 'valid', None)} "
            f"small_bypass={use_small_token_bypass}"
        )
    _clear_prefill_graph_attn_body_shadow_result_if_enabled(v4)
    _run_static_prefill_copy_shadow_graph(h)
    graph_h = _try_run_static_prefill_layer_loop_graph(h)
    static_eager_diag = _prefill_graph_static_eager_run_allowed(
        v4,
        graph_decision,
        kv_cache=kv_cache,
        static_state_updated_this_forward=static_state_updated_this_forward,
        graph_static_bind_allowed=graph_static_bind_allowed,
        graph_replay_requested=graph_replay_requested,
        use_small_token_bypass=use_small_token_bypass,
        write_cache_store_impl=write_cache_store_impl,
        rt_on=_rt_on,
    )
    if graph_h is not None:
        h = graph_h
    else:
        if static_eager_diag:
            if _env_flag("DSV4_PREFILL_GRAPH_LOG_DECISION", "0"):
                _log_prefill_graph_line(
                    "[DSV4PrefillGraph] stage=static_eager "
                    "enabled=True reason=diagnostic_static_buffers"
                )
        elif (
            graph_replay_requested
            or _env_flag("DSV4_PREFILL_GRAPH_UPDATE_STATIC", "0")
            or static_args_bound_this_forward
        ):
            input_ids = live_input_ids
            h = live_h
            positions = live_positions
            cu_seqlens = live_cu_seqlens
            block_tables_by_type = live_block_tables_by_type
            meta_by_ratio = live_meta_by_ratio
            if _env_flag("DSV4_PREFILL_GRAPH_STATIC_EAGER_RUN", "0") and _env_flag(
                "DSV4_PREFILL_GRAPH_LOG_DECISION", "0"
            ):
                _log_prefill_graph_line(
                    "[DSV4PrefillGraph] stage=static_eager "
                    "enabled=False reason=gate_failed"
                )
        with record_range_ctx():
            try:
                _bind_prefill_meta_refs(meta_by_ratio)
                h = _run_layer_loop(
                    h,
                    loop_input_ids=input_ids,
                    loop_positions=positions,
                    loop_cu_seqlens=cu_seqlens,
                    loop_kv_cache=kv_cache,
                    loop_block_tables_by_type=block_tables_by_type,
                )
            finally:
                _clear_static_prefill_meta_refs()

    if v4._mtp_hidden_buffer is not None:
        _pre_hc_flat = h.flatten(-2)
        v4._write_mtp_hidden_buffer(_pre_hc_flat, is_cuda_graph=False)
        if v4._mtp_last_hidden_buffer is not None:
            _last_pre_hc = _last_hidden_by_request(_pre_hc_flat, cu_seqlens, cp_ctx)
            v4._write_mtp_last_hidden_buffer(_last_pre_hc)

    # _hc_head_reduce is flat-native: [T, hc, dim] -> [T, dim].
    # Framework ``RMSNorm`` expects 2D, which matches the [T, dim] shape here.
    with record_range_ctx():
        if use_small_token_bypass and hasattr(v4.head_hc, "_head_impl"):
            h = v4.head_hc._head_impl(h)  # [T, dim]
        else:
            h = v4._hc_head_reduce(h)  # [T, dim]
        if _rt_on:
            _rt.record("prefill_hc_reduced", h)
        h = v4.norm(h)  # [T, dim]
    if _rt_on:
        _rt.record("prefill_final_norm", h)

    if _rt_on:
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

    Pulls flat metadata off :attr:`PyModelInputs.attention_inputs` directly:

    * ``positions``  = ``attn.position_ids``     — ``[T_total]`` int32 global pos
    * ``cu_seqlens`` = ``attn.cu_seqlens``       — ``[B+1]`` int32 prefix sum
    * ``block_tables_by_type`` = Dict[attn_type, [B, max_blocks]] — full-batch
      block tables (B axis = request axis), built via
      :func:`build_block_tables_batched`.

    Downstream (``block.py`` / ``attention.py`` / ``compressor.py`` /
    ``indexer.py``) consumes the cu_seqlens-aware metadata directly; under CP
    the per-layer setup swaps in CPContext's request-absolute positions and
    full-length write-side view.

    Returns ``PyModelOutputs`` with ``[T_total, dim]`` pre-lm-head hidden.
    """
    attn = inputs.attention_inputs

    # Context-Parallel setup must precede the per-layer loop because
    # forward_layers reads v4._cp_info to build the CP context.
    set_cp_info(v4, parallelism_config, attn, is_prefill=True)

    input_ids: torch.Tensor = inputs.input_ids  # [T_total] flat 1D

    # Framework already populates these — don't recompute.
    #  * ``attn.cu_seqlens``   : [B+1]     per-request cumulative prefix sum
    #  * ``attn.position_ids`` : [T_total] per-token global absolute position
    cu_seqlens = attn.cu_seqlens
    positions = attn.position_ids
    # warmup / cudagraph capture path doesn't populate position_ids —
    # synthesize from (prefix_lengths, input_lengths). Prefer ``_d`` (GPU)
    # variants when available: during cudagraph capture, the host-side
    # ``input_lengths`` / ``prefix_lengths`` are pinned int32 CPU tensors,
    # but a dtype-converting ``.to(device=..., dtype=int64)`` on a pinned
    # tensor produces an unpinned intermediate which capture rejects.
    if positions is None:
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

    block_tables_by_type = build_block_tables_batched(kv_cache, attn)

    hidden = forward_layers(
        v4,
        kv_cache,
        input_ids,
        positions,
        cu_seqlens,
        block_tables_by_type,
        attn_inputs=attn,
        prepare_hidden_fn=prepare_hidden_fn,
    )  # [T_total, dim]
    return PyModelOutputs(hidden)
