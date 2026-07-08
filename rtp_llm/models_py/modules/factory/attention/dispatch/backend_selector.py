"""Startup-time decode backend selection orchestration (GPU + NCCL).

Called by the engine per bs during capture: for that bs, real-machine benchmark
the candidate backends intersected with support (going through the real capture
path, see backend_bench), and pick the winner by the aggregation criterion.
Only rank0 benchmarks and selects, then broadcasts the winner's registry index
to Group.TP so every rank gets the same winner; otherwise each rank would pick
a different backend due to noise, bake different graphs, and NCCL would hang.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from rtp_llm.models_py.modules.factory.attention.dispatch import backend_bench
from rtp_llm.models_py.modules.factory.attention.dispatch.selector import (
    STABLE_CLUSTER_MARGIN,
    STABLE_THRESHOLD,
    Selector,
    kv_grid,
    select_min_mean,
    select_stable,
)

logger = logging.getLogger(__name__)


def _decode_registry() -> List[str]:
    """All-rank ordered registry class names (position is identity); DECODE_MHA_IMPS is already registered per device at import time."""
    from rtp_llm.models_py.modules.factory.attention.attn_factory import DECODE_MHA_IMPS

    return [c.__name__ for c in DECODE_MHA_IMPS]


def _tp_geometry(parallelism_config):
    """(tp_size, is_tp_root, src_global_rank). The TP group = the tp_size ranks with the same dp_rank,
    the global root = dp_rank*tp_size (broadcast src is a global rank, not literal 0).
    """
    tp_size = int(parallelism_config.tp_size)
    tp_rank = int(parallelism_config.tp_rank)
    dp_rank = int(parallelism_config.dp_rank)
    return tp_size, (tp_rank == 0), dp_rank * tp_size


def _eligible(
    attn_configs, attn_inputs, parallelism_config, fmha_config=None
) -> List[str]:
    import os as _os

    from rtp_llm.models_py.modules.factory.attention.attn_factory import (
        DECODE_MHA_IMPS,
        _is_fmha_impl_disabled,
    )

    excl = set(filter(None, _os.environ.get("DYN_DECODE_EXCLUDE", "").split(",")))
    names: List[str] = []
    for impl in DECODE_MHA_IMPS:
        name = impl.__name__
        if name in excl:
            continue
        # Respect fmha_config disable flags (disable_flash_infer / enable_xqa / ...),
        # mirroring the fixed-priority get_fmha_impl path; without this, dynamic
        # selection could benchmark and pick a backend the user explicitly disabled.
        if _is_fmha_impl_disabled(name, fmha_config):
            continue
        try:
            if not impl.support(attn_configs, attn_inputs):
                continue
            if not impl.support_parallelism_config(parallelism_config):
                continue
            # 3-arg unconditionally (all impls take parallelism_config=None); a
            # TypeError fallback would mask a real internal TypeError, see instantiate_decode_impl.
            inst = impl(attn_configs, attn_inputs, parallelism_config)
            if not inst.support_cuda_graph():
                continue
            del inst
        except Exception as e:
            logger.warning(
                "[dispatcher] %s support probe exception, excluded: %r", name, e
            )
            continue
        names.append(name)
    return names


def _fallback(
    attn_configs, attn_inputs, fmha_config, max_seq_len, parallelism_config
) -> Optional[str]:
    """Fixed-priority first-match (production get_fmha_impl), the fallback when a whole bucket has no usable backend."""
    from rtp_llm.models_py.modules.factory.attention.attn_factory import get_fmha_impl

    try:
        impl = get_fmha_impl(
            attn_configs,
            None,
            attn_inputs,
            fmha_config,
            None,
            True,
            max_seq_len,
            parallelism_config,
        )
        return type(impl).__name__
    except Exception as e:
        logger.warning("[dispatcher] fallback get_fmha_impl failed: %r", e)
        return None


def run_backend_selection(
    model,
    inputs,
    *,
    selector: Optional[Selector] = None,
    warmup: int = 10,
    iters: int = 50,
    l2_fill_mode: str = "store",
    use_graph: bool = True,
) -> Optional[str]:
    """Select the decode backend for a single capture bs, returning the winner class name (None = left empty, capture falls back to fixed priority).

    Only rank0 does the real-machine benchmark + selection; the winner registry
    index is broadcast to Group.TP so every rank gets the same winner.
    """
    parallelism_config = model.parallelism_config
    attn_configs = model.config.getAttentionConfigs(
        parallelism_config.get_attn_tp_size()
    )
    attn_inputs = inputs.attention_inputs
    bs = int(attn_inputs.input_lengths.size(0))
    max_seq_len = int(getattr(model.config, "max_seq_len", 0)) or int(
        attn_configs.max_seq_len
    )
    grid = kv_grid(max_seq_len)
    registry = _decode_registry()
    tp_size, is_tp_root, src = _tp_geometry(parallelism_config)

    code = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    if is_tp_root:
        # Contract: benchmarked decode attention forward must be free of collective
        # communication (allreduce/broadcast). rank0 runs bench alone; a backend
        # whose forward triggers TP collectives would deadlock here.
        # root must always reach the broadcast below (otherwise non-root ranks stall at broadcast -> hang).
        try:
            winner = _select_on_root(
                model,
                attn_configs,
                attn_inputs,
                parallelism_config,
                grid,
                selector,
                bs,
                max_seq_len,
                warmup,
                iters,
                l2_fill_mode,
                use_graph,
            )
            if winner is not None and winner in registry:
                code[0] = registry.index(winner)
        except (
            Exception
        ) as e:  # noqa: BLE001 -- leave empty and fall back to fixed priority, but still broadcast to keep lockstep
            logger.warning(
                "[dispatcher] bs=%d root selection exception, fall back to fixed priority: %r",
                bs,
                e,
            )

    if tp_size > 1:
        from rtp_llm.models_py.distributed.collective_torch import Group, broadcast

        broadcast(code, src, group=Group.TP)  # src is a global rank (dp_rank*tp_size)

    idx = int(code[0].item())
    chosen = registry[idx] if 0 <= idx < len(registry) else None
    logger.info(
        "[dispatcher] bs=%d -> %s",
        bs,
        chosen or "(left empty, fall back to fixed priority)",
    )
    return chosen


def _select_on_root(
    model,
    attn_configs,
    attn_inputs,
    parallelism_config,
    grid,
    selector,
    bs,
    max_seq_len,
    warmup,
    iters,
    l2_fill_mode,
    use_graph,
) -> Optional[str]:
    eligible = _eligible(
        attn_configs, attn_inputs, parallelism_config, model.fmha_config
    )
    # For hybrid models (linear + full attention), layer 0 may be linear
    # attention whose KV cache layout is incompatible with MHA backends.
    # Select the first standard-attention layer's cache for benchmarking.
    # HybridAttentionType.NONE = standard full attention (not LINEAR/SLIDING_WINDOW).
    bench_layer_idx = 0
    hybrid_cfg = getattr(model.config, "hybrid_attention_config", None)
    if hybrid_cfg is not None:
        from rtp_llm.ops import HybridAttentionType

        for i, t in enumerate(hybrid_cfg.hybrid_attention_types):
            if t != HybridAttentionType.LINEAR:
                bench_layer_idx = i
                break
    layer_kv_cache = model.kv_cache.get_layer_cache(bench_layer_idx)
    # clip the kv grid to the upper bound of the engine's capture sequence_lengths (real-machine dump: = max_seq_len-2;
    # the page table only covers up to here, exceeding it goes out of bounds, see backend_bench._set_uniform_seq_len).
    seq_cap = (
        int(attn_inputs.sequence_lengths.flatten()[0].item())
        if attn_inputs.sequence_lengths.numel()
        else max_seq_len
    )
    kv_list = [kv for kv in grid if kv <= seq_cap] or [seq_cap]
    matrix = {}
    for name in eligible:
        impl_cls = _impl_by_name(name)
        if impl_cls is None:
            continue
        # multi-kv-point: for each kv, set sequence_lengths on a clone and time real-machine capture+replay;
        # select_min_mean averages latency across the kv grid points and picks the backend that is "most stable throughout" for this bs bucket.
        lats = backend_bench.bench_backend_grid(
            impl_cls,
            attn_configs,
            attn_inputs,
            layer_kv_cache,
            parallelism_config,
            kv_list,
            warmup=warmup,
            iters=iters,
            l2_fill_mode=l2_fill_mode,
            use_graph=use_graph,
        )
        if lats is not None:
            matrix[name] = lats
    torch.cuda.synchronize()  # ordering invariant: return only after all bench finishes -> engine's actual capture

    if not matrix:
        reason = "eligible empty" if not eligible else "eligible all N/A"
        fb = _fallback(
            attn_configs,
            attn_inputs,
            model.fmha_config,
            max_seq_len,
            parallelism_config,
        )
        logger.warning(
            "[dispatcher] bs=%d %s -> fall back to fixed priority %s",
            bs,
            reason,
            fb or "(also failed, left empty)",
        )
        return fb

    # Use select_stable by default: two-level deterministic selection
    # (threshold filter + registry-order tiebreaker) for CI and production stability.
    if selector is not None:
        choice = selector(matrix)
    else:
        import os as _os

        threshold = float(
            _os.environ.get("DYN_DECODE_THRESHOLD", str(STABLE_THRESHOLD))
        )
        cluster_margin = float(
            _os.environ.get("DYN_DECODE_CLUSTER_MARGIN", str(STABLE_CLUSTER_MARGIN))
        )
        default_backend = _fallback(
            attn_configs,
            attn_inputs,
            model.fmha_config,
            max_seq_len,
            parallelism_config,
        )
        registry = _decode_registry()
        choice = select_stable(
            matrix, registry, default_backend, threshold, cluster_margin
        )

    if choice is not None:
        return choice
    # select_stable returns default_backend when nothing is significantly better,
    # so reaching here means default_backend was also None.
    fb = _fallback(
        attn_configs, attn_inputs, model.fmha_config, max_seq_len, parallelism_config
    )
    logger.warning(
        "[dispatcher] bs=%d selection returned None -> fall back to fixed priority %s",
        bs,
        fb or "(also failed, left empty)",
    )
    return fb


def _impl_by_name(name: str):
    from rtp_llm.models_py.modules.factory.attention.attn_factory import DECODE_MHA_IMPS

    for c in DECODE_MHA_IMPS:
        if c.__name__ == name:
            return c
    return None


def instantiate_decode_impl(model, attn_inputs, name: str, is_cuda_graph: bool):
    """Instantiate the backend class name chosen by the plan (used by prepare_fmha_impl's table-lookup branch).

    If any condition is not met (class does not exist / not supported / not
    instantiable / does not support cuda graph), returns None, and the caller falls
    back to fixed-priority get_fmha_impl (never asserts).
    """
    cls = _impl_by_name(name)
    if cls is None:
        return None
    # Respect fmha_config disable flags, consistent with _eligible() and the
    # fixed-priority get_fmha_impl path: a plan entry for a now-disabled backend must
    # fall back to fixed priority instead of being instantiated.
    from rtp_llm.models_py.modules.factory.attention.attn_factory import (
        _is_fmha_impl_disabled,
    )

    if _is_fmha_impl_disabled(name, getattr(model, "fmha_config", None)):
        return None
    attn_configs = model.config.getAttentionConfigs(
        model.parallelism_config.get_attn_tp_size()
    )
    # Set the same two dynamic attributes the production fixed-priority path sets
    # (get_fmha_impl -> attn_inputs.is_cuda_graph; AttnImplFactory.get_fmha_impl ->
    # attn_inputs.headwise_config), to identical values. Not a leak/dirty-object hazard:
    # on any return None below, prepare_fmha_impl falls through to
    # AttnImplFactory.get_fmha_impl, which re-sets both fields to the same values --
    # restoring them here would be immediately overwritten and is pointless.
    attn_inputs.is_cuda_graph = is_cuda_graph
    attn_inputs.headwise_config = getattr(model.config, "headwise_config", None)
    try:
        if not cls.support(attn_configs, attn_inputs):
            return None
        if not cls.support_parallelism_config(model.parallelism_config):
            return None
        # All impls take (attn_configs, attn_inputs, parallelism_config=None); call
        # 3-arg unconditionally like production get_fmha_impl. A former try/except
        # TypeError fallback could mask a genuine internal TypeError by silently
        # reconstructing 2-arg with parallelism_config=None (losing TP info).
        inst = cls(attn_configs, attn_inputs, model.parallelism_config)
        if is_cuda_graph and not inst.support_cuda_graph():
            return None
        return inst
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[dispatcher] instantiating %s per plan failed, fall back to fixed priority: %r",
            name,
            e,
        )
        return None
