"""Symm-mem buffer cache + capability gates for DeepGEMM Mega MoE.

The Mega MoE kernel needs a PyTorch ``symmetric_memory`` buffer for cross-rank
NVLink dispatch + combine. The buffer holds only single-layer staging
(per-token x/sf, topk, l1_acts/sf, l2_acts/sf) — the previous layer's data is
no longer needed once the next layer's MoE starts. So we share **one buffer**
across all MoE layers via a module-level cache; otherwise V4-Flash's 64+ MoE
layers each allocate ~3.4 GiB at CP=4 → ~218 GiB symm memory per rank,
OOMing the GB200's 188 GiB after dozens of allocs.

The cache key set MUST stay invariant across the refactor — see Phase 1 risk
#9 in ``.claude/plans/optimized-riding-mist.md``.
"""

import gc
import logging
import os
import weakref

import torch

# Module-level cache for the Mega MoE symm-mem dispatch buffer. Keyed by the
# shape parameters so different model configs in the same process don't
# collide; in practice there's only ever one entry per process.
_MEGA_BUF_CACHE: dict = {}
_MEGA_OUTPUT_CACHE: dict = {}

# Live Mega MoE strategy instances. Each MoE layer captures a STRONG reference to
# the shared symm buffer in ``self._mega_buf`` (read directly on the forward hot
# path), so clearing ``_MEGA_BUF_CACHE`` alone cannot free the buffer at sleep --
# every layer's reference must be dropped too. This weak registry lets the sleep
# reclaim reach them without keeping them alive. See ``release_mega_symm_buffers``.
_MEGA_STRATEGY_REGISTRY: "weakref.WeakSet" = weakref.WeakSet()

# Set True when this engine captures its decode forward into a CUDA graph.
#
# Both Mega MoE device buffers -- the symmetric-memory dispatch buffer
# (``_mega_buf``) and the bf16 output staging buffer (``_mega_y``) -- are cached
# at module scope and read directly by the MoE kernel, so their device pointers
# get BAKED into the captured graph. A CUDA graph is REPLAYED after wake, and
# Python does not run during replay, so the lazy re-create in
# ``_ensure_mega_buffers`` never fires. If sleep freed either buffer (drop ref +
# ``empty_cache`` returns the segment to the driver, unmapping the VA), the first
# post-wake decode replay writes into a dangling VA -> illegal access on every
# rank. The sticky CUDA error then surfaces at the async output dispatch and
# escalates to ``std::terminate`` (SIGABRT).
#
# L1/L2 therefore keep graph-baked buffers at stable VAs. L3 first destroys every
# graph owner, releases these buffers, eagerly rebuilds them after PG restore, and
# then recaptures the graphs. No-graph roles use the same eager L3 rebuild; lower
# sleep levels retain the forward fallback.
_MEGA_BUFFERS_GRAPH_BAKED: bool = False
_MEGA_CUDA_GRAPH_GENERATION: int = 0
_MEGA_CUDA_GRAPH_INVALIDATED: bool = False
_MEGA_CUDA_GRAPH_OWNERS: set[int] = set()
_MEGA_CUDA_GRAPH_INVALIDATED_OWNERS: set[int] = set()


def set_mega_buffers_graph_baked(enabled: bool) -> None:
    """Record whether this engine captures CUDA graphs (see the flag docstring).

    Called once at model setup from the config's ``enable_cuda_graph``. When set,
    the sleep-time buffer releases below become no-ops so the graph-baked device
    pointers stay valid across the wake boundary.
    """
    global _MEGA_BUFFERS_GRAPH_BAKED
    _MEGA_BUFFERS_GRAPH_BAKED = bool(enabled) or bool(_MEGA_CUDA_GRAPH_OWNERS)


def register_mega_cuda_graph_owner(owner: int) -> None:
    global _MEGA_BUFFERS_GRAPH_BAKED
    _MEGA_CUDA_GRAPH_OWNERS.add(int(owner))
    _MEGA_BUFFERS_GRAPH_BAKED = True


def mega_buffers_graph_baked() -> bool:
    return _MEGA_BUFFERS_GRAPH_BAKED


def mega_buffer_generation() -> int:
    return _MEGA_CUDA_GRAPH_GENERATION


def mega_cuda_graph_resources_invalidated() -> bool:
    return _MEGA_CUDA_GRAPH_INVALIDATED


def mega_output_buffer_gib() -> float:
    """Best-effort resident size of the Mega MoE output staging buffer(s)."""
    total = 0
    for buf in _MEGA_OUTPUT_CACHE.values():
        try:
            total += buf.numel() * buf.element_size()
        except Exception:
            pass
    return total / (1024**3)


def _register_mega_strategy(strategy) -> None:
    """Track a live Mega MoE strategy so its per-layer buffer refs can be dropped
    at sleep. Registration is fail-closed when discard-mode wake depends on it;
    other configurations retain best-effort behavior.

    Also stamp the owning model's build scope (``id(base_model)`` while its
    py-model is under construction) so the level-2 wake reload can attribute this
    strategy to one model. A checkpoint-backed MTP draft coexisting with the main
    model registers here too and its lone layer collides on ``layer_id=0`` with
    the main model's layer 0; the stamp lets each ``WeightManager`` re-derive only
    its own layers. ``None`` when built outside a scope (e.g. non-sleep runs) —
    harmless, as the reload filter matches ``None`` scope managers to ``None``
    stamps."""
    from rtp_llm.model_loader.weight_memory_saver import (
        current_model_scope,
        keep_loader_database_for_wake,
    )

    strict_reload = keep_loader_database_for_wake()
    try:
        strategy._sleep_model_scope = current_model_scope()
        _MEGA_STRATEGY_REGISTRY.add(strategy)
    except Exception:
        if strict_reload:
            raise
        return


def iter_mega_strategies() -> list:
    """Snapshot of the live Mega MoE strategies (one per MoE layer).

    Used by the level-2 wake reload (``WeightManager.reload_weights_from_loader``)
    to re-derive each layer's kernel weights from the re-streamed checkpoint via
    ``MegaStrategy.reload_routed_weights``. Returns a plain list so callers do not
    hold the weakset during iteration."""
    return list(_MEGA_STRATEGY_REGISTRY)


def rebuild_mega_symm_buffers() -> int:
    """Eagerly rebuild every live Mega buffer after Level3 PG restore.

    The creation path is collective, so use the saved buffer arguments to keep
    distinct buffer configurations in the same order on every rank. Strategies
    sharing a configuration reuse the module cache after the first rendezvous.
    Returns the number of live strategies rebuilt for lifecycle diagnostics.
    """
    strategies = list(_MEGA_STRATEGY_REGISTRY)
    strategies.sort(
        key=lambda strategy: tuple(
            (name, repr(value))
            for name, value in sorted(strategy._mega_buf_kwargs.items())
        )
    )
    for strategy in strategies:
        strategy._ensure_mega_buffers()
    if strategies:
        logging.info(
            "[DSV4 MegaMoE] eagerly rebuilt buffers for %d live strategies",
            len(strategies),
        )
    return len(strategies)


def _release_mega_output_buffers_unchecked() -> tuple[int, float]:
    freed_bytes = 0
    for buf in _MEGA_OUTPUT_CACHE.values():
        try:
            freed_bytes += buf.numel() * buf.element_size()
        except Exception:
            pass
    entries = len(_MEGA_OUTPUT_CACHE)
    for strat in list(_MEGA_STRATEGY_REGISTRY):
        try:
            strat._mega_y = None
        except Exception:
            pass
    _MEGA_OUTPUT_CACHE.clear()
    return entries, freed_bytes / (1024**3)


def release_mega_output_buffers() -> tuple[int, float]:
    """Drop per-layer refs to the shared bf16 output staging buffer.

    Returns ``(cache_entries, GiB)`` for sleep-reclaim diagnostics. A no-op when
    CUDA graphs are captured; L3 uses the explicit invalidation path below only
    after every graph owner has stopped replaying the baked addresses.
    """
    if _MEGA_BUFFERS_GRAPH_BAKED:
        return 0, 0.0
    return _release_mega_output_buffers_unchecked()


def _release_mega_symm_buffers_unchecked() -> float:
    freed_bytes = 0.0
    cached_buffers = list(_MEGA_BUF_CACHE.items())
    strategies = list(_MEGA_STRATEGY_REGISTRY)
    buffers_to_destroy: dict[int, tuple[object, list[str]]] = {}

    def add_buffer(buf, owner: str) -> None:
        if buf is None:
            return
        entry = buffers_to_destroy.get(id(buf))
        if entry is None:
            buffers_to_destroy[id(buf)] = (buf, [owner])
        else:
            entry[1].append(owner)

    for key, buf in cached_buffers:
        add_buffer(buf, f"cache[{key!r}]")
        try:
            freed_bytes += buf.buffer.numel() * buf.buffer.element_size()
        except Exception:
            pass

    cleanup_errors: list[tuple[str, Exception]] = []
    for index, strat in enumerate(strategies):
        owner = f"strategy[{index}]"
        try:
            add_buffer(getattr(strat, "_mega_buf", None), owner)
        except Exception as error:
            cleanup_errors.append((f"{owner} read _mega_buf", error))
        for attribute in ("_mega_buf", "_mega_y", "_mega_group"):
            try:
                setattr(strat, attribute, None)
            except Exception as error:
                cleanup_errors.append((f"{owner} clear {attribute}", error))

    # Clear all module and owner references even when a native destroy fails.
    _MEGA_BUF_CACHE.clear()
    try:
        _release_mega_output_buffers_unchecked()
    except Exception as error:
        cleanup_errors.append(("clear Mega output buffers", error))

    for buf, owners in buffers_to_destroy.values():
        try:
            buf.destroy()
        except Exception as error:
            cleanup_errors.append(
                (f"destroy Mega symmetric buffer owned by {', '.join(owners)}", error)
            )

    try:
        gc.collect()
    except Exception as error:
        cleanup_errors.append(("collect released Mega buffers", error))

    if cleanup_errors:
        details = "; ".join(
            f"{operation}: {type(error).__name__}: {error}"
            for operation, error in cleanup_errors
        )
        raise RuntimeError(
            f"failed to release Mega symmetric buffers "
            f"({len(cleanup_errors)} error(s)): {details}"
        ) from cleanup_errors[0][1]

    return freed_bytes / (1024**3)


def release_mega_symm_buffers() -> float:
    """Sleep-time reclaim of the Mega symmetric and output staging buffers.

    This is the existing L1/L2 path. Graph-baked buffers remain resident at
    stable VAs; L3 bypasses this gate only after all graph owners invalidate.
    """
    global _MEGA_CUDA_GRAPH_GENERATION
    if _MEGA_BUFFERS_GRAPH_BAKED:
        return 0.0
    # ProcessGroupNCCL is destroyed immediately after this call in L3. Force
    # every strategy to bind the rebuilt WORLD group before the next rendezvous.
    _MEGA_CUDA_GRAPH_GENERATION += 1
    return _release_mega_symm_buffers_unchecked()


def invalidate_mega_cuda_graph_resources(owner: int | None = None) -> int:
    """L3-only invalidation after captured graphs have been destroyed.

    This deliberately bypasses the L1/L2 graph-baked gate. The generation bump
    makes every live strategy reject its previous references before the
    coordinated L3 wake eagerly materializes the new generation.
    """
    global _MEGA_CUDA_GRAPH_GENERATION, _MEGA_CUDA_GRAPH_INVALIDATED
    if not _MEGA_BUFFERS_GRAPH_BAKED:
        return _MEGA_CUDA_GRAPH_GENERATION

    if _MEGA_CUDA_GRAPH_OWNERS:
        if owner is None:
            _MEGA_CUDA_GRAPH_INVALIDATED_OWNERS.update(_MEGA_CUDA_GRAPH_OWNERS)
        elif int(owner) in _MEGA_CUDA_GRAPH_OWNERS:
            _MEGA_CUDA_GRAPH_INVALIDATED_OWNERS.add(int(owner))
        else:
            return _MEGA_CUDA_GRAPH_GENERATION
        if not _MEGA_CUDA_GRAPH_OWNERS.issubset(_MEGA_CUDA_GRAPH_INVALIDATED_OWNERS):
            return _MEGA_CUDA_GRAPH_GENERATION

    if _MEGA_CUDA_GRAPH_INVALIDATED:
        return _MEGA_CUDA_GRAPH_GENERATION
    _MEGA_CUDA_GRAPH_INVALIDATED = True
    _MEGA_CUDA_GRAPH_GENERATION += 1
    _release_mega_symm_buffers_unchecked()
    return _MEGA_CUDA_GRAPH_GENERATION


def mark_mega_cuda_graph_resources_recaptured(owner: int | None = None) -> None:
    """Mark the current Mega generation reusable after successful recapture."""
    global _MEGA_CUDA_GRAPH_INVALIDATED
    if owner is None:
        _MEGA_CUDA_GRAPH_INVALIDATED_OWNERS.clear()
    else:
        _MEGA_CUDA_GRAPH_INVALIDATED_OWNERS.discard(int(owner))
    if _MEGA_BUFFERS_GRAPH_BAKED:
        _MEGA_CUDA_GRAPH_INVALIDATED = bool(_MEGA_CUDA_GRAPH_INVALIDATED_OWNERS)


def estimate_mega_moe_symm_buffer_bytes(
    group_size: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    use_fp8_dispatch: bool = True,
    activation: str = "swiglu",
) -> int | None:
    try:
        import deep_gemm

        return int(
            deep_gemm._C.get_symm_buffer_size_for_mega_moe(
                group_size,
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                intermediate_hidden,
                use_fp8_dispatch,
                activation,
            )[0]
        )
    except Exception:
        return None


def _get_or_create_mega_buf(
    group,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
    use_fp8_dispatch,
    activation,
):
    import deep_gemm

    key = (
        id(group),
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        bool(use_fp8_dispatch),
        activation,
    )
    buf = _MEGA_BUF_CACHE.get(key)
    if buf is None:
        try:
            group_size = int(group.size())
        except Exception:
            group_size = 0
        estimated_bytes = (
            estimate_mega_moe_symm_buffer_bytes(
                group_size=group_size,
                num_experts=num_experts,
                num_max_tokens_per_rank=num_max_tokens_per_rank,
                num_topk=num_topk,
                hidden=hidden,
                intermediate_hidden=intermediate_hidden,
                use_fp8_dispatch=use_fp8_dispatch,
                activation=activation,
            )
            if group_size > 0
            else None
        )
        buf = deep_gemm.get_symm_buffer_for_mega_moe(
            group=group,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            use_fp8_dispatch=use_fp8_dispatch,
            activation=activation,
        )
        actual_bytes = None
        try:
            actual_bytes = int(buf.buffer.numel() * buf.buffer.element_size())
        except Exception:
            pass
        if actual_bytes is not None:
            if estimated_bytes is not None:
                logging.info(
                    "[DSV4 MegaMoE] allocated symm buffer: group_size=%d "
                    "num_experts=%d max_tokens_per_rank=%d topk=%d hidden=%d "
                    "intermediate=%d actual=%.3f GiB estimated=%.3f GiB",
                    group_size,
                    num_experts,
                    num_max_tokens_per_rank,
                    num_topk,
                    hidden,
                    intermediate_hidden,
                    actual_bytes / (1024**3),
                    estimated_bytes / (1024**3),
                )
            else:
                logging.info(
                    "[DSV4 MegaMoE] allocated symm buffer: group_size=%d "
                    "num_experts=%d max_tokens_per_rank=%d topk=%d hidden=%d "
                    "intermediate=%d actual=%.3f GiB",
                    group_size,
                    num_experts,
                    num_max_tokens_per_rank,
                    num_topk,
                    hidden,
                    intermediate_hidden,
                    actual_bytes / (1024**3),
                )
        elif estimated_bytes is not None:
            logging.info(
                "[DSV4 MegaMoE] allocated symm buffer: group_size=%d "
                "num_experts=%d max_tokens_per_rank=%d topk=%d hidden=%d "
                "intermediate=%d actual=unavailable estimated=%.3f GiB",
                group_size,
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                intermediate_hidden,
                estimated_bytes / (1024**3),
            )
        _MEGA_BUF_CACHE[key] = buf
    return buf


def _get_or_create_mega_output(
    capacity,
    hidden,
    dtype,
    device,
):
    key = (device, hidden, dtype)
    cached = _MEGA_OUTPUT_CACHE.get(key)
    if cached is not None and cached.size(0) >= capacity:
        return cached
    cached = torch.empty((max(capacity, 1), hidden), dtype=dtype, device=device)
    _MEGA_OUTPUT_CACHE[key] = cached
    return cached


def _mega_moe_unavailable_reason() -> str | None:
    """Return ``None`` when Mega MoE can run, otherwise a human-readable reason."""
    try:
        import deep_gemm

        if not hasattr(deep_gemm, "fp8_fp4_mega_moe"):
            return "deep_gemm.fp8_fp4_mega_moe is missing"
    except Exception as e:
        return f"failed to import deep_gemm: {e}"
    try:
        import torch.distributed as dist

        if not dist.is_initialized():
            return "torch.distributed is not initialized"
        if dist.get_world_size() <= 1:
            return f"distributed world_size={dist.get_world_size()} is not > 1"
    except Exception as e:
        return f"failed to query torch.distributed: {e}"
    if not torch.cuda.is_available():
        return "CUDA is not available"
    cap = torch.cuda.get_device_capability()
    if cap[0] < 10:
        return f"CUDA device capability sm{cap[0]}{cap[1]} is below SM100"
    return None


def _mega_moe_available() -> bool:
    """Whether DeepGEMM's ``fp8_fp4_mega_moe`` (symm-mem fused dispatch +
    L1 GEMM + SwiGLU + L2 GEMM + combine, SM100-only) is usable here.

    Requires: deep_gemm >= 2.5 (commit 891d57b introduced it), torch >= 2.9
    for ``torch.distributed._symmetric_memory``, CUDA device SM100+, and
    an initialised world-size process group of size > 1."""
    return _mega_moe_unavailable_reason() is None


def _mega_moe_enabled() -> bool:
    """Default on when ``_mega_moe_available()`` holds.

    ``DSV4_USE_MEGA_MOE=0`` disables Mega explicitly. EP>1 callers must treat
    that as a configuration error rather than falling back to DeepEP.
    """
    if os.environ.get("DSV4_USE_MEGA_MOE", "1") == "0":
        return False
    return _mega_moe_available()


def _mega_moe_disabled_or_unavailable_reason() -> str:
    if os.environ.get("DSV4_USE_MEGA_MOE", "1") == "0":
        return "DSV4_USE_MEGA_MOE=0 disables Mega MoE"
    return _mega_moe_unavailable_reason() or "unknown Mega MoE availability failure"
