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


def _register_mega_strategy(strategy) -> None:
    """Track a live Mega MoE strategy so its per-layer buffer refs can be dropped
    at sleep. Best-effort; never raises.

    Also stamp the owning model's build scope (``id(base_model)`` while its
    py-model is under construction) so the level-2 wake reload can attribute this
    strategy to one model. A checkpoint-backed MTP draft coexisting with the main
    model registers here too and its lone layer collides on ``layer_id=0`` with
    the main model's layer 0; the stamp lets each ``WeightManager`` re-derive only
    its own layers. ``None`` when built outside a scope (e.g. non-sleep runs) —
    harmless, as the reload filter matches ``None`` scope managers to ``None``
    stamps."""
    try:
        from rtp_llm.model_loader.weight_memory_saver import current_model_scope

        strategy._sleep_model_scope = current_model_scope()
    except Exception:
        pass
    try:
        _MEGA_STRATEGY_REGISTRY.add(strategy)
    except Exception:
        pass


def iter_mega_strategies() -> list:
    """Snapshot of the live Mega MoE strategies (one per MoE layer).

    Used by the level-2 wake reload (``WeightManager.reload_weights_from_loader``)
    to re-derive each layer's kernel weights from the re-streamed checkpoint via
    ``MegaStrategy.reload_routed_weights``. Returns a plain list so callers do not
    hold the weakset during iteration."""
    return list(_MEGA_STRATEGY_REGISTRY)


def release_mega_output_buffers() -> tuple[int, float]:
    """Drop per-layer refs to the shared bf16 output staging buffer.

    Returns ``(cache_entries, GiB)`` for sleep-reclaim diagnostics.
    """
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


def release_mega_symm_buffers() -> float:
    """Sleep-time reclaim of the Mega MoE symmetric-memory buffer (+ bf16 output
    staging buffer).

    Drops every per-layer strong reference (``self._mega_buf`` / ``self._mega_y``
    across all registered strategies), destroys the cached buffers, clears the
    module caches, and gc's so the symmetric-memory allocation is actually
    released to the driver. Returns the GiB of symm buffer dropped (best-effort
    estimate).

    Non-collective (pure Python ref-drops + ``SymmBuffer.destroy()``), so it is
    safe on the quiesced sleep path. The buffers are lazily re-created on the
    first forward after wake via ``MegaStrategy._ensure_mega_buffers`` -- that
    path runs ``symm_mem.rendezvous`` (a collective), which is safe because all
    ranks execute the same MoE layer's forward in lockstep, exactly as at init.
    """
    freed_bytes = 0.0
    for buf in _MEGA_BUF_CACHE.values():
        try:
            freed_bytes += buf.buffer.numel() * buf.buffer.element_size()
        except Exception:
            pass
    # 1) Drop per-layer strong refs first, else the buffers stay alive below.
    for strat in list(_MEGA_STRATEGY_REGISTRY):
        try:
            strat._mega_buf = None
        except Exception:
            pass
    # 2) Destroy the cached buffers (nulls each SymmBuffer's handle/tensor refs).
    for buf in list(_MEGA_BUF_CACHE.values()):
        try:
            buf.destroy()
        except Exception:
            pass
    _MEGA_BUF_CACHE.clear()
    release_mega_output_buffers()
    gc.collect()
    return freed_bytes / (1024**3)


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
