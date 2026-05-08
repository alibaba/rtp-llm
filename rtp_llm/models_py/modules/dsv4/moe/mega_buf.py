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

import logging
import os

import torch

# Module-level cache for the Mega MoE symm-mem dispatch buffer. Keyed by the
# shape parameters so different model configs in the same process don't
# collide; in practice there's only ever one entry per process.
_MEGA_BUF_CACHE: dict = {}
_MEGA_OUTPUT_CACHE: dict = {}


def get_mega_moe_group():
    """Return the default process group for DeepGEMM symmetric memory."""
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError("Mega MoE process group requires torch.distributed")
    ranks = list(range(dist.get_world_size()))
    logging.info("Use default Mega MoE process group WORLD with ranks=%s", ranks)
    return dist.group.WORLD


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
        if torch.cuda.is_available():
            logging.info(
                "Create Mega MoE symm buffer: group=%s current_cuda=%d "
                "experts=%d tokens_per_rank=%d topk=%d hidden=%d inter=%d",
                getattr(group, "group_name", group),
                torch.cuda.current_device(),
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                intermediate_hidden,
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
        _MEGA_BUF_CACHE[key] = buf
    else:
        logging.info(
            "Reuse Mega MoE symm buffer: group=%s tokens_per_rank=%d hidden=%d inter=%d",
            getattr(group, "group_name", group),
            num_max_tokens_per_rank,
            hidden,
            intermediate_hidden,
        )
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
