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

import os

import torch

# Module-level cache for the Mega MoE symm-mem dispatch buffer. Keyed by the
# shape parameters so different model configs in the same process don't
# collide; in practice there's only ever one entry per process.
_MEGA_BUF_CACHE: dict = {}


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
    return buf


def _mega_moe_available() -> bool:
    """Whether DeepGEMM's ``fp8_fp4_mega_moe`` (symm-mem fused dispatch +
    L1 GEMM + SwiGLU + L2 GEMM + combine, SM100-only) is usable here.

    Requires: deep_gemm ≥ 2.5 (commit 891d57b introduced it), torch ≥ 2.9
    for ``torch.distributed._symmetric_memory``, CUDA device SM100+, and
    an initialised world-size process group of size > 1."""
    try:
        import deep_gemm

        if not hasattr(deep_gemm, "fp8_fp4_mega_moe"):
            return False
    except Exception:
        return False
    try:
        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return False
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 10


def _mega_moe_enabled() -> bool:
    """Default on when ``_mega_moe_available()`` holds; set
    ``DSV4_USE_MEGA_MOE=0`` to force the pre-mega per-expert path."""
    if os.environ.get("DSV4_USE_MEGA_MOE", "1") == "0":
        return False
    return _mega_moe_available()
