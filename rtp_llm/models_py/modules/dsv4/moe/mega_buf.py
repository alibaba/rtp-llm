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
import socket
from datetime import timedelta

import torch

# Module-level cache for the Mega MoE symm-mem dispatch buffer. Keyed by the
# shape parameters so different model configs in the same process don't
# collide; in practice there's only ever one entry per process.
_MEGA_BUF_CACHE: dict = {}
_MEGA_OUTPUT_CACHE: dict = {}
_SINGLE_RANK_DIST_INITIALIZED = False


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "on", "yes")


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


def _init_single_rank_process_group_if_needed() -> str | None:
    """Create a one-rank NCCL group for the vLLM-compatible EP1 MegaMoE path.

    RTP normally skips ``torch.distributed`` initialisation when
    ``world_size == 1``. DeepGEMM MegaMoE still needs a ProcessGroup because
    its API routes through PyTorch symmetric memory even for a single rank.
    This helper is deliberately opt-in via ``DSV4_MEGA_MOE_EP1=1``.
    """
    global _SINGLE_RANK_DIST_INITIALIZED
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return None

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=60),
        )
        _SINGLE_RANK_DIST_INITIALIZED = True
        logging.info("[DSV4 MegaMoE] initialized single-rank NCCL process group")
        return None
    except Exception as e:
        return f"failed to initialize single-rank torch.distributed: {e}"


def _mega_moe_unavailable_reason(
    *,
    allow_single_rank: bool = False,
    initialize_single_rank: bool = False,
) -> str | None:
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
            if initialize_single_rank:
                init_reason = _init_single_rank_process_group_if_needed()
                if init_reason is not None:
                    return init_reason
            if not dist.is_initialized():
                return "torch.distributed is not initialized"
        if dist.get_world_size() <= 1 and not allow_single_rank:
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


def _mega_moe_ep1_enabled() -> bool:
    """Opt-in vLLM-compatible MegaMoE path for EP1/TP1 precision alignment."""
    if os.environ.get("DSV4_USE_MEGA_MOE", "1") == "0":
        return False
    if not _env_truthy("DSV4_MEGA_MOE_EP1"):
        return False
    return (
        _mega_moe_unavailable_reason(
            allow_single_rank=True,
            initialize_single_rank=True,
        )
        is None
    )


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
