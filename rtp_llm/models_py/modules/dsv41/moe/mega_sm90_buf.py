"""DeepGEMM SM90 MegaMoE symmetric buffer and availability helpers.

The Hopper fused kernel (``sm90_fp8_fp4_mega_moe``) owns a distinct collective
protocol: ``SymmBufferSM90`` exposes ``x`` / ``x_sf`` / ``topk_idx`` /
``topk_weights`` staging views, requires SM90, and does not support shared
expert fusion or reuse of the SM100 base buffer.

Opt-in only: ``DSV4_USE_MEGA_MOE_SM90=1``. The SM100 Mega path keeps its own
``DSV4_USE_MEGA_MOE*`` switches and buffers.
"""

from __future__ import annotations

import logging
import os

import torch

_MEGA_SM90_BUF_CACHE: dict[tuple, object] = {}
_MEGA_SM90_OUTPUT_CACHE: dict[tuple, torch.Tensor] = {}

_SM90_MEGA_MOE_ENV = "DSV4_USE_MEGA_MOE_SM90"
_SM90_HIDDEN_ALIGN = 512
_SM90_INTER_ALIGN = 256
_SM90_MAX_HIDDEN = 8192
_SM90_MAX_INTER = 4096
_SM90_MAX_GROUP = 32
_SM90_ACTIVATION = "swiglu"


def _sm90_mega_moe_requested() -> bool:
    """Whether the operator opted into the Hopper fused MegaMoE kernel."""
    return os.environ.get(_SM90_MEGA_MOE_ENV, "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _sm90_mega_moe_unavailable_reason() -> str | None:
    """Return ``None`` when the SM90 MegaMoE kernel can run, else the reason."""
    try:
        import deep_gemm

        for symbol in (
            "sm90_fp8_fp4_mega_moe",
            "transform_weights_for_mega_moe_sm90",
            "get_symm_buffer_for_mega_moe_sm90",
        ):
            if not hasattr(deep_gemm, symbol):
                return f"deep_gemm.{symbol} is missing"
    except Exception as e:  # noqa: BLE001 - report any import failure verbatim
        return f"failed to import deep_gemm: {e}"
    try:
        import torch.distributed as dist

        if not dist.is_initialized():
            return "torch.distributed is not initialized"
        if dist.get_world_size() <= 1:
            return f"distributed world_size={dist.get_world_size()} is not > 1"
    except Exception as e:  # noqa: BLE001
        return f"failed to query torch.distributed: {e}"
    if not torch.cuda.is_available():
        return "CUDA is not available"
    capability = torch.cuda.get_device_capability()
    if tuple(capability) != (9, 0):
        return f"CUDA device capability sm{capability[0]}{capability[1]} is not SM90"
    return None


def _sm90_mega_moe_available() -> bool:
    return _sm90_mega_moe_unavailable_reason() is None


def _sm90_mega_moe_enabled() -> bool:
    """Opt-in gate: the env switch must be set and the kernel must be usable."""
    if not _sm90_mega_moe_requested():
        return False
    return _sm90_mega_moe_available()


def _sm90_shape_supported(
    hidden: int,
    intermediate: int,
    num_experts: int,
    group_size: int,
    num_topk: int,
) -> bool:
    """Mirror ``SymmBufferSM90``'s own geometry validation before allocating."""
    return (
        1 <= group_size <= _SM90_MAX_GROUP
        and num_experts > 0
        and num_experts % group_size == 0
        and 1 <= num_topk <= min(32, num_experts)
        and 0 < hidden <= _SM90_MAX_HIDDEN
        and hidden % _SM90_HIDDEN_ALIGN == 0
        and 0 < intermediate <= _SM90_MAX_INTER
        and intermediate % _SM90_INTER_ALIGN == 0
    )


def _get_or_create_mega_sm90_buf(
    *,
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    activation: str = _SM90_ACTIVATION,
):
    """Allocate (or reuse) the collective buffer so graph replay keeps addresses."""
    import deep_gemm

    key = (
        id(group),
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        activation,
    )
    buf = _MEGA_SM90_BUF_CACHE.get(key)
    if buf is None:
        buf = deep_gemm.get_symm_buffer_for_mega_moe_sm90(
            group,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            activation,
        )
        _MEGA_SM90_BUF_CACHE[key] = buf
        try:
            actual_bytes = int(buf.buffer.numel() * buf.buffer.element_size())
        except Exception:  # noqa: BLE001 - logging only
            actual_bytes = None
        logging.info(
            "[DSV4 SM90 MegaMoE] allocated symm buffer: group_size=%d num_experts=%d "
            "max_tokens_per_rank=%d topk=%d hidden=%d intermediate=%d ring_tokens=%s "
            "actual=%s",
            group.size(),
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            getattr(buf, "num_ring_tokens", "?"),
            "unavailable" if actual_bytes is None else f"{actual_bytes / 1024**3:.3f} GiB",
        )
    return buf


def _get_or_create_mega_sm90_output(capacity, hidden, dtype, device):
    key = (device, hidden, dtype)
    cached = _MEGA_SM90_OUTPUT_CACHE.get(key)
    if cached is not None and cached.size(0) >= capacity:
        return cached
    cached = torch.empty((max(capacity, 1), hidden), dtype=dtype, device=device)
    _MEGA_SM90_OUTPUT_CACHE[key] = cached
    return cached
