"""Startup topology validation for SM120 DeepSeek-V4 worker profiles."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from rtp_llm.device.cuda_capability import (
    participating_cuda_capabilities,
    participating_sm120_state,
)
from rtp_llm.ops import RoleType, SpeculativeType

# Speculative types whose fused engine step pads a missing prefill or decode
# phase (NormalEngine::isMTPEagle). Keep in sync with the C++ predicate.
_SM120_PHASE_PADDING_SPECULATIVE_TYPES = (
    SpeculativeType.MTP,
    SpeculativeType.EAGLE,
    SpeculativeType.DSPARK,
    SpeculativeType.DFLASH,
)


@dataclass(frozen=True)
class _Sm120ProfileSpec:
    required: str
    matches: Callable[[Any, bool], bool]
    canonical_strategy: Callable[[Any], Optional[str]]


_SM120_DSV4_PROFILES: dict[RoleType, _Sm120ProfileSpec] = {
    RoleType.PREFILL: _Sm120ProfileSpec(
        required="TP4/CP4/DP1/EP4/world4 PREFILL",
        matches=lambda p, cp: (
            p.tp_size == 4
            and p.dp_size == 1
            and p.ep_size == 4
            and p.world_size == 4
            and cp
        ),
        canonical_strategy=lambda p: "sm120_fused_moe",
    ),
    RoleType.DECODE: _Sm120ProfileSpec(
        required="TP1/DP4/EP4/world4 DECODE (CP off)",
        matches=lambda p, cp: (
            p.tp_size == 1
            and p.dp_size == 4
            and p.ep_size == 4
            and p.world_size == 4
            and not cp
        ),
        canonical_strategy=lambda p: "sm120_a2a_decode",
    ),
    RoleType.PDFUSION: _Sm120ProfileSpec(
        required="TP1 DP=EP=world PDFUSION (PP1, no local CP)",
        matches=lambda p, cp: (
            p.tp_size == 1 and p.dp_size == p.ep_size == p.world_size and not cp
        ),
        canonical_strategy=lambda p: "sm120_a2a_decode" if p.dp_size > 1 else None,
    ),
}


def _resolve_canonical_strategy(
    role: RoleType,
    p: Any,
    moe: Any,
    spec: _Sm120ProfileSpec,
) -> str:
    canonical = spec.canonical_strategy(p)
    raw = getattr(moe, "moe_strategy", "auto")
    current = (raw.strip() if isinstance(raw, str) else "") or "auto"
    if canonical is None:
        if current not in ("auto", "grouped_fp4"):
            raise ValueError(
                f"Unsupported single-GPU SM120 DSv4 {role} moe_strategy={current}. "
                "Required: auto or grouped_fp4."
            )
        return current
    if current == "auto":
        moe.moe_strategy = canonical
        logging.info("Resolved SM120 DSv4 %s moe_strategy to %s", role, canonical)
        return canonical
    if current != canonical:
        raise ValueError(
            f"Unsupported SM120 DSv4 {role} moe_strategy={current}. "
            f"Required: {canonical}."
        )
    return current


def _is_unpadded_decode_pdfusion(
    role: RoleType,
    p: Any,
    strategy: str,
    *,
    pads_phases: bool,
    cp_on: bool,
) -> bool:
    if role != RoleType.PDFUSION or strategy != "sm120_a2a_decode":
        return False
    if pads_phases or cp_on or p.pp_size != 1 or p.tp_size != 1 or p.dp_size <= 1:
        return False
    return p.dp_size == p.ep_size == p.world_size


def _validate_execution_guards(
    role: RoleType,
    p: Any,
    strategy: str,
    *,
    cp_on: bool,
    pads_phases: bool,
    graph_on: bool,
    actual: str,
) -> None:
    if _is_unpadded_decode_pdfusion(
        role, p, strategy, pads_phases=pads_phases, cp_on=cp_on
    ):
        raise ValueError(
            "Unsupported SM120 DSv4 sm120_a2a_decode PDFUSION profile "
            f"without MTP, EAGLE, DSPARK or DFLASH speculation: {actual}. "
            "Only those modes pad a missing prefill/decode phase; use one of "
            "them or separate PREFILL/DECODE roles."
        )
    if strategy == "sm120_a2a_decode" and graph_on:
        raise ValueError(
            "sm120_a2a_decode requires eager execution: CUDA graphs are "
            "unsupported for the SM120 DSv4 DECODE and "
            "PDFUSION(DP>1) collective protocol; set enable_cuda_graph=false"
        )
    if cp_on and strategy == "sm120_fused_moe" and graph_on:
        raise ValueError(
            f"sm120_fused_moe CP{p.tp_size} PREFILL requires eager execution: "
            "CUDA graphs are unsupported for the SM120 DSv4 pure "
            f"context-parallel MoE path ({actual}); set enable_cuda_graph=false"
        )


def validate_sm120_dsv4_worker_profile(py_env_configs: Any) -> None:
    """Validate supported SM120 DeepSeek-V4 worker topologies and MoE strategies.

    Keep in sync with ``rtp_llm/cpp/normal_engine/Sm120MoeStrategyPolicy.h`` and
    ``rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/strategy/fp8_fp4.py``.
    """
    role = py_env_configs.role_config.role_type
    spec = _SM120_DSV4_PROFILES.get(role)
    if spec is None:
        return
    p = py_env_configs.parallelism_config
    sm120_state = participating_sm120_state(p.local_world_size)
    if sm120_state == "none":
        return
    if sm120_state == "mixed":
        caps = participating_cuda_capabilities(p.local_world_size)
        raise ValueError(
            "Unsupported mixed SM120 and non-SM120 participating GPUs for "
            f"DeepSeek-V4 worker: local_world_size={p.local_world_size}, "
            f"capabilities={caps}. Restrict CUDA_VISIBLE_DEVICES to homogeneous GPUs."
        )

    cp_cfg = getattr(p, "prefill_cp_config", None)
    cp_on = bool(cp_cfg and cp_cfg.is_enabled())
    spec_type = py_env_configs.sp_config.type
    spec_on = spec_type != SpeculativeType.NONE
    pads_phases = spec_type in _SM120_PHASE_PADDING_SPECULATIVE_TYPES
    graph_on = bool(
        getattr(
            getattr(py_env_configs, "py_hw_kernel_config", None),
            "enable_cuda_graph",
            False,
        )
    )
    actual = (
        f"role={role}, pp={p.pp_size}, tp={p.tp_size}, dp={p.dp_size}, "
        f"ep={p.ep_size}, world={p.world_size}, prefill_cp={cp_on}, "
        f"speculative={spec_on}"
    )

    if p.pp_size != 1 or not spec.matches(p, cp_on):
        required = "pp_size=1 (non-PP)" if p.pp_size != 1 else spec.required
        raise ValueError(
            f"Unsupported SM120 (compute capability 12.0) DSv4 worker profile: "
            f"{actual}. Required: {required}."
        )

    moe = py_env_configs.moe_config
    deep_ep = getattr(py_env_configs, "deep_ep_config", None)
    if bool(getattr(moe, "use_deepep_moe", False)) or (
        deep_ep is not None and getattr(deep_ep, "use_deepep_moe", None) is True
    ):
        raise ValueError(
            "DeepEP is unsupported on SM120 (compute capability 12.0) for "
            f"DeepSeek-V4 workers ({actual}); remove use_deepep_moe=true"
        )
    if bool(getattr(moe, "use_mori_ep", False)) or (
        deep_ep is not None and getattr(deep_ep, "use_mori_ep", None) is True
    ):
        raise ValueError(
            "MoriEP is unsupported on SM120 (compute capability 12.0) for "
            f"DeepSeek-V4 workers ({actual}); remove use_mori_ep=true"
        )

    strategy = _resolve_canonical_strategy(role, p, moe, spec)
    _validate_execution_guards(
        role,
        p,
        strategy,
        cp_on=cp_on,
        pads_phases=pads_phases,
        graph_on=graph_on,
        actual=actual,
    )
