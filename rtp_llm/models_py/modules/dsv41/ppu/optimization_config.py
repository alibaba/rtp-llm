"""Operator switches shared by qualified PPU execution paths.

A canonical switch overrides its phase-specific aliases. Without it, each
caller retains its historical default and each alias retains its parser.
These switches select implementations; callers still enforce eligibility.
"""

import os

GROUPED_QO_ROPE = "RTP_V41_PPU_GROUPED_QO_ROPE"
BATCHED_WO_A = "RTP_V41_PPU_BATCHED_WO_A"
DIRECT_MOE_ROUTES = "RTP_V41_PPU_DIRECT_MOE_ROUTES"
AIU_PRENORM = "RTP_V41_PPU_AIU_PRENORM"
SPLIT_L20 = "RTP_V41_PPU_SPLIT_L20"
LAZY_PRODUCER_SLOTS = "RTP_V41_PPU_LAZY_PRODUCER_SLOTS"
MHC_POST = "RTP_V41_PPU_MHC_POST"
MHC_REDUCE_BLOCK_M = "RTP_V41_PPU_MHC_REDUCE_BLOCK_M"

ALIASES = {
    "RTP_V41_PPU_CP8_GROUPED_QO_ROPE": GROUPED_QO_ROPE,
    "RTP_V41_PPU_LOCAL_PREFILL_GROUPED_QO_ROPE": GROUPED_QO_ROPE,
    "RTP_V41_PPU_CP8_WO_A_BMM": BATCHED_WO_A,
    "RTP_V41_PPU_GRAPH_BATCHED_WO_A": BATCHED_WO_A,
    "RTP_V41_PPU_PROPOSAL_BATCHED_WO_A": BATCHED_WO_A,
    "RTP_V41_PPU_CP8_DIRECT_MOE_ROUTES": DIRECT_MOE_ROUTES,
    "RTP_V41_PPU_LOCAL_PREFILL_DIRECT_MOE_ROUTES": DIRECT_MOE_ROUTES,
    "RTP_V41_PPU_VERIFY_DIRECT_MOE_ROUTES": DIRECT_MOE_ROUTES,
    "RTP_V41_PPU_CP8_YOCO_AIU_PRENORM": AIU_PRENORM,
    "RTP_V41_PPU_LOCAL_PREFILL_AIU_PRENORM": AIU_PRENORM,
    "RTP_V41_PPU_CP8_YOCO_SPLIT_L20": SPLIT_L20,
    "RTP_V41_PPU_LOCAL_PREFILL_SPLIT_L20": SPLIT_L20,
    "RTP_V41_PPU_CP8_LAZY_PRODUCER_SLOTS": LAZY_PRODUCER_SLOTS,
    "RTP_V41_PPU_LOCAL_PREFILL_LAZY_PRODUCER_SLOTS": LAZY_PRODUCER_SLOTS,
}
VALUE_ALIASES = {
    "RTP_V41_PPU_VERIFY_MHC_POST": MHC_POST,
    "RTP_V41_PPU_PROPOSAL_MHC_POST": MHC_POST,
    "RTP_V41_PPU_VERIFY_MHC_REDUCE_BLOCK_M": MHC_REDUCE_BLOCK_M,
    "RTP_V41_PPU_PROPOSAL_MHC_REDUCE_BLOCK_M": MHC_REDUCE_BLOCK_M,
}
_CANONICAL = frozenset(ALIASES.values())
_STRICT_ALIASES = frozenset(
    (
        "RTP_V41_PPU_CP8_GROUPED_QO_ROPE",
        "RTP_V41_PPU_LOCAL_PREFILL_GROUPED_QO_ROPE",
        "RTP_V41_PPU_CP8_WO_A_BMM",
    )
)


def enabled(name, *, default=False, env=None):
    """Resolve one operator, preserving unset and legacy-only behavior.

    Canonical values are explicitly 0/1. They take precedence even when an
    alias is set to the opposite value. Aliases stay phase-local: setting a
    CP8 alias does not enable a local-Prefill or Graph alias.
    """
    env = os.environ if env is None else env
    canonical = ALIASES.get(name, name)
    if canonical not in _CANONICAL:
        raise ValueError(f"unknown PPU optimization switch: {name}")
    if canonical in env:
        value = env[canonical]
        if value not in ("0", "1"):
            raise ValueError(f"{canonical} must be explicitly 0 or 1")
        return value == "1"
    value = env.get(name, "1" if default else "0")
    if name in (
        "RTP_V41_PPU_GRAPH_BATCHED_WO_A",
        "RTP_V41_PPU_PROPOSAL_BATCHED_WO_A",
    ):
        # This alias historically used config.v41_runtime.env_bool.
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{name} must be a boolean, got {normalized!r}")
    if name in _STRICT_ALIASES and value not in ("0", "1"):
        raise ValueError(f"{name} must be explicitly 0 or 1")
    return value == "1"


def value(name, *, default, env=None):
    """Resolve a mode/tuning value; the consumer retains its validation."""
    env = os.environ if env is None else env
    canonical = VALUE_ALIASES.get(name, name)
    if canonical not in VALUE_ALIASES.values():
        raise ValueError(f"unknown PPU optimization setting: {name}")
    return env.get(canonical, env.get(name, default))


def graph_mhc_options(phase, *, reduce_block_m, env=None):
    """Shared Verify/Proposal settings after the caller admits the bucket."""
    if phase is None:
        return None, None
    if phase not in ("VERIFY", "PROPOSAL"):
        raise ValueError(f"unsupported Graph mHC phase: {phase}")
    post_env = f"RTP_V41_PPU_{phase}_MHC_POST"
    post = value(post_env, default="dot", env=env)
    if post not in ("off", "dot", "epilogue"):
        raise ValueError(f"{post_env} must be off, dot or epilogue")
    rows = None
    if reduce_block_m:
        reduce_env = f"RTP_V41_PPU_{phase}_MHC_REDUCE_BLOCK_M"
        raw = value(reduce_env, default="1", env=env)
        if raw not in ("0", "1", "2", "4", "8"):
            raise ValueError(f"{reduce_env} must be 0, 1, 2, 4 or 8")
        if raw != "0":
            rows = int(raw)
    return (None if post == "off" else post), rows
