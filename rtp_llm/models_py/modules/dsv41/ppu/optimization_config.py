"""Operator switches shared by qualified PPU execution paths.

A canonical switch overrides its phase-specific aliases. Qualified fusion
and local-Prefill/Decode switches default on; other callers keep their defaults.
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

# These fusion paths keep their existing per-call qualification.
MEGA_ATTENTION = "RTP_V41_PPU_MEGA_ATTENTION"
NORM_STORE = "RTP_V41_PPU_NORM_STORE"
MEGA_GATE = "RTP_V41_PPU_MEGA_GATE"
MEGA_GATE_FINITE_DOMAIN = "RTP_V41_PPU_MEGA_GATE_FINITE_DOMAIN"
CROSS_LAYER_MHC = "RTP_V41_PPU_CROSS_LAYER_MHC"
VERIFY_MEGA_MHC = "RTP_V41_PPU_VERIFY_MEGA_MHC"
MHC_REDUCE_SINKHORN = "RTP_V41_PPU_MHC_REDUCE_SINKHORN"
DENSE_PROJECTIONS = "RTP_V41_PPU_DENSE_PROJECTIONS"
SPARSE_TOPK = "RTP_V41_PPU_SPARSE_TOPK"
_FUSION_SWITCHES = frozenset((
    MEGA_ATTENTION, NORM_STORE, MEGA_GATE, MEGA_GATE_FINITE_DOMAIN, CROSS_LAYER_MHC,
    VERIFY_MEGA_MHC, MHC_REDUCE_SINKHORN, DENSE_PROJECTIONS, SPARSE_TOPK,
))

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
# Defaults mirror the qualified PPU fusion, cached-Prefill and Decode configuration.
# Eligibility, dependency receipts and explicit opt-outs remain caller-owned.
DEFAULT_ENABLED = frozenset((
    "RTP_V41_PPU_AIU_PRENORM",
    "RTP_V41_PPU_CP8_DENSE_INDEX_BOUNDS",
    "RTP_V41_PPU_CP8_INDEXER_FLAT_LAYOUT",
    "RTP_V41_PPU_CP8_INDEXER_REQUEST_SLICES",
    "RTP_V41_PPU_CP8_REUSE_ATTENTION_METADATA",
    "RTP_V41_PPU_CP8_YOCO_FAST_PREFILL",
    "RTP_V41_PPU_CROSS_LAYER_MHC",
    "RTP_V41_PPU_DECODE_LAUNCH_REUSE",
    "RTP_V41_PPU_DENSE_PROJECTIONS",
    "RTP_V41_PPU_DIRECT_MOE_ROUTES",
    "RTP_V41_PPU_DP_PREFILL_FORWARD_MEGA_CAPACITY",
    "RTP_V41_PPU_FUSED_BF16_ROPE",
    "RTP_V41_PPU_FUSED_KV_ROW_PACK",
    "RTP_V41_PPU_GRAPH_LM_HEAD",
    "RTP_V41_PPU_GROUPED_QO_ROPE",
    "RTP_V41_PPU_INDEXER_PAGED_PREFIX",
    "RTP_V41_PPU_LOCAL_PREFILL_AIU_PRENORM",
    "RTP_V41_PPU_LOCAL_PREFILL_ATTENTION_PROVIDER",
    "RTP_V41_PPU_LOCAL_PREFILL_COMPACT_ROWS",
    "RTP_V41_PPU_LOCAL_PREFILL_CROSS_LAYER_MHC",
    "RTP_V41_PPU_LOCAL_PREFILL_DIRECT_MOE_ROUTES",
    "RTP_V41_PPU_LOCAL_PREFILL_ENGRAM_GATE",
    "RTP_V41_PPU_LOCAL_PREFILL_ENGRAM_ROUNDS",
    "RTP_V41_PPU_LOCAL_PREFILL_FULL_ROWS",
    "RTP_V41_PPU_LOCAL_PREFILL_GROUPED_QO_ROPE",
    "RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA",
    "RTP_V41_PPU_LOCAL_PREFILL_LAZY_PRODUCER_SLOTS",
    "RTP_V41_PPU_LOCAL_PREFILL_ROTARY_PACK",
    "RTP_V41_PPU_LOCAL_PREFILL_ROWS",
    "RTP_V41_PPU_LOCAL_PREFILL_SOURCE_CACHE",
    "RTP_V41_PPU_LOCAL_PREFILL_SPLIT_L20",
    "RTP_V41_PPU_LOCAL_PREFILL_SWA_SEGMENTS",
    "RTP_V41_PPU_LOCAL_PREFILL_TOKEN_TOPK",
    "RTP_V41_PPU_LOCAL_PREFILL_WORKSPACE",
    "RTP_V41_PPU_MEGA_ATTENTION",
    "RTP_V41_PPU_MEGA_GATE",
    "RTP_V41_PPU_MEGA_GATE_FINITE_DOMAIN",
    "RTP_V41_PPU_MHC_REDUCE_SINKHORN",
    "RTP_V41_PPU_NORM_STORE",
    "RTP_V41_PPU_PTPC_FUSION",
    "RTP_V41_PPU_PTPC_WOA_DIRECT",
    "RTP_V41_PPU_SPARSE_TOPK",
    "RTP_V41_PPU_SPLIT_L20",
    "RTP_V41_PPU_SWA_BOUNDED_REPLAY",
    "RTP_V41_PPU_VERIFY_BATCHED_JOURNAL",
    "RTP_V41_PPU_VERIFY_FUSED_EPILOGUE",
    "RTP_V41_PPU_VERIFY_MEGA_MHC",
))

_CANONICAL = frozenset(ALIASES.values()) | _FUSION_SWITCHES | DEFAULT_ENABLED
_STRICT_ALIASES = frozenset(
    (
        "RTP_V41_PPU_CP8_GROUPED_QO_ROPE",
        "RTP_V41_PPU_LOCAL_PREFILL_GROUPED_QO_ROPE",
        "RTP_V41_PPU_CP8_WO_A_BMM",
    )
)


def env_value(name, default="0", *, env=None):
    """Read a qualified default without mutating the process environment."""
    env = os.environ if env is None else env
    canonical = ALIASES.get(name, name)
    if canonical in env:
        return env[canonical]
    return env.get(name, "1" if name in DEFAULT_ENABLED else default)


def enabled(name, *, default=False, env=None):
    """Resolve one operator, retaining explicit and phase-local opt-outs.

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
    value = env_value(name, "1" if default else "0", env=env)
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
