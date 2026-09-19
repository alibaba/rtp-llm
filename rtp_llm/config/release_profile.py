"""Canonical release profiles, fail-closed startup validation and the resolved rank manifest.

Why this exists. A serving group can be started with a configuration that is individually plausible but
collectively broken: decode padding disabled, a per-rank admission cap that does not match the padded
batch, a capture set that does not contain the padded batch, geometry that the kernels were not
qualified at, test-only routing left on, or a prefill token bound above the MoE all-to-all payload
guard. None of those is a crash at argument-parse time; they surface later as a divergent collective,
an uncaptured batch shape, a silent slow path, or a group that half-serves. Validating after the fact
is expensive because a bad configuration is discovered during model load or, worse, during capture.

Design rules followed here:

* Resolve once. The snapshot is built from the same ``PyEnvConfigs`` object that is used to launch, so
  validation and the manifest cannot drift from what the process actually runs with.
* Fail closed. Anything the snapshot cannot resolve is a violation under a release profile, not a
  silent pass. An absent knob must not be treated as a correct value.
* Opt in. Nothing runs unless a release profile is selected, so developer modes and unrelated
  models/topologies are untouched.
* No new per-step collective. Cross-rank agreement is a startup comparison of manifest digests, which
  the caller can publish through the existing startup store.

What this module deliberately does NOT check, and where those belong instead:

* KV/workspace memory sufficiency and effective limits -- needs the allocator and the real device, so
  it is a cold-boot verification, not an argument-time one.
* Decode endpoint advertisement and reachability -- a runtime/health property, verified at readiness.
* Model/tokenizer/template identity and compiled-artifact hashes -- supplied by the caller as artifact
  identifiers and folded into the manifest digest, not derived here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Selects a canonical release profile. Unset means developer mode: nothing in this module runs.
RELEASE_PROFILE_ENV = "RTP_LLM_RELEASE_PROFILE"

# Knobs that are release-critical but are not represented on the resolved configuration object: the
# decode padding batch is read directly by the engine, and the transport/backend overrides are plain
# process environment. They are named explicitly here rather than silently ignored.
DECODE_FIXED_BS_ENV = "RTP_LLM_DECODE_FIXED_BS"
NCCL_P2P_LEVEL_ENV = "NCCL_P2P_LEVEL"
MOE_FP4_BACKEND_ENV = "DSV4_MOE_FP4_BACKEND"

# The MoE all-to-all dispatch rejects a batch whose summed per-rank token counts exceed this bound and
# falls back to a fixed-EP path that allocates an O(world * tokens * hidden) FP32 reduction buffer per
# layer. At prefill scale that fallback is a large latency cliff, so a prefill admission bound above it
# is a misconfiguration rather than a tuning choice.
A2A_PAYLOAD_TOKEN_BOUND = 65536


class _Unset:
    """Sentinel for a field the resolved configuration did not supply."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return "<unset>"

    def __bool__(self) -> bool:
        return False


UNSET = _Unset()


def _resolve(root: Any, path: str) -> Any:
    """Walk a dotted path over the resolved config, returning UNSET if any hop is missing.

    Returning a sentinel instead of a default is what makes validation fail closed: a knob that moved,
    was renamed or was never bound shows up as a violation under a release profile.
    """
    cur = root
    for part in path.split("."):
        if cur is None:
            return UNSET
        cur = getattr(cur, part, None)
    return UNSET if cur is None else cur


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        # A malformed value must not be read as the default; -1 is rejected by every rule below.
        return -1


@dataclass(frozen=True)
class ReleaseSnapshot:
    """The resolved, release-relevant configuration of ONE rank."""

    role_type: Any = UNSET
    rank_id: Any = UNSET
    tp_size: Any = UNSET
    dp_size: Any = UNSET
    ep_size: Any = UNSET
    concurrency_limit: Any = UNSET
    sp_type: Any = UNSET
    gen_num_per_cycle: Any = UNSET
    max_seq_len: Any = UNSET
    kernel_seq_size_per_block: Any = UNSET
    decode_capture_batch_sizes: Any = UNSET
    reuse_cache: Any = UNSET
    max_batch_tokens_size: Any = UNSET
    max_context_batch_size: Any = UNSET
    fake_balance_expert: Any = UNSET
    use_batch_decode_scheduler: Any = UNSET
    # Environment-only knobs (see the module docstring).
    decode_fixed_bs: int = 0
    nccl_p2p_level: Optional[str] = None
    moe_fp4_backend: Optional[str] = None

    def resolved_fields(self) -> Dict[str, Any]:
        """Fields that came from the resolved config object, for the manifest."""
        out: Dict[str, Any] = {}
        for name in (
            "role_type",
            "rank_id",
            "tp_size",
            "dp_size",
            "ep_size",
            "concurrency_limit",
            "sp_type",
            "gen_num_per_cycle",
            "max_seq_len",
            "kernel_seq_size_per_block",
            "decode_capture_batch_sizes",
            "reuse_cache",
            "max_batch_tokens_size",
            "max_context_batch_size",
            "fake_balance_expert",
            "use_batch_decode_scheduler",
            "decode_fixed_bs",
            "nccl_p2p_level",
            "moe_fp4_backend",
        ):
            value = getattr(self, name)
            if isinstance(value, _Unset):
                out[name] = None  # unresolved: visible in the manifest, and a violation under a profile
            elif isinstance(value, (list, tuple)):
                out[name] = [int(v) for v in value]
            else:
                out[name] = value if isinstance(value, (bool, int, float, str, type(None))) else str(value)
        return out


def snapshot_from_env_configs(cfg: Any, env: Optional[Mapping[str, str]] = None) -> ReleaseSnapshot:
    """Build the snapshot from the resolved ``PyEnvConfigs`` used to launch this rank."""
    environment: Mapping[str, str] = os.environ if env is None else env
    return ReleaseSnapshot(
        role_type=_resolve(cfg, "role_config.role_type"),
        rank_id=_resolve(cfg, "distribute_config.rank_id"),
        tp_size=_resolve(cfg, "parallelism_config.tp_size"),
        dp_size=_resolve(cfg, "parallelism_config.dp_size"),
        ep_size=_resolve(cfg, "parallelism_config.ep_size"),
        concurrency_limit=_resolve(cfg, "concurrency_config.concurrency_limit"),
        sp_type=_resolve(cfg, "sp_config.type"),
        gen_num_per_cycle=_resolve(cfg, "sp_config.gen_num_per_cycle"),
        max_seq_len=_resolve(cfg, "model_args.max_seq_len"),
        kernel_seq_size_per_block=_resolve(cfg, "kv_cache_config.kernel_seq_size_per_block"),
        decode_capture_batch_sizes=_resolve(cfg, "py_hw_kernel_config.decode_capture_batch_sizes"),
        reuse_cache=_resolve(cfg, "kv_cache_config.reuse_cache"),
        max_batch_tokens_size=_resolve(cfg, "runtime_config.fifo_scheduler_config.max_batch_tokens_size"),
        max_context_batch_size=_resolve(cfg, "runtime_config.fifo_scheduler_config.max_context_batch_size"),
        fake_balance_expert=_resolve(cfg, "moe_config.fake_balance_expert"),
        use_batch_decode_scheduler=_resolve(cfg, "runtime_config.use_batch_decode_scheduler"),
        decode_fixed_bs=_env_int(environment, DECODE_FIXED_BS_ENV, 0),
        nccl_p2p_level=environment.get(NCCL_P2P_LEVEL_ENV) or None,
        moe_fp4_backend=environment.get(MOE_FP4_BACKEND_ENV) or None,
    )


@dataclass(frozen=True)
class ReleaseProfile:
    """A canonical, version-controlled deployment configuration.

    ``role`` selects which expectations apply; a PD group validates each rank against its own role.
    ``approved_moe_backends``/``approved_nccl_p2p_levels`` are the allow-lists for knobs that are
    otherwise rejected as unapproved experimental or non-default overrides.
    """

    name: str
    # decode role
    decode_tp_size: int = 1
    decode_dp_size: int = 4
    decode_ep_size: int = 4
    decode_fixed_bs: int = 2
    decode_kernel_seq_size_per_block: int = 256
    # prefill role
    prefill_cp_size: int = 4
    prefill_ep_size: int = 4
    prefill_max_batch_tokens_size: int = 64000
    # shared
    gen_num_per_cycle: int = 3
    # Matched as a substring of the resolved enum's name, so the check does not depend on how the
    # speculative type is spelled in the configuration layer. The depth alone is not enough: the wrong
    # speculative mechanism at the right depth would be a silent behaviour change.
    speculative_type: str = "DSPARK"
    reuse_cache: bool = False
    min_max_seq_len: int = 32832
    approved_moe_backends: Tuple[str, ...] = ()
    approved_nccl_p2p_levels: Tuple[str, ...] = ()


# The N2 candidate. Two admitted decode requests per DP rank, eight configured decode slots per group.
RELEASE_PROFILES: Dict[str, ReleaseProfile] = {
    "sm120_dp4ep4_n2": ReleaseProfile(name="sm120_dp4ep4_n2"),
}


def _is_decode_role(snapshot: ReleaseSnapshot) -> bool:
    return "DECODE" in str(snapshot.role_type).upper()


def _is_prefill_role(snapshot: ReleaseSnapshot) -> bool:
    return "PREFILL" in str(snapshot.role_type).upper()


def validate(snapshot: ReleaseSnapshot, profile: ReleaseProfile) -> List[str]:
    """Return every way this rank's resolved configuration violates the profile.

    An empty list means the rank may proceed. The caller must treat a non-empty list as fatal before
    model loading or distributed capture, so a misconfigured rank never leaves its peers entering
    collectives alone.
    """
    v: List[str] = []

    def need(name: str) -> Any:
        value = getattr(snapshot, name)
        if isinstance(value, _Unset):
            v.append(f"{name}: not resolvable from the configuration object (fail closed)")
        return value

    # ---- identity --------------------------------------------------------------------------------
    role = need("role_type")
    decode = _is_decode_role(snapshot)
    prefill = _is_prefill_role(snapshot)
    if not (decode or prefill):
        v.append(f"role_type: {role!r} is neither a decode nor a prefill role for this profile")

    # ---- test-only knobs must be off in a release profile ----------------------------------------
    if need("fake_balance_expert") is True:
        v.append("fake_balance_expert: test-only forced expert routing must be disabled in release mode")
    if need("use_batch_decode_scheduler") is True:
        v.append(
            "use_batch_decode_scheduler: test-only exact-batch scheduler must be disabled in release mode"
        )

    # ---- non-default transport / unapproved experimental backends --------------------------------
    if snapshot.nccl_p2p_level is not None and snapshot.nccl_p2p_level not in profile.approved_nccl_p2p_levels:
        v.append(
            f"{NCCL_P2P_LEVEL_ENV}={snapshot.nccl_p2p_level}: non-default NCCL transport override; "
            f"approved values: {list(profile.approved_nccl_p2p_levels) or 'none (use the default transport)'}"
        )
    if snapshot.moe_fp4_backend is not None and snapshot.moe_fp4_backend not in profile.approved_moe_backends:
        v.append(
            f"{MOE_FP4_BACKEND_ENV}={snapshot.moe_fp4_backend}: unapproved experimental MoE backend; "
            f"approved values: {list(profile.approved_moe_backends) or 'none (use the default)'}"
        )

    # ---- cache reuse policy ----------------------------------------------------------------------
    reuse = need("reuse_cache")
    if not isinstance(reuse, _Unset) and bool(reuse) != profile.reuse_cache:
        v.append(
            f"reuse_cache={reuse}: profile requires {profile.reuse_cache} "
            "(prefix/device cache reuse must be separately qualified before it is enabled)"
        )

    # ---- token envelope --------------------------------------------------------------------------
    max_seq_len = need("max_seq_len")
    if not isinstance(max_seq_len, _Unset) and int(max_seq_len) < profile.min_max_seq_len:
        v.append(
            f"max_seq_len={max_seq_len} < {profile.min_max_seq_len}: requests could be admitted that this "
            "role cannot finish"
        )

    # ---- speculative (MTP) shape -----------------------------------------------------------------
    depth = need("gen_num_per_cycle")
    if not isinstance(depth, _Unset) and int(depth) != profile.gen_num_per_cycle:
        v.append(f"gen_num_per_cycle={depth}: profile requires {profile.gen_num_per_cycle}")

    sp_type = need("sp_type")
    if not isinstance(sp_type, _Unset) and profile.speculative_type not in str(sp_type).upper():
        v.append(
            f"sp_type={sp_type!r}: profile requires the {profile.speculative_type} speculative mechanism; "
            "a different mechanism at the same draft depth would silently change model behaviour"
        )

    # ---- role-specific rules ---------------------------------------------------------------------
    if decode:
        fixed_bs = snapshot.decode_fixed_bs
        if fixed_bs <= 0:
            v.append(
                f"{DECODE_FIXED_BS_ENV}={fixed_bs}: fixed decode padding must be enabled for this profile; "
                "without it ranks replay divergent graph keys under uneven batch and the expert-parallel "
                "collectives deadlock"
            )
        elif fixed_bs != profile.decode_fixed_bs:
            v.append(f"{DECODE_FIXED_BS_ENV}={fixed_bs}: profile requires {profile.decode_fixed_bs}")

        cap = need("concurrency_limit")
        if not isinstance(cap, _Unset) and fixed_bs > 0 and int(cap) != fixed_bs:
            v.append(
                f"concurrency_limit={cap} != {DECODE_FIXED_BS_ENV}={fixed_bs}: the per-rank admission cap "
                "must equal the padded batch, or a rank admits more streams than it can replay and the "
                "batch fails"
            )

        captures = need("decode_capture_batch_sizes")
        if not isinstance(captures, _Unset):
            sizes = [int(x) for x in captures]
            if fixed_bs > 0 and fixed_bs not in sizes:
                v.append(
                    f"decode_capture_batch_sizes={sizes} does not contain {DECODE_FIXED_BS_ENV}={fixed_bs}: "
                    "the padded batch has no captured graph"
                )

        for name, expected in (
            ("tp_size", profile.decode_tp_size),
            ("dp_size", profile.decode_dp_size),
            ("ep_size", profile.decode_ep_size),
        ):
            actual = need(name)
            if not isinstance(actual, _Unset) and int(actual) != expected:
                v.append(f"{name}={actual}: profile requires {expected}")

        geom = need("kernel_seq_size_per_block")
        if not isinstance(geom, _Unset) and int(geom) != profile.decode_kernel_seq_size_per_block:
            v.append(
                f"kernel_seq_size_per_block={geom}: profile requires "
                f"{profile.decode_kernel_seq_size_per_block} (the paged kernels are qualified at that geometry)"
            )

    if prefill:
        bound = need("max_batch_tokens_size")
        if not isinstance(bound, _Unset):
            bound = int(bound)
            if bound <= 0:
                v.append(
                    f"max_batch_tokens_size={bound}: an unset/auto-derived prefill admission bound admits "
                    "context batches past the MoE all-to-all payload guard"
                )
            elif bound > A2A_PAYLOAD_TOKEN_BOUND:
                v.append(
                    f"max_batch_tokens_size={bound} exceeds the MoE all-to-all payload guard "
                    f"({A2A_PAYLOAD_TOKEN_BOUND}): prefill would fall back to the fixed-EP path, which "
                    "allocates an O(world * tokens * hidden) FP32 reduction buffer per layer"
                )
            elif bound > profile.prefill_max_batch_tokens_size:
                v.append(
                    f"max_batch_tokens_size={bound} > profile value "
                    f"{profile.prefill_max_batch_tokens_size}"
                )
        for name, expected in (("ep_size", profile.prefill_ep_size),):
            actual = need(name)
            if not isinstance(actual, _Unset) and int(actual) != expected:
                v.append(f"{name}={actual}: profile requires {expected} for the prefill role")

    rank_id = need("rank_id")
    if not isinstance(rank_id, _Unset) and int(rank_id) < 0:
        v.append(f"rank_id={rank_id}: negative")

    return v


# ---------------------------------------------------------------------------
# Manifest: the sanitized resolved configuration this rank will actually run.
# ---------------------------------------------------------------------------


# Fields that legitimately differ between ranks of the same role. They are checked separately (a
# duplicate rank id is its own violation) and must not make otherwise-identical ranks look like a
# mixed configuration.
PER_RANK_FIELDS: Tuple[str, ...] = ("rank_id",)


def _group_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in config.items() if k not in PER_RANK_FIELDS}


def build_manifest(
    snapshot: ReleaseSnapshot,
    profile: ReleaseProfile,
    artifact_ids: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """A sanitized manifest: resolved configuration plus caller-supplied artifact identifiers.

    Only configuration and opaque identifiers go in, so it is safe to log and to exchange between
    ranks. Callers must not put credentials, tokens, prompts or tenant data into ``artifact_ids``.

    Two configuration digests are emitted because they answer different questions: ``config_digest``
    covers everything this rank resolved (so it identifies this rank's manifest), while
    ``group_config_digest`` excludes per-rank identity and is what peers of the same role must agree
    on.
    """
    manifest: Dict[str, Any] = {
        "profile": profile.name,
        "config": snapshot.resolved_fields(),
        "artifact": dict(artifact_ids or {}),
    }
    manifest["config_digest"] = _digest(manifest["config"])
    manifest["group_config_digest"] = _digest(_group_config(manifest["config"]))
    manifest["artifact_digest"] = _digest(manifest["artifact"])
    manifest["manifest_digest"] = _digest(
        {
            "profile": profile.name,
            "config_digest": manifest["config_digest"],
            "artifact_digest": manifest["artifact_digest"],
        }
    )
    return manifest


def _digest(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    return str(manifest.get("manifest_digest", ""))


def group_token_envelope(manifests: Mapping[Any, Mapping[str, Any]]) -> Tuple[Optional[int], Optional[int]]:
    """-> (smallest prefill max_seq_len, smallest decode max_seq_len) advertised in the group.

    Exposed separately from the violation list because a prefill leg advertising more than decode can
    finish is not by itself a rejected configuration: the prefill value sizes its own workspace, and the
    bound that actually protects a request is the public/ingress token envelope. Until that envelope is
    enforced (see the release plan's token-limit work package) the gap must stay visible rather than
    either silently passing or blocking the candidate.
    """

    def smallest(role_substr: str) -> Optional[int]:
        values = [
            int((m.get("config") or {}).get("max_seq_len"))
            for m in manifests.values()
            if role_substr in str((m.get("config") or {}).get("role_type")).upper()
            and (m.get("config") or {}).get("max_seq_len") is not None
        ]
        return min(values) if values else None

    return smallest("PREFILL"), smallest("DECODE")


def check_group_consistency(
    manifests: Mapping[Any, Mapping[str, Any]], enforce_token_envelope: bool = False
) -> List[str]:
    """Compare resolved per-rank manifests at startup. No per-step collective is involved.

    Catches mixed artifacts or configurations across ranks and duplicate rank ids within a role. The
    prefill/decode token-envelope mismatch is reported only when ``enforce_token_envelope`` is set, so
    that the rule can be turned on together with the ingress bound that actually satisfies it.
    """
    v: List[str] = []
    if not manifests:
        return ["no rank manifests supplied"]

    artifact_digests = {manifest_digest(m) for m in manifests.values()}
    by_artifact: Dict[str, List[Any]] = {}
    for rank, m in manifests.items():
        by_artifact.setdefault(str(m.get("artifact_digest", "")), []).append(rank)
    if len(by_artifact) > 1:
        v.append(f"MIXED ARTIFACTS across ranks: {by_artifact}")

    # Same role => identical resolved configuration, ignoring per-rank identity.
    by_role: Dict[str, Dict[str, List[Any]]] = {}
    for rank, m in manifests.items():
        role = str((m.get("config") or {}).get("role_type"))
        by_role.setdefault(role, {}).setdefault(str(m.get("group_config_digest", "")), []).append(rank)
    for role, digests in by_role.items():
        if len(digests) > 1:
            v.append(f"MIXED CONFIGURATION across ranks with role {role}: {digests}")

    # Duplicate rank ids, WITHIN a role. The two legs of a PD group each number their ranks from 0, so
    # the same id appearing in different roles is normal; a repeat inside one role means two processes
    # claim the same slot.
    ids_by_role: Dict[str, List[int]] = {}
    for m in manifests.values():
        cfg = m.get("config") or {}
        rid = cfg.get("rank_id")
        if rid is None:
            continue
        ids_by_role.setdefault(str(cfg.get("role_type")), []).append(int(rid))
    for role, ids in sorted(ids_by_role.items()):
        dupes = sorted({r for r in ids if ids.count(r) > 1})
        if dupes:
            v.append(f"DUPLICATE rank ids among ranks with role {role}: {dupes}")

    # Token envelope: decode must be able to finish anything prefill advertises.
    if enforce_token_envelope:
        prefill_limit, decode_limit = group_token_envelope(manifests)
        if prefill_limit is not None and decode_limit is not None and prefill_limit > decode_limit:
            v.append(
                f"TOKEN ENVELOPE MISMATCH: prefill advertises max_seq_len={prefill_limit} but decode can "
                f"only finish {decode_limit}; requests could be admitted that decode cannot complete"
            )
    return v


class ReleaseProfileError(RuntimeError):
    """Raised when a rank's resolved configuration violates the selected release profile."""


def enforce_release_profile(cfg: Any, env: Optional[Mapping[str, str]] = None,
                            artifact_ids: Optional[Mapping[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Validate the resolved config against the selected profile and log this rank's manifest.

    Returns the manifest, or ``None`` when no release profile is selected (developer mode, behaviour
    unchanged). Raises ``ReleaseProfileError`` on any violation so the process refuses to serve instead
    of loading a model it cannot run correctly.
    """
    environment: Mapping[str, str] = os.environ if env is None else env
    name = (environment.get(RELEASE_PROFILE_ENV) or "").strip()
    if not name:
        return None
    profile = RELEASE_PROFILES.get(name)
    if profile is None:
        raise ReleaseProfileError(
            f"{RELEASE_PROFILE_ENV}={name!r} is not a known release profile; "
            f"known profiles: {sorted(RELEASE_PROFILES)}"
        )

    snapshot = snapshot_from_env_configs(cfg, environment)
    manifest = build_manifest(snapshot, profile, artifact_ids)
    violations = validate(snapshot, profile)

    # The manifest is logged whether or not validation passes: a rejected start must still show what
    # was resolved, and the digest is what peers compare at startup.
    logging.info(
        "[RELEASE-PROFILE] name=%s rank=%s role=%s config_digest=%s artifact_digest=%s manifest_digest=%s",
        profile.name,
        manifest["config"].get("rank_id"),
        manifest["config"].get("role_type"),
        manifest["config_digest"][:16],
        manifest["artifact_digest"][:16],
        manifest["manifest_digest"][:16],
    )
    logging.info("[RELEASE-PROFILE] resolved manifest: %s", json.dumps(manifest, sort_keys=True, default=str))

    if violations:
        raise ReleaseProfileError(
            f"release profile {profile.name!r} rejected this rank's resolved configuration "
            f"({len(violations)} violation(s)); refusing to start so the group does not serve with a "
            "misconfigured peer:\n  - " + "\n  - ".join(violations)
        )
    return manifest
