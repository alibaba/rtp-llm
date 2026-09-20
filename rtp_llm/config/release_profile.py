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
* No new per-step collective. Cross-rank agreement is a bounded, startup-only comparison of the
  published rank manifests (``join_group_and_verify``): every rank publishes to one directory and waits
  for the declared membership before model loading, so a missing, stale, mixed or differently-built peer
  stops the whole group instead of leaving the others to enter capture alone.

What this module deliberately does NOT check, and where those belong instead:

* KV/workspace memory sufficiency and effective limits -- needs the allocator and the real device, so
  it is a cold-boot verification, not an argument-time one.
* Decode endpoint advertisement and reachability -- a runtime/health property, verified at readiness.
* Model/tokenizer/template identity and compiled-artifact hashes -- supplied by the caller as artifact
  identifiers and folded into the manifest digest, not derived here.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Selects a canonical release profile. Unset means developer mode: nothing in this module runs.
RELEASE_PROFILE_ENV = "RTP_LLM_RELEASE_PROFILE"

# Optional explicit provenance label injected by the packaging step (image digest, release tag, ...).
# When absent the manifest still carries a non-empty identity derived from the running files, so a
# manifest is never a digest of an empty artifact record.
ARTIFACT_LABEL_ENV = "RTP_LLM_RELEASE_ARTIFACT_ID"

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

# Startup group verification (section 4.2 / 15.2): every rank publishes its manifest to a directory
# both legs can see and waits, bounded, for the profile's declared membership before it loads a model.
# A peer that never arrives (crashed, rejected locally, stuck) times out every other rank, so a partial
# group cannot start; a peer that arrives with a different config or artifact is a rejection. This is a
# startup-only file rendezvous, not a collective and not a per-step consensus.
GROUP_DIR_ENV = "RTP_LLM_RELEASE_MANIFEST_DIR"
GROUP_ATTEMPT_ENV = "RTP_LLM_RELEASE_ATTEMPT"
GROUP_TIMEOUT_ENV = "RTP_LLM_RELEASE_JOIN_TIMEOUT_S"
DEFAULT_GROUP_TIMEOUT_S = 120.0
GROUP_POLL_S = 0.25

# Artifact identity is CONTENT, not size/mtime: two builds or two checkpoints can share a size and an
# mtime and still be different binaries. The cost is bounded instead of avoided -- the compiled
# libraries and small metadata files are hashed in full, while multi-hundred-GB weight shards are hashed
# to a bounded head+tail sample that is marked as such, never passed off as a full digest. A packaging
# step that already hashed the weights can supply those digests and have their coverage enforced.
FULL_HASH_MAX_ENV = "RTP_LLM_RELEASE_FULL_HASH_MAX_MB"
SAMPLE_BYTES_ENV = "RTP_LLM_RELEASE_SAMPLE_MB"
WEIGHTS_MANIFEST_ENV = "RTP_LLM_RELEASE_WEIGHTS_MANIFEST"
VERIFY_WEIGHTS_MANIFEST_ENV = "RTP_LLM_RELEASE_VERIFY_WEIGHTS_MANIFEST"
REQUIRE_FULL_WEIGHTS_ENV = "RTP_LLM_RELEASE_REQUIRE_FULL_WEIGHTS"
DEFAULT_FULL_HASH_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_SAMPLE_BYTES = 1024 * 1024
MAX_TREE_FILES = 4096
MAX_LISTED_FILES = 64
# The libraries the process actually loads: the engine/ops shared objects and the kernel/collective
# libraries shipped in the package's libs directory.
LIBRARY_DIR = "libs"
LIBRARY_GLOBS = ("libth_transformer*.so", "librtp_compute_ops*.so")


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


def _call(root: Any, path: str) -> Any:
    """Invoke a zero-argument method on the resolved config, UNSET if it is absent or raises.

    Some release-relevant facts are only exposed as methods (for example whether context parallelism
    is enabled for prefill), and calling one must never be able to abort startup on its own.
    """
    cur = root
    for part in path.split("."):
        if cur is None:
            return UNSET
        cur = getattr(cur, part, None)
    if not callable(cur):
        return UNSET
    try:
        return cur()
    except Exception:  # noqa: BLE001 - a probe must not be able to fail startup by itself
        return UNSET


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
    world_size: Any = UNSET
    tp_size: Any = UNSET
    dp_size: Any = UNSET
    ep_size: Any = UNSET
    concurrency_limit: Any = UNSET
    enable_cuda_graph: Any = UNSET
    # CP facts, named after the engine's own predicates (ConfigModules.h): is_enabled() means THIS leg
    # computes context parallelism (ALL_GATHER / ..._WITH_OVERLAP / ALLTOALL), while is_prefill_enabled()
    # means this leg CONSUMES a CP prefill (method == PREFILL_CP, used by the decode side). Conflating
    # them rejects a correct prefill leg.
    prefill_cp_method: Any = UNSET
    cp_computes: Any = UNSET
    cp_consumes_prefill: Any = UNSET
    prefill_cp_kv_cache_sharded: Any = UNSET
    prefill_cp_size: Any = UNSET
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
            "world_size",
            "tp_size",
            "dp_size",
            "ep_size",
            "concurrency_limit",
            "enable_cuda_graph",
            "prefill_cp_method",
            "cp_computes",
            "cp_consumes_prefill",
            "prefill_cp_kv_cache_sharded",
            "prefill_cp_size",
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
        world_size=_resolve(cfg, "parallelism_config.world_size"),
        tp_size=_resolve(cfg, "parallelism_config.tp_size"),
        dp_size=_resolve(cfg, "parallelism_config.dp_size"),
        ep_size=_resolve(cfg, "parallelism_config.ep_size"),
        concurrency_limit=_resolve(cfg, "concurrency_config.concurrency_limit"),
        enable_cuda_graph=_resolve(cfg, "py_hw_kernel_config.enable_cuda_graph"),
        prefill_cp_method=_resolve(cfg, "prefill_cp_config.method"),
        cp_computes=_call(cfg, "prefill_cp_config.is_enabled"),
        cp_consumes_prefill=_call(cfg, "prefill_cp_config.is_prefill_enabled"),
        prefill_cp_kv_cache_sharded=_resolve(cfg, "prefill_cp_config.kv_cache_sharded"),
        prefill_cp_size=_resolve(cfg, "prefill_cp_config.prefill_cp_size"),
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


def _truthy(raw: Optional[str]) -> bool:
    """An env switch: only an explicit affirmative value enables it."""
    return (raw or "").strip().lower() in ("1", "true", "yes", "on")


def _env_bytes(environment: Mapping[str, str], name: str, default: int) -> int:
    """An env value in MiB as bytes, falling back to the default for anything unusable."""
    raw = environment.get(name)
    if not raw:
        return default
    try:
        value = int(float(raw) * 1024 * 1024)
    except ValueError:
        return default
    return value if value > 0 else default


def _content_digest(
    path: str,
    full_max_bytes: int,
    sample_bytes: int,
    force_full: bool = False,
) -> Dict[str, Any]:
    """A CONTENT identity for one file: full sha256 when affordable, else a marked head+tail sample.

    ``mode`` says which one it is, so a sampled digest is never presented as a verified full one and a
    comparison against a packaging-supplied digest stays meaningful.
    """
    size = int(os.stat(path).st_size)
    if force_full or size <= full_max_bytes:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        return {"size": size, "sha256": h.hexdigest(), "mode": "full"}
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(sample_bytes))
        if size > sample_bytes:
            f.seek(max(size - sample_bytes, sample_bytes))
            h.update(f.read(sample_bytes))
    h.update(str(size).encode())
    return {"size": size, "sha256": h.hexdigest(), "mode": "sampled", "sample_bytes": sample_bytes}


def _tree_identity(root: str, full_max_bytes: int, sample_bytes: int) -> Dict[str, Any]:
    """Content identity of a model/tokenizer directory tree, file by file.

    The COMPLETE inventory is kept for validation: coverage comparison and full-hash verification need
    every file, only the *reporting* copy is capped. A file that cannot be read, and a tree too large to
    walk in full, are errors rather than silent omissions -- an unreadable shard must not leave the
    remaining tree looking complete.
    """
    files: Dict[str, Any] = {}
    errors: List[str] = []
    unreadable: List[str] = []
    complete = True
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if len(files) >= MAX_TREE_FILES:
                complete = False
                errors.append(
                    f"the tree has more than {MAX_TREE_FILES} files, so it was not walked in full and no "
                    "digest over it can be complete"
                )
                break
            path = os.path.join(dirpath, name)
            try:
                files[os.path.relpath(path, root)] = _content_digest(path, full_max_bytes, sample_bytes)
            except OSError as e:
                unreadable.append(f"{os.path.relpath(path, root)}: {e}")
        if not complete:
            break
    if unreadable:
        complete = False
        errors.append(f"{len(unreadable)} file(s) could not be read: {unreadable[:3]}")
    ident: Dict[str, Any] = {
        "kind": "tree",
        "root": root,
        "file_count": len(files),
        "total_bytes": sum(int(v["size"]) for v in files.values()),
        "sha256": _digest(files),
        "complete": complete,
        "mode": ("full" if all(v["mode"] == "full" for v in files.values()) else "sampled")
        if complete and files
        else "incomplete",
        "files": files,
    }
    if errors:
        ident["errors"] = errors
    return ident


def _tree_for_report(ident: Mapping[str, Any]) -> Dict[str, Any]:
    """The report copy of a tree identity: the aggregate covers every file, the listing is capped."""
    if "files" not in ident:
        return dict(ident)
    out = dict(ident)
    files = out.pop("files")
    listed = sorted(files)
    out["files_listed"] = len(listed)
    out["files"] = {name: files[name] for name in listed[:MAX_LISTED_FILES]}
    return out


def _weights_manifest(
    path: str,
    tree: Optional[Dict[str, Any]],
    verify_full: bool = False,
    full_max_bytes: int = DEFAULT_FULL_HASH_MAX_BYTES,
    sample_bytes: int = DEFAULT_SAMPLE_BYTES,
) -> Tuple[Dict[str, Any], List[str]]:
    """Read a packaging-supplied ``<sha256>  <path>`` digest list and check it against the tree.

    A digest list is a DECLARATION: on its own it proves nothing about the bytes on disk, so it is never
    reported as verified unless ``verify_full`` re-hashed every covered file and they matched. Coverage
    is always required over the tree's COMPLETE inventory (a capped listing must not make an uncovered
    tree look covered). Returns (manifest identity, errors).
    """
    errors: List[str] = []
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        return {}, [f"cannot read {path}: {e}"]
    declared: Dict[str, str] = {}
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(" *", "  ").split()
        if len(parts) != 2 or len(parts[0]) != 64 or any(c not in "0123456789abcdefABCDEF" for c in parts[0]):
            errors.append(f"malformed digest line: {line[:80]!r}")
            continue
        declared[parts[1].lstrip("./")] = parts[0].lower()
    ident: Dict[str, Any] = {
        "path": path,
        "declared_files": len(declared),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "verified": False,
    }
    if tree is None:
        return ident, errors
    if not tree.get("complete", False):
        errors.append(
            "the weights tree could not be walked in full, so digest coverage and verification cannot "
            "be established"
        )
        return ident, errors
    files = tree.get("files") or {}
    missing = sorted(
        name for name in files
        if name not in declared and os.path.basename(name) not in declared
    )
    if missing:
        errors.append(
            f"the weights digest manifest does not cover {len(missing)} file(s), e.g. {missing[:3]}"
        )
    ident["covered"] = len(files) - len(missing)
    if verify_full:
        root = str(tree.get("root") or "")
        mismatched: List[str] = []
        for name in sorted(files):
            want = declared.get(name) or declared.get(os.path.basename(name))
            if want is None:
                continue
            try:
                got = _content_digest(os.path.join(root, name), 0, 0, force_full=True)["sha256"]
            except OSError as e:
                mismatched.append(f"{name} (unreadable: {e})")
                continue
            if got != want:
                mismatched.append(name)
        if mismatched:
            errors.append(
                f"{len(mismatched)} declared digest(s) do not match the mounted tree, e.g. "
                f"{mismatched[:3]}"
            )
        else:
            ident["verified"] = True
            ident["verified_files"] = len(files)
    return ident, errors


def _select_libraries(ordered: Sequence[str]) -> Tuple[Dict[str, str], Dict[str, List[str]], List[str]]:
    """Pick one file per library name, in the order the loader would search, and report shadowed copies.

    The first candidate for a name wins (the search order is the loader's); every later candidate that
    is a DIFFERENT file (not a symlink alias of the same file) is recorded as shadowed and makes the
    identity ambiguous, because the manifest cannot prove which of them the loader opened.
    """
    selected: Dict[str, str] = {}
    shadowed: Dict[str, List[str]] = {}
    for path in ordered:
        name = os.path.basename(path)
        real = os.path.realpath(path)
        if name not in selected:
            selected[name] = path
            shadowed[name] = []
            continue
        if os.path.realpath(selected[name]) == real or real in shadowed[name]:
            continue
        shadowed[name].append(path)
    errors = [
        f"{name} exists as more than one distinct file ({[selected[name]] + shadowed[name]}); the "
        "loader picks one of them, so this manifest cannot say which binary is running"
        for name in sorted(shadowed)
        if shadowed[name]
    ]
    return selected, shadowed, errors


def _resolve_libraries() -> Tuple[Dict[str, str], Dict[str, List[str]], List[str]]:
    """The compiled shared objects the deployment loads -> ({name: selected path}, shadowed, errors).

    A library is identified by its BASENAME, because that is how the loader resolves it. Two DIFFERENT
    files sharing one name on two search paths therefore make the identity ambiguous: whichever the
    loader picks, a manifest that described the other would be wrong. The search order here mirrors the
    loader's (the package's libs directory, then beside and above the package, then the import path).
    """
    try:
        import rtp_llm
    except Exception:  # noqa: BLE001
        return {}, {}, ["rtp_llm could not be imported, so no compiled library could be identified"]
    pkg = os.path.dirname(os.path.abspath(rtp_llm.__file__))
    search_dirs = [os.path.join(pkg, LIBRARY_DIR), pkg, os.path.dirname(pkg)]
    for entry in list(sys.path)[:32]:
        if entry and os.path.isdir(entry):
            search_dirs.append(entry)
    ordered = sorted(glob.glob(os.path.join(pkg, LIBRARY_DIR, "*.so")))
    for directory in search_dirs:
        for pattern in LIBRARY_GLOBS:
            ordered.extend(sorted(glob.glob(os.path.join(directory, pattern))))
    return _select_libraries(ordered)


def artifact_identity_from(cfg: Any, env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Non-empty, CONTENT-based provenance for this rank: the binaries it loaded and the weights it
    will serve.

    Size and mtime are not an identity -- two builds can share both -- so every compiled library and
    every model/tokenizer file contributes a content digest, and a library whose name resolves to more
    than one distinct file is an error rather than a coin flip. The packaging step can add an explicit
    release label (image digest or tag) through ``RTP_LLM_RELEASE_ARTIFACT_ID`` and, for weights too
    large to re-hash at startup, a digest manifest through ``RTP_LLM_RELEASE_WEIGHTS_MANIFEST``: its
    coverage is required, and it counts as proof only when it was VERIFIED against the mounted tree
    (``RTP_LLM_RELEASE_VERIFY_WEIGHTS_MANIFEST``, implied by ``RTP_LLM_RELEASE_REQUIRE_FULL_WEIGHTS``).
    Anything that could not be established is reported in ``errors`` so the caller can refuse the start
    instead of proceeding with an unknown artifact.
    """
    environment: Mapping[str, str] = os.environ if env is None else env
    full_max = _env_bytes(environment, FULL_HASH_MAX_ENV, DEFAULT_FULL_HASH_MAX_BYTES)
    sample = _env_bytes(environment, SAMPLE_BYTES_ENV, DEFAULT_SAMPLE_BYTES)
    require_full = _truthy(environment.get(REQUIRE_FULL_WEIGHTS_ENV))
    verify_manifest = require_full or _truthy(environment.get(VERIFY_WEIGHTS_MANIFEST_ENV))
    ids: Dict[str, Any] = {}
    errors: List[str] = []
    label = (environment.get(ARTIFACT_LABEL_ENV) or "").strip()
    if label:
        ids["release_label"] = label

    # Compiled libraries: always hashed in full. The set is bounded (the engine plus the kernel and
    # collective libraries) and it is the code that actually runs, so a sampled digest would be the
    # wrong trade here. Each entry records the path that was hashed, so a different copy of the same
    # name is visible in the manifest itself.
    selected, shadowed, lib_errors = _resolve_libraries()
    errors.extend(lib_errors)
    libs: Dict[str, Any] = {}
    for name, path in sorted(selected.items()):
        try:
            entry = _content_digest(path, full_max, sample, force_full=True)
        except OSError as e:
            errors.append(f"cannot hash library {path}: {e}")
            continue
        entry["path"] = path
        if shadowed.get(name):
            entry["shadowed_copies"] = list(shadowed[name])
        libs[name] = entry
    if libs:
        ids["libraries"] = {
            "count": len(libs),
            "total_bytes": sum(int(v["size"]) for v in libs.values()),
            "sha256": _digest(libs),
            "files": libs,
        }
    else:
        errors.append("no compiled library could be identified, so the running binary is not identified")

    manifest_path = (environment.get(WEIGHTS_MANIFEST_ENV) or "").strip()
    # The same tree can be referenced through several config paths (a model directory is also its own
    # tokenizer directory); hash it once and reference the result.
    seen_trees: Dict[str, str] = {}
    for key, path in (
        ("model", _resolve(cfg, "model_args.ckpt_path")),
        ("tokenizer", _resolve(cfg, "model_args.tokenizer_path")),
        ("sp_model", _resolve(cfg, "sp_config.checkpoint_path")),
    ):
        if isinstance(path, _Unset) or not path:
            continue
        path = str(path)
        try:
            real = os.path.realpath(path)
        except OSError:
            continue
        if real in seen_trees:
            ids[key] = {"name": os.path.basename(path.rstrip("/")), "same_as": seen_trees[real]}
            continue
        try:
            if os.path.isdir(path):
                ident = _tree_identity(path, full_max, sample)
            elif os.path.isfile(path):
                ident = _content_digest(path, full_max, sample)
            else:
                errors.append(f"{key} path does not exist: {path}")
                continue
        except OSError as e:
            errors.append(f"cannot identify {key} at {path}: {e}")
            continue
        ident["name"] = os.path.basename(path.rstrip("/"))
        errors.extend(ident.pop("errors", []))
        if manifest_path and ident.get("kind") == "tree":
            wm, wm_errors = _weights_manifest(
                manifest_path, ident, verify_full=verify_manifest,
                full_max_bytes=full_max, sample_bytes=sample)
            if wm:
                ident["weights_manifest"] = wm
            errors.extend(wm_errors)
        # Strict mode accepts only a FULL identity that was actually established: a sampled tree with an
        # unverified (or zero) digest list is a declaration, not proof.
        if require_full and ident.get("mode") != "full":
            verified = bool((ident.get("weights_manifest") or {}).get("verified"))
            if not verified:
                errors.append(
                    f"{key} identity is '{ident.get('mode')}': {REQUIRE_FULL_WEIGHTS_ENV} requires a "
                    f"verified full content digest -- raise {FULL_HASH_MAX_ENV}, or supply a weights "
                    f"manifest ({WEIGHTS_MANIFEST_ENV}) that is verified against the mounted tree"
                )
        ids[key] = _tree_for_report(ident)
        seen_trees[real] = key
    if errors:
        ids["errors"] = errors
    return ids


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
    # The decode leg must replay captured graphs: without capture it is not the qualified candidate,
    # and a configured capture set alone says nothing if graph execution is switched off.
    decode_enable_cuda_graph: bool = True
    # The decode leg names how the prefill sharded its KV, and how many ways.
    decode_cp_rotate_method: str = "PREFILL_CP"
    decode_prefill_cp_size: int = 4
    # prefill role: context parallelism is what makes this the CP4 leg, so its topology, CP mode and
    # PD KV-sharing contract are all part of the candidate rather than incidental settings.
    prefill_tp_size: int = 4
    prefill_dp_size: int = 1
    prefill_ep_size: int = 4
    prefill_kernel_seq_size_per_block: int = 256
    prefill_cp_rotate_method: str = "ALL_GATHER"
    prefill_max_batch_tokens_size: int = 64000
    # shared
    pd_kv_cache_sharded: bool = True
    # Membership: each PD leg numbers its own ranks from 0, so these are per-leg rank counts. The group
    # comparison requires exactly this many manifests per role.
    expected_decode_ranks: int = 4
    expected_prefill_ranks: int = 4
    # A manifest must identify what it is running; an empty artifact record identifies nothing.
    require_artifact_identity: bool = True
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
        # Graph execution must actually be on. A configured capture set is not evidence that the
        # captured shapes are ever replayed, and a candidate serving path that silently ran eager
        # would not be the configuration that was qualified.
        graph = need("enable_cuda_graph")
        if not isinstance(graph, _Unset) and bool(graph) != profile.decode_enable_cuda_graph:
            v.append(
                f"enable_cuda_graph={graph}: the decode leg must replay captured graphs "
                f"(profile requires {profile.decode_enable_cuda_graph}); a configured capture set alone "
                "does not mean the captured shapes are used"
            )

        # The decode leg states how the prefill sharded its KV; that contract is what makes a PD pair
        # mutually compatible.
        cp_method = need("prefill_cp_method")
        if not isinstance(cp_method, _Unset) and profile.decode_cp_rotate_method not in str(cp_method).upper():
            v.append(
                f"cp_rotate_method={cp_method!r}: the decode leg of this profile expects "
                f"{profile.decode_cp_rotate_method}"
            )
        cp_size = need("prefill_cp_size")
        if not isinstance(cp_size, _Unset) and int(cp_size) != profile.decode_prefill_cp_size:
            v.append(f"prefill_cp_size={cp_size}: profile requires {profile.decode_prefill_cp_size}")
        # With a sharded KV, the decode side must declare that it consumes a CP prefill and must size it:
        # the engine CHECKs prefill_cp_size > 1 at allocation time for exactly this combination, and a
        # preflight rejection is cheaper than that assertion.
        consumes = need("cp_consumes_prefill")
        if not isinstance(consumes, _Unset) and not bool(consumes):
            v.append(
                "prefill_cp_config.is_prefill_enabled() is false: the decode leg of this profile must "
                "declare that it consumes a context-parallel prefill"
            )
        if (
            not isinstance(cp_size, _Unset)
            and int(cp_size) <= 1
            and bool(snapshot.prefill_cp_kv_cache_sharded) is True
        ):
            v.append(
                f"prefill_cp_size={cp_size}: a sharded prefill KV requires an explicit prefill CP size > 1"
            )

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
        # The prefill leg is the CP4 leg: its topology, CP mode and PD KV-sharing contract are part of
        # the candidate. Checking the topology only on the decode leg would accept a prefill running a
        # different parallelism arrangement entirely.
        for name, expected in (
            ("tp_size", profile.prefill_tp_size),
            ("dp_size", profile.prefill_dp_size),
            ("ep_size", profile.prefill_ep_size),
        ):
            actual = need(name)
            if not isinstance(actual, _Unset) and int(actual) != expected:
                v.append(f"{name}={actual}: profile requires {expected} for the prefill role")

        geom = need("kernel_seq_size_per_block")
        if not isinstance(geom, _Unset) and int(geom) != profile.prefill_kernel_seq_size_per_block:
            v.append(
                f"kernel_seq_size_per_block={geom}: the prefill leg must use the qualified geometry "
                f"{profile.prefill_kernel_seq_size_per_block}"
            )

        cp_method = need("prefill_cp_method")
        if not isinstance(cp_method, _Unset) and profile.prefill_cp_rotate_method not in str(cp_method).upper():
            v.append(
                f"cp_rotate_method={cp_method!r}: the prefill leg of this profile expects "
                f"{profile.prefill_cp_rotate_method}"
            )
        cp_on = need("cp_computes")
        if not isinstance(cp_on, _Unset) and not bool(cp_on):
            v.append(
                "prefill_cp_config.is_enabled() is false: this leg is the CP4 leg and must actually "
                "compute context parallelism"
            )

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
    # PD KV-sharing contract, checked on both legs: they must agree on whether the prefill shards its
    # paged KV across context-parallel ranks.
    sharded = need("prefill_cp_kv_cache_sharded")
    if not isinstance(sharded, _Unset) and bool(sharded) != profile.pd_kv_cache_sharded:
        v.append(
            f"prefill_cp_kv_cache_sharded={sharded}: profile requires {profile.pd_kv_cache_sharded} "
            "(the PD pairing assumes a matching KV-sharing contract on both legs)"
        )

    rank_id = need("rank_id")
    if not isinstance(rank_id, _Unset) and int(rank_id) < 0:
        v.append(f"rank_id={rank_id}: negative")

    # A rank outside its own leg cannot take part in the collectives this profile assumes, so it must
    # not start. Each leg numbers its own ranks, so the expected count is the role's rank count, not the
    # two legs added together.
    world = need("world_size")
    if not isinstance(world, _Unset):
        expected_ranks = profile.expected_decode_ranks if decode else profile.expected_prefill_ranks
        if int(world) != expected_ranks:
            v.append(
                f"world_size={world}: this role's profile declares {expected_ranks} rank(s) per leg"
            )
        if not isinstance(rank_id, _Unset) and not (0 <= int(rank_id) < int(world)):
            v.append(
                f"rank_id={rank_id} is outside the group [0, {world}): a rank outside the declared "
                "membership cannot join the collectives this profile assumes"
            )

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
    manifests: Mapping[Any, Mapping[str, Any]],
    enforce_token_envelope: bool = False,
    duplicates: Optional[Mapping[Any, Sequence[Mapping[str, Any]]]] = None,
) -> List[str]:
    """Compare resolved per-rank manifests at startup. No per-step collective is involved.

    Catches mixed artifacts or configurations across ranks, duplicate rank ownership (two claimants for
    one slot, passed in by the rendezvous rather than collapsed away) and duplicate rank ids within a
    role. The prefill/decode token-envelope mismatch is reported only when ``enforce_token_envelope``
    is set, so that the rule can be turned on together with the ingress bound that actually satisfies
    it.
    """
    v: List[str] = []
    if not manifests:
        return ["no rank manifests supplied"]

    # Two processes claiming one slot is not agreement between ranks: it is an identity collision, and the
    # extra claimants must be visible even though the manifest map holds only one of them.
    for key, claimants in sorted((duplicates or {}).items(), key=lambda kv: str(kv[0])):
        digests = sorted({str(c.get("manifest_digest", ""))[:16] for c in claimants})
        role, rank = key if isinstance(key, tuple) else (str(key), "?")
        v.append(
            f"DUPLICATE RANK OWNERSHIP: {role}#{rank} was claimed by {len(claimants) + 1} processes "
            f"(extra manifest digests {digests}); a rank slot has exactly one publisher"
        )

    # The profile each rank names drives the membership and provenance requirements, so the caller does
    # not have to remember to pass them.
    profile_names = {str(m.get("profile", "")) for m in manifests.values()}
    profile = RELEASE_PROFILES.get(next(iter(profile_names))) if len(profile_names) == 1 else None
    if len(profile_names) > 1:
        v.append(f"MIXED RELEASE PROFILES across ranks: {sorted(profile_names)}")

    # Different artifacts in one group: the ranks would be running different binaries or weights while
    # still entering the same collectives.
    by_artifact: Dict[str, List[Any]] = {}
    for rank, m in manifests.items():
        by_artifact.setdefault(str(m.get("artifact_digest", "")), []).append(rank)
    if len(by_artifact) > 1:
        v.append(f"MIXED ARTIFACTS across ranks: {by_artifact}")

    # Required provenance: a manifest with no artifact identity cannot say what it is running, which is
    # exactly the case that must not be read as agreement between ranks.
    if profile is not None and profile.require_artifact_identity:
        no_identity = sorted(str(r) for r, m in manifests.items() if not m.get("artifact"))
        if no_identity:
            v.append(
                f"MISSING ARTIFACT IDENTITY for ranks {no_identity}: an empty artifact record identifies "
                "nothing, so these manifests cannot establish what is running"
            )

    # Declared membership: a group is only validated when every rank it expects is present, with the
    # expected identity. A partial group must not be reported as consistent.
    if profile is not None:
        for role_substr, expected in (
            ("PREFILL", profile.expected_prefill_ranks),
            ("DECODE", profile.expected_decode_ranks),
        ):
            present = sorted(
                int((m.get("config") or {}).get("rank_id"))
                for m in manifests.values()
                if role_substr in str((m.get("config") or {}).get("role_type")).upper()
                and (m.get("config") or {}).get("rank_id") is not None
            )
            if len(present) != expected:
                v.append(
                    f"INCOMPLETE MEMBERSHIP for {role_substr}: {len(present)} rank manifest(s) {present}, "
                    f"the profile declares {expected}"
                )
            elif present != list(range(expected)):
                v.append(
                    f"MEMBERSHIP MISMATCH for {role_substr}: ranks {present}, expected {list(range(expected))}"
                )

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


def _manifest_identity(manifest: Mapping[str, Any]) -> Tuple[str, int]:
    cfg = manifest.get("config") or {}
    role = str(cfg.get("role_type") or "UNKNOWN")
    rank = cfg.get("rank_id")
    if rank is None:
        raise ReleaseProfileError("a rank manifest without a rank_id cannot be published or compared")
    return role, int(rank)


def publish_rank_manifest(manifest: Mapping[str, Any], directory: str, attempt: str) -> str:
    """Claim this rank's slot by publishing its manifest, atomically and EXCLUSIVELY.

    A rank is a single slot: a second process claiming the same ``(role, rank)`` is a misconfiguration
    or an identity collision, not an update to accept. The manifest is written to a private temporary
    file and then linked into place, which fails with ``EEXIST`` when the slot is already claimed, so
    two publishers can never silently overwrite each other. The attempt id keeps a previous start's
    records from being mistaken for this one's (section 15.2: stale records must not be accepted).
    """
    role, rank = _manifest_identity(manifest)
    safe_role = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in role)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{attempt}__{safe_role}__{rank}.json")
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(manifest, f, sort_keys=True, default=str)
    try:
        os.link(tmp, path)  # atomic create-if-absent: fails when the slot is already claimed
    except FileExistsError:
        raise ReleaseProfileError(
            f"rank {role}#{rank} is already claimed in {directory}; a rank slot has exactly one "
            "publisher, so two processes claiming it cannot be treated as agreement"
        )
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return path


def scan_group_directory(
    directory: str, attempt: str
) -> Tuple[Dict[Tuple[str, int], dict], int, Dict[Tuple[str, int], List[dict]]]:
    """-> ({(role, rank): manifest}, files from other attempts, {(role, rank): [extra claimants]}).

    Identity comes from the manifest content, never from the file name. Duplicate claimants for one
    slot are preserved, not collapsed: the group must see the collision and refuse it, so the earliest
    claimant is kept in the manifest map and every further claimant is retained separately. An
    unreadable file is skipped: a peer can be observed mid-publish, and the caller's deadline decides
    whether the group is complete.
    """
    manifests: Dict[Tuple[str, int], dict] = {}
    duplicates: Dict[Tuple[str, int], List[dict]] = {}
    stale = 0
    prefix = attempt + "__"
    try:
        names = os.listdir(directory)
    except OSError:
        return manifests, stale, duplicates
    for name in names:
        if not name.endswith(".json") or ".tmp-" in name:
            continue
        if not name.startswith(prefix):
            stale += 1
            continue
        try:
            with open(os.path.join(directory, name)) as f:
                manifest = json.load(f)
            key = _manifest_identity(manifest)
        except (OSError, ValueError, ReleaseProfileError):
            continue
        if key in manifests:
            duplicates.setdefault(key, []).append(manifest)
        else:
            manifests[key] = manifest
    return manifests, stale, duplicates


def collect_group_manifests(
    directory: str,
    attempt: str,
    expected_total: int,
    timeout_s: Optional[float] = None,
    env: Optional[Mapping[str, str]] = None,
    clock=time.monotonic,
    sleep=time.sleep,
) -> Tuple[Dict[Tuple[str, int], dict], int, bool, Dict[Tuple[str, int], List[dict]]]:
    """Wait, bounded, for ``expected_total`` distinct rank manifests of this attempt.

    -> (manifests, stale_files, timed_out, duplicate_claimants). The bounded wait IS the group-wide
    abort: a rank whose peer never publishes cannot proceed to load a model, so one failed peer stops
    the group rather than leaving the others entering capture alone. A non-finite timeout is rejected
    rather than accepted: ``nan`` never compares greater than the clock and ``inf`` never expires, so
    either would turn the bounded wait into an unbounded one.
    """
    environment = os.environ if env is None else env
    if timeout_s is None:
        raw = environment.get(GROUP_TIMEOUT_ENV)
        try:
            timeout_s = float(raw) if raw else DEFAULT_GROUP_TIMEOUT_S
        except ValueError:
            timeout_s = DEFAULT_GROUP_TIMEOUT_S
        if not math.isfinite(float(timeout_s)):
            raise ReleaseProfileError(
                f"{GROUP_TIMEOUT_ENV}={raw!r} is not a finite number of seconds; a non-finite budget "
                "would never expire, which is exactly the unbounded wait this rendezvous must not have"
            )
    timeout_s = float(timeout_s)
    if not math.isfinite(timeout_s):
        raise ReleaseProfileError(f"join timeout {timeout_s!r} is not finite")
    deadline = clock() + max(timeout_s, 0.0)
    while True:
        manifests, stale, duplicates = scan_group_directory(directory, attempt)
        if len(manifests) >= expected_total:
            return manifests, stale, False, duplicates
        if clock() >= deadline:
            return manifests, stale, True, duplicates
        sleep(GROUP_POLL_S)


def join_group_and_verify(
    manifest: Mapping[str, Any],
    env: Optional[Mapping[str, str]] = None,
    directory: Optional[str] = None,
    attempt: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> List[str]:
    """Publish this rank's manifest and verify the group it is about to join. -> violations.

    Called from the rank process (after its real rank is known, before a model is loaded), never from
    the per-leg parent, which has no rank identity of its own. Returns violations instead of raising so
    the caller can report them together with the local ones; an empty list means the declared group is
    present, complete and homogeneous.
    """
    environment = os.environ if env is None else env
    profile = RELEASE_PROFILES.get(str(manifest.get("profile") or ""))
    if profile is None:
        return [f"GROUP VERIFICATION: manifest names unknown profile {manifest.get('profile')!r}"]
    directory = directory or (environment.get(GROUP_DIR_ENV) or "").strip()
    if not directory:
        return [
            f"GROUP VERIFICATION NOT CONFIGURED: {GROUP_DIR_ENV} is unset, so this rank cannot establish "
            "that the group it is about to enter is complete and homogeneous"
        ]
    attempt = attempt or (environment.get(GROUP_ATTEMPT_ENV) or "").strip()
    if not attempt:
        return [
            f"GROUP VERIFICATION NOT CONFIGURED: {GROUP_ATTEMPT_ENV} is unset; without an attempt id a "
            "manifest from an earlier start could be accepted as one of this group's"
        ]
    expected_total = int(profile.expected_decode_ranks) + int(profile.expected_prefill_ranks)
    try:
        path = publish_rank_manifest(manifest, directory, attempt)
    except ReleaseProfileError as e:
        # The slot is already claimed: this rank must not overwrite the incumbent and must not treat two
        # processes as one rank's agreement.
        return [str(e)]
    logging.info(
        "[RELEASE-PROFILE] published %s; waiting for %d rank manifests of attempt %s in %s",
        path, expected_total, attempt, directory,
    )
    manifests, stale, timed_out, duplicates = collect_group_manifests(
        directory, attempt, expected_total, timeout_s, environment
    )
    if timed_out:
        seen = sorted(f"{role}#{rank}" for role, rank in manifests)
        return [
            f"GROUP INCOMPLETE: {len(manifests)} of {expected_total} rank manifests after the bounded "
            f"wait; seen {seen}. A rank that never publishes stops every rank, so no partial group may "
            "proceed to model loading"
        ]
    violations = check_group_consistency(manifests, duplicates=duplicates)
    if not violations:
        logging.info(
            "[RELEASE-PROFILE] group verified: %d ranks, digests agree%s",
            len(manifests),
            f" ({stale} manifest(s) from other attempts ignored)" if stale else "",
        )
    return violations


# The canonical, runnable settings of each profile live beside this module as JSON, so the deployment
# or benchmark that consumes them and the validator that checks them read ONE versioned artifact rather
# than two hand-maintained descriptions of the same thing. The JSON is generated from the definitions
# below by ``dump_profile_json`` and a unit test fails if the checked-in file drifts from it.
PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "release_profiles")


def dump_profile_json(name: str, indent: int = 2) -> str:
    """The canonical resolved settings of a profile, as JSON, for a runner to consume."""
    profile = RELEASE_PROFILES[name]
    payload = {
        "profile": profile.name,
        "per_leg": {
            "decode": {
                "tp_size": profile.decode_tp_size,
                "dp_size": profile.decode_dp_size,
                "ep_size": profile.decode_ep_size,
                "world_size": profile.expected_decode_ranks,
                "fixed_bs": profile.decode_fixed_bs,
                "concurrency_limit": profile.decode_fixed_bs,
                "capture_batch_sizes": list(range(1, profile.decode_fixed_bs + 1)),
                "enable_cuda_graph": profile.decode_enable_cuda_graph,
                "cp_rotate_method": profile.decode_cp_rotate_method,
                "prefill_cp_size": profile.decode_prefill_cp_size,
                "kernel_seq_size_per_block": profile.decode_kernel_seq_size_per_block,
                "max_seq_len": profile.min_max_seq_len,
                "reuse_cache": profile.reuse_cache,
            },
            "prefill": {
                "tp_size": profile.prefill_tp_size,
                "dp_size": profile.prefill_dp_size,
                "ep_size": profile.prefill_ep_size,
                "world_size": profile.expected_prefill_ranks,
                "cp_rotate_method": profile.prefill_cp_rotate_method,
                "prefill_cp_kv_cache_sharded": profile.pd_kv_cache_sharded,
                "kernel_seq_size_per_block": profile.prefill_kernel_seq_size_per_block,
                "max_batch_tokens_size": profile.prefill_max_batch_tokens_size,
                "reuse_cache": profile.reuse_cache,
            },
        },
        "shared": {
            "pd_kv_cache_sharded": profile.pd_kv_cache_sharded,
            "expected_decode_ranks": profile.expected_decode_ranks,
            "expected_prefill_ranks": profile.expected_prefill_ranks,
            "gen_num_per_cycle": profile.gen_num_per_cycle,
            "speculative_type": profile.speculative_type,
            "reuse_cache": profile.reuse_cache,
            "nccl_transport": "default",
            "moe_backend": "default",
        },
    }
    return json.dumps(payload, indent=indent, sort_keys=True) + "\n"


def profile_json_path(name: str) -> str:
    return os.path.join(PROFILE_DIR, f"{name}.json")


class ReleaseProfileError(RuntimeError):
    """Raised when a rank's resolved configuration violates the selected release profile."""


def enforce_release_profile(cfg: Any, env: Optional[Mapping[str, str]] = None,
                            artifact_ids: Optional[Mapping[str, Any]] = None,
                            join_group: bool = False) -> Optional[Dict[str, Any]]:
    """Validate the resolved config against the selected profile and log this rank's manifest.

    Returns the manifest, or ``None`` when no release profile is selected (developer mode, behaviour
    unchanged). Raises ``ReleaseProfileError`` on any violation so the process refuses to serve instead
    of loading a model it cannot run correctly.

    ``join_group`` is set only by a rank process whose real rank is known: it publishes this rank's
    manifest and requires the profile's declared membership to appear, with agreeing configuration and
    artifact digests, within a bounded wait before the caller loads a model.
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
    artifacts = dict(artifact_ids) if artifact_ids is not None else artifact_identity_from(cfg, environment)
    manifest = build_manifest(snapshot, profile, artifacts)
    violations = validate(snapshot, profile)

    # Provenance is part of the profile contract: an empty artifact record identifies nothing, so the
    # manifest could not tell one deployment from another or detect a stale one. Anything the identity
    # could not establish (an unhashable library, a weights manifest that does not cover the tree, a
    # required full digest that is only sampled) is a violation too: an unknown artifact must not start.
    if profile.require_artifact_identity and not manifest["artifact"]:
        violations.append(
            "artifact identity is empty: the manifest must identify the running artifact (set "
            f"{ARTIFACT_LABEL_ENV}, or ensure the ops library and model path resolve)"
        )
    for problem in manifest["artifact"].get("errors", []) if isinstance(manifest["artifact"], Mapping) else []:
        violations.append(f"artifact identity incomplete: {problem}")

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
    # A compact, greppable line for run evidence: how the artifact was identified and whether anything
    # about it could not be established (a sampled weights digest, a missing manifest, ...).
    artifact = manifest.get("artifact") if isinstance(manifest.get("artifact"), Mapping) else {}
    libs = artifact.get("libraries") or {}
    weights = artifact.get("model") or {}
    logging.info(
        "[RELEASE-PROFILE] artifact: libraries=%s libraries_digest=%s weights_mode=%s weights_files=%s "
        "identity_errors=%d",
        libs.get("count"), str(libs.get("sha256", ""))[:16], weights.get("mode"),
        weights.get("file_count"), len(artifact.get("errors", []) or []),
    )

    if violations:
        raise ReleaseProfileError(
            f"release profile {profile.name!r} rejected this rank's resolved configuration "
            f"({len(violations)} violation(s)); refusing to start so the group does not serve with a "
            "misconfigured peer:\n  - " + "\n  - ".join(violations)
        )

    if join_group:
        group_violations = join_group_and_verify(manifest, environment, timeout_s=None)
        if group_violations:
            logging.error(
                "[RELEASE-PROFILE] group verification failed for rank=%s role=%s: %s",
                manifest["config"].get("rank_id"), manifest["config"].get("role_type"),
                "; ".join(group_violations),
            )
            raise ReleaseProfileError(
                f"release profile {profile.name!r} rejected this rank's group "
                f"({len(group_violations)} violation(s)); refusing to start so no partial or mixed "
                "group enters collectives:\n  - " + "\n  - ".join(group_violations)
            )
    return manifest
