#!/usr/bin/env python3
"""Manifest-composed inputs and digests: one implementation for producer and verifier.

``relock.py`` stamps each lock (and ``absent_map.bzl``) with digests computed here;
``check_lock_freshness.py`` recomputes the same digests to judge staleness. Producer
and stamp-verifier must share one implementation byte for byte — a drifting copy would
silently mark every lock stale (or worse, fresh). The internal ``rtpcli`` keeps its own
port of this algorithm (``deps_sync.py``, policed by ``tests/test_deps_sync.py``)
because the public checkout cannot depend on it.

``check_manifest.py`` deliberately does NOT use the absent-map derivation from the
producer side: it is the judge of that derivation and re-derives independently.
"""

import hashlib
import json
import os
import sys

# Loaded from a [file path] by internal-source tests / Bazel tools; sys.path may not
# contain deps/, so make the sibling pkgname importable first (same as check_manifest).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pkgname import norm_marker

__all__ = [
    "render",
    "composed_inputs",
    "composed_digest",
    "absent_digest",
    "manifest_common",
    "manifest_per_profile",
    "GENERATOR_VERSION",
    "SRC_NAME",
    "ALIYUN_INDEX",
    "canonical_resolution_argv",
    "resolution_digest",
]

# Bump when lock rendering changes so every stamp invalidates and locks re-resolve once.
GENERATOR_VERSION = 2

# The uv input file is composed into a temp dir, not a committed per-profile file; the
# name lands in every lock's `#   -r <name>` provenance line, so it is part of the bytes.
SRC_NAME = "profile_requirements.txt"

ALIYUN_INDEX = "https://mirrors.aliyun.com/pypi/simple/"

OSS_HOST_TMPL = "https://{bucket}.oss-cn-hangzhou.aliyuncs.com/rtp_llm"


def _profile_entry(manifest, profile):
    for prof in manifest["profiles"]:
        if prof["name"] == profile:
            return prof
    raise KeyError(profile)


def oss_extra_index(manifest, profile):
    bucket = manifest.get("indexes", {}).get("oss_base", "rtp-opensource")
    return (
        OSS_HOST_TMPL.format(bucket=bucket)
        + "/"
        + _profile_entry(manifest, profile)["index_view"]
        + "/"
    )


def canonical_resolution_argv(manifest, profile):
    """The uv invocation that resolves one profile, with bare-name paths and no
    --constraint: the constraint file derives from (base, src, prior lock), and base/src
    are stamped separately, so it adds no resolution input the digest would miss."""
    prof = _profile_entry(manifest, profile)
    return [
        "uv",
        "pip",
        "compile",
        SRC_NAME,
        "-o",
        "generated-lock.txt",
        "--generate-hashes",
        "--python-version",
        manifest.get("python", {}).get("version", "3.10"),
        "--python-platform",
        prof["platform"],
        "--no-header",
        "--index-url",
        ALIYUN_INDEX,
        "--extra-index-url",
        oss_extra_index(manifest, profile),
        "--index-strategy",
        "unsafe-best-match",
    ]


def resolution_digest(manifest, profile, uv_pin):
    """The `# input-hash:` stamp of a lock: sha256 over EVERY input uv resolution
    depends on — generator version, the full canonical argv (src name, python version,
    platform, index urls), the pinned uv version, and the composed base/src contents.

    A stamp hit proves an unchanged resolution, so `relock --check` (and the internal
    `deps check`) can skip uv entirely instead of re-resolving every profile on each
    run; a miss resolves only the affected profiles.
    """
    base, source = composed_inputs(manifest, profile)
    blob = "\0".join(
        [str(GENERATOR_VERSION)]
        + canonical_resolution_argv(manifest, profile)
        + [uv_pin or "", base, source]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def render(name, info):
    """One requirements line: ``name<spec>`` plus a normalized `` ; marker`` suffix."""
    line = name + (info.get("spec") or "")
    marker = norm_marker(info.get("marker"))
    return line + (" ; " + marker if marker else "")


def composed_inputs(manifest, profile):
    """(base_content, src_content): base = every ``common``; src = ``-r requirements_base.txt`` + that profile's per_profile."""
    base = [
        render(p["name"], p["common"]) for p in manifest["packages"] if "common" in p
    ]
    source = ["-r requirements_base.txt"]
    for package in manifest["packages"]:
        info = package.get("per_profile", {}).get(profile)
        if info is not None:
            source.append(render(package["name"], info))
    return "\n".join(base) + "\n", "\n".join(source) + "\n"


def composed_digest(manifest, profile):
    """sha256(base_content + src_content) — the ``# input-hash:`` stamp of a lock."""
    base, source = composed_inputs(manifest, profile)
    digest = hashlib.sha256()
    digest.update(base.encode("utf-8"))
    digest.update(source.encode("utf-8"))
    return digest.hexdigest()


def absent_digest(manifest):
    """sha256 over the inputs the absent_map derivation consumes (same algorithm as deps_sync.absent_input_hash)."""
    payload = {
        "exceptions": manifest.get("exceptions", []),
        "profiles": [p["name"] for p in manifest.get("profiles", [])],
    }
    blob = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def manifest_common(manifest):
    """Common deps: {name: {spec, marker}}."""
    out = {}
    for pkg in manifest["packages"]:
        if "common" in pkg:
            out[pkg["name"]] = {
                "spec": pkg["common"].get("spec", ""),
                "marker": norm_marker(pkg["common"].get("marker")),
            }
    return out


def manifest_per_profile(manifest, profile):
    """Per-profile overrides for one profile: {name: {spec, marker}}."""
    out = {}
    for pkg in manifest["packages"]:
        pp = pkg.get("per_profile", {})
        if profile in pp:
            out[pkg["name"]] = {
                "spec": pp[profile].get("spec", ""),
                "marker": norm_marker(pp[profile].get("marker")),
            }
    return out
