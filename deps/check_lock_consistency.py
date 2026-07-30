#!/usr/bin/env python3
"""Check consistency between pip requirements sources and their lockfiles.

For each platform (cuda12_9 / cuda12_arm / rocm) this script verifies that:
  1. Every explicit `name==version` pin in the source requirements file
     (including pins pulled in via `-r requirements_base.txt`) appears in the
     corresponding lockfile with the exact same version (PEP 440 zero-padding
     tolerated, e.g. `4.25` == `4.25.0`).
  2. The lockfile contains no unexpected direct URLs (`@ https`). Only the
     ROCm lockfile is allowed exactly one: amdsmi ships as a tar (not a
     wheel), so it cannot be served through the PEP 503 simple index and must
     stay a direct URL (see arch_config/arch_select.bzl whl_deps()).

Standard library only, no network access. Exit code is non-zero if any
platform fails. Run from anywhere: paths are resolved relative to this file.
"""

import os
import re
import sys

DEPS_DIR = os.path.dirname(os.path.abspath(__file__))

# (platform name, source requirements, lockfile, platform_machine of the lock)
PLATFORMS = [
    (
        "cuda12_9",
        "requirements_torch_gpu_cuda12_9.txt",
        "requirements_lock_torch_gpu_cuda12_9.txt",
        "x86_64",
    ),
    (
        "cuda12_arm",
        "requirements_cuda12_arm.txt",
        "requirements_lock_cuda12_arm.txt",
        "aarch64",
    ),
    (
        "rocm",
        "requirements_rocm.txt",
        "requirements_lock_rocm.txt",
        "x86_64",
    ),
]

# Allowed `@ https` direct-URL count per lockfile. rocm is exactly 1 because
# amdsmi is distributed as a tar archive which a PEP 503 index cannot serve;
# everything else must resolve through the configured indexes.
ALLOWED_DIRECT_URLS = {
    "cuda12_9": 0,
    "cuda12_arm": 0,
    "rocm": 1,
}

PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*(\S+)$")
MACHINE_RE = re.compile(r"platform_machine\s*(==|!=)\s*['\"]([^'\"]+)['\"]")


def normalize_name(name):
    """PEP 503 name normalization: lowercase, -/_/. are equivalent."""
    return re.sub(r"[-_.]+", "-", name).lower()


def canon_version(version):
    """Canonicalize a version for comparison.

    - strips trailing `.0` release segments (PEP 440: 4.25 == 4.25.0)
    - lowercases pre/post/dev suffixes and the local tag (case-insensitive
      per PEP 440)
    """
    if "+" in version:
        public, local = version.split("+", 1)
        local = "+" + local.lower()
    else:
        public, local = version, ""
    match = re.match(r"^(\d+(?:\.\d+)*)(.*)$", public)
    if not match:
        return version.lower()
    release, suffix = match.groups()
    parts = release.split(".")
    while len(parts) > 1 and parts[-1] == "0":
        parts.pop()
    return ".".join(parts) + suffix.lower() + local


def marker_applies(marker, machine):
    """Lightweight platform_machine marker evaluation.

    Only `platform_machine ==/!= "..."` clauses are interpreted (the only
    marker kind used in deps/requirements*.txt). Unknown markers are treated
    as applicable so a mismatch is reported rather than silently skipped.
    """
    for op, value in MACHINE_RE.findall(marker):
        if op == "==" and value != machine:
            return False
        if op == "!=" and value == machine:
            return False
    return True


def strip_comment(line):
    """Drop full-line and trailing ` # ...` comments."""
    if line.lstrip().startswith("#"):
        return ""
    return re.sub(r"\s+#.*$", "", line).strip()


def parse_source_pins(path, machine, seen=None):
    """Return {normalized_name: (raw_name, version)} of explicit == pins."""
    if seen is None:
        seen = set()
    real = os.path.realpath(path)
    if real in seen:
        return {}
    seen.add(real)

    pins = {}
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = strip_comment(raw)
            if not line:
                continue
            if line.startswith("-r") or line.startswith("--requirement"):
                ref = line.split(None, 1)[1].strip()
                ref_path = os.path.join(os.path.dirname(path), ref)
                pins.update(parse_source_pins(ref_path, machine, seen))
                continue
            if line.startswith("-"):  # other pip options (-c, --hash, ...)
                continue
            req, _, marker = line.partition(";")
            req = req.strip()
            if marker and not marker_applies(marker, machine):
                continue
            if " @ " in req or "@ http" in req:
                continue  # direct URLs handled by the @ https count check
            match = PIN_RE.match(req)
            if match:
                name, version = match.groups()
                pins[normalize_name(name)] = (name, version)
    return pins


def parse_lock_versions(path):
    """Return ({normalized_name: version}, direct_url_count) for a lockfile."""
    versions = {}
    direct_urls = 0
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            if "@ https" in strip_comment(raw):
                direct_urls += 1
            line = raw.rstrip("\n")
            if not line or line[0] in "# \t-":
                continue  # comments, hash/via continuations, index options
            line = line.rstrip("\\").strip()
            req = line.partition(";")[0].strip()
            match = PIN_RE.match(req)
            if match:
                name, version = match.groups()
                versions[normalize_name(name)] = version
    return versions, direct_urls


def check_platform(platform, source, lock, machine):
    errors = []
    source_path = os.path.join(DEPS_DIR, source)
    lock_path = os.path.join(DEPS_DIR, lock)

    pins = parse_source_pins(source_path, machine)
    lock_versions, direct_urls = parse_lock_versions(lock_path)

    for norm_name in sorted(pins):
        raw_name, want = pins[norm_name]
        if norm_name not in lock_versions:
            errors.append(
                "%s: pinned %s==%s in %s but missing from %s"
                % (raw_name, raw_name, want, source, lock)
            )
            continue
        got = lock_versions[norm_name]
        if canon_version(want) != canon_version(got):
            errors.append(
                "%s: source pins ==%s but lockfile has ==%s" % (raw_name, want, got)
            )

    allowed = ALLOWED_DIRECT_URLS[platform]
    if direct_urls != allowed:
        errors.append(
            "%s: found %d direct '@ https' URL(s), expected exactly %d"
            % (lock, direct_urls, allowed)
        )

    return len(pins), errors


def main():
    failed = False
    for platform, source, lock, machine in PLATFORMS:
        pin_count, errors = check_platform(platform, source, lock, machine)
        if errors:
            failed = True
            print("[FAIL] %s (%d pins checked)" % (platform, pin_count))
            for err in errors:
                print("       - %s" % err)
        else:
            print("[PASS] %s (%d pins checked)" % (platform, pin_count))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
