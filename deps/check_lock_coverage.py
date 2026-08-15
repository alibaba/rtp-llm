#!/usr/bin/env python3
"""MODULE.bazel.lock must cover the pip extension and all 6 hubs.

A host-only hollow lock (recording only the host platform extension) is not allowed to
pass. Also guards recordedFileInputs digests: for main-repo files read by extensions via
mctx.read (deps.json, each profile lock), their sha256 is recorded in the lock; changing
the source without regenerating the lock shows no local anomaly and only blows up in the
open-source-view build (--lockfile_mode=error) -- this catches it offline in seconds.

Usage: check_lock_coverage.py <repo root>
"""
import hashlib
import json
import os
import sys

RTP_EXT_KEY = "//deps/extensions:defs.bzl%rtp_non_module_deps"


def _expected_hubs(root):
    manifest_path = os.path.join(root, "deps", "deps.json")
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        expected_hubs = {profile["hub"] for profile in manifest["profiles"]}
    except (OSError, TypeError, KeyError, ValueError) as e:
        return None, f"failed to parse deps/deps.json: {e}"
    if not expected_hubs:
        return (
            None,
            "deps/deps.json profiles have no hub -- cannot complete the lock coverage check",
        )
    return expected_hubs, None


def check_rtp_ext_section(exts):
    """The committed lock must carry the open-source-view rtp_non_module_deps section.

    The internal-source view is reproducible=True and does not enter the lock: running
    explicit --lockfile_mode=update from the internal view strips this whole section
    (update-mode ping-pong), and CI's open-source view in error mode then reports
    "does not exist in the lockfile". Also env must be recorded as null (variable
    absent): generating with --repo_env=RTP_INTERNAL_SOURCE= records an empty string,
    whose evaluation fingerprint differs from CI where the variable is truly absent, and
    error mode reports "environment variables ... have changed". Both bad shapes were
    observed on CI.
    """
    if RTP_EXT_KEY not in exts:
        return [
            f"lock is missing the {RTP_EXT_KEY} open-source section -- internal-view update mode strips it;"
            " re-evaluate from the open-source view (RTP_INTERNAL_SOURCE truly absent) and commit"
        ]
    problems = []
    for variant in exts[RTP_EXT_KEY].values():
        env = variant.get("envVariables", {})
        if "RTP_INTERNAL_SOURCE" not in env:
            continue
        if env["RTP_INTERNAL_SOURCE"] is not None:
            problems.append(
                "the rtp_non_module_deps section of the lock records RTP_INTERNAL_SOURCE as "
                f"{env['RTP_INTERNAL_SOURCE']!r} -- it must be null (variable absent);"
                " do not generate with --repo_env=RTP_INTERNAL_SOURCE=, let the variable be truly absent"
            )
    return problems


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_recorded_file_inputs(root, exts):
    """For main-repo files read by extensions, sha256 must match the lock record.

    `//deps/extensions:defs.bzl%rtp_non_module_deps` reads deps.json via mctx.read;
    rules_python's pip extension reads the six profile locks; both digests enter the
    lock's recordedFileInputs. Only labels starting with `@@//` (main repo) count;
    files of external repos are not this repo's responsibility.
    """
    problems = []
    for ext_key, variants in exts.items():
        for variant in variants.values():
            for label, want in (variant.get("recordedFileInputs") or {}).items():
                if not label.startswith("@@//"):
                    continue
                rel = label[len("@@//") :].replace(":", "/")
                path = os.path.join(root, rel)
                if not os.path.exists(path):
                    problems.append(
                        f"lock records a non-existent file {rel} (extension {ext_key})"
                    )
                    continue
                got = _sha256(path)
                if got != want:
                    problems.append(
                        f"{rel} changed but the lock was not synced (extension {ext_key}):"
                        f" recorded {want[:12]}… actual {got[:12]}…"
                    )
    return problems


def check_lock_coverage(root):
    expected_hubs, error = _expected_hubs(root)
    if error:
        return [error]

    lock_path = os.path.join(root, "MODULE.bazel.lock")
    if not os.path.exists(lock_path):
        return [
            "MODULE.bazel.lock does not exist -- run one evaluation under 7.7.1 to generate and commit it"
        ]
    try:
        with open(lock_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:
        return [f"failed to parse MODULE.bazel.lock: {e}"]
    exts = data.get("moduleExtensions", {})
    pip_key = next((k for k in exts if k.endswith("pip.bzl%pip")), None)
    if pip_key is None:
        return [
            f"lock has no pip extension (present: {sorted(exts)[:4]}…) -- coverage insufficient, re-run a full evaluation"
        ]
    repos = set()
    for variant in exts[pip_key].values():
        repos |= set(variant.get("generatedRepoSpecs", {}))
    missing = sorted(expected_hubs - repos)
    if missing:
        return [
            f"the lock's pip extension is missing hubs {missing} -- regenerate after a full-extension-coverage evaluation"
        ]
    return check_rtp_ext_section(exts) + check_recorded_file_inputs(root, exts)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    if not os.path.exists(os.path.join(root, "MODULE.bazel")):
        print(
            "FAIL: MODULE.bazel does not exist -- a Bzlmod-only repo cannot skip the lock coverage check"
        )
        sys.exit(1)
    problems = check_lock_coverage(root)
    if problems:
        print("FAIL: MODULE.bazel.lock coverage insufficient:")
        for p in problems:
            print("  " + p)
        print(
            "  Fix: from the repository root run `scripts/rtpcli bazel lock-update`, "
            "then rerun `scripts/rtpcli deps check --all` and commit MODULE.bazel.lock"
        )
        sys.exit(1)
    expected_hubs, _ = _expected_hubs(root)
    print(
        f"OK: MODULE.bazel.lock covers the pip extension and all {len(expected_hubs)} hubs"
    )


if __name__ == "__main__":
    main()
