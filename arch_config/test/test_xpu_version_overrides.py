"""Consistency test: XPU wheel version overrides must match the XPU lockfile.

``_XPU_VERSION_OVERRIDES`` in ``arch_config/arch_select.bzl`` pins the versions
that the XPU wheel advertises for a handful of packages whose CUDA/ROCm pins
differ from XPU.  Those overrides must stay in sync with the versions actually
locked in ``deps/requirements_lock_xpu.txt`` — otherwise the XPU wheel would
declare dependencies that were never validated against the build/test
environment and ``pip install`` could resolve conflicting versions.

Pure-Python test (no XPU device, no bazel-only deps required).
"""

import ast
import os
import re
from unittest import TestCase, main


def _find_repo_root(start):
    cur = os.path.abspath(start)
    while True:
        if os.path.exists(os.path.join(cur, "arch_config", "arch_select.bzl")) and \
           os.path.exists(os.path.join(cur, "deps", "requirements_lock_xpu.txt")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise RuntimeError(
                "Could not locate repo root containing arch_config/arch_select.bzl "
                "and deps/requirements_lock_xpu.txt"
            )
        cur = parent


_REPO_ROOT = _find_repo_root(os.path.dirname(__file__))
_BZL_PATH = os.path.join(_REPO_ROOT, "arch_config", "arch_select.bzl")
_LOCK_PATH = os.path.join(_REPO_ROOT, "deps", "requirements_lock_xpu.txt")


def _parse_version_overrides(bzl_path):
    """Extract the _XPU_VERSION_OVERRIDES dict literal from the .bzl file."""
    with open(bzl_path, "r", encoding="utf-8") as f:
        text = f.read()
    match = re.search(
        r"_XPU_VERSION_OVERRIDES\s*=\s*(\{.*?\})", text, re.DOTALL
    )
    if not match:
        raise AssertionError("_XPU_VERSION_OVERRIDES not found in arch_select.bzl")
    return ast.literal_eval(match.group(1))


def _normalize(name):
    return name.lower().replace("_", "-").replace(".", "-")


def _parse_lock_versions(lock_path):
    """Map normalized package name -> locked version from the XPU lockfile."""
    versions = {}
    pat = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\]+)")
    with open(lock_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                versions[_normalize(m.group(1))] = m.group(2)
    return versions


class TestXpuVersionOverrides(TestCase):
    def test_overrides_match_lockfile(self):
        overrides = _parse_version_overrides(_BZL_PATH)
        lock_versions = _parse_lock_versions(_LOCK_PATH)
        self.assertTrue(overrides, "expected non-empty _XPU_VERSION_OVERRIDES")

        mismatches = []
        for pkg, req in overrides.items():
            m = re.match(r"^([A-Za-z0-9._-]+)==([^\s]+)$", req.strip())
            self.assertIsNotNone(
                m, f"override for {pkg!r} must be an exact pin 'name==version', got {req!r}"
            )
            override_name, override_ver = m.group(1), m.group(2)
            norm = _normalize(override_name)
            self.assertIn(
                norm, lock_versions,
                f"{override_name} is overridden but not present in requirements_lock_xpu.txt",
            )
            if lock_versions[norm] != override_ver:
                mismatches.append(
                    f"{override_name}: override=={override_ver} but lockfile=={lock_versions[norm]}"
                )

        self.assertEqual(
            [], mismatches,
            "XPU wheel version overrides are out of sync with "
            "requirements_lock_xpu.txt:\n  " + "\n  ".join(mismatches),
        )


if __name__ == "__main__":
    main()
