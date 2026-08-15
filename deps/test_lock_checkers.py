#!/usr/bin/env python3
"""Focused regression tests for the dependency lock checkers."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FLAVOR = load("check_lock_flavor")
ARTIFACTS = load("check_lock_artifacts")
COVERAGE = load("check_lock_coverage")
MODULE_PIP = load("check_module_pip")
MANIFEST = load("check_manifest")
PROFILE_GUARDS = load("check_profile_guards")
MIRROR = load("check_mirror_coverage")

# check_mirror_coverage needs network (HEADs the published objects), so it runs through
# `rtpcli deps verify`, not the offline gate.
GATE_EXEMPT_CHECKERS = {"check_mirror_coverage.py"}


class GateRegistrationTest(unittest.TestCase):
    def test_every_checker_is_registered_in_gate(self):
        """A checker gate.sh forgets is dead code, and gate still prints all-green."""
        gate = (HERE / "gate.sh").read_text(encoding="utf-8")
        found = {p.name for p in HERE.glob("check_*.py")} | {
            p.name for p in HERE.glob("check_*.sh")
        }
        self.assertTrue(found, "no checkers discovered")
        missing = sorted(
            name for name in found - GATE_EXEMPT_CHECKERS if name not in gate
        )
        self.assertEqual(missing, [], "checkers not registered in gate.sh")


class LockCheckerTest(unittest.TestCase):
    def _assert_missing_module_fails(self, module):
        with tempfile.TemporaryDirectory() as root:
            old_argv = sys.argv
            sys.argv = [module.__file__, root]
            try:
                with self.assertRaises(SystemExit) as raised:
                    with contextlib.redirect_stdout(io.StringIO()):
                        module.main()
                self.assertEqual(raised.exception.code, 1)
            finally:
                sys.argv = old_argv

    def test_lock_coverage_missing_module_fails(self):
        self._assert_missing_module_fails(COVERAGE)

    def test_lock_coverage_uses_supplied_root_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            (root_path / "MODULE.bazel").write_text("", encoding="utf-8")
            deps_path = root_path / "deps"
            deps_path.mkdir()
            (deps_path / "deps.json").write_text(
                json.dumps({"profiles": [{"hub": "custom_hub"}]}),
                encoding="utf-8",
            )
            lock = {
                "moduleExtensions": {
                    COVERAGE.RTP_EXT_KEY: {
                        "general": {"envVariables": {"RTP_INTERNAL_SOURCE": None}}
                    },
                    "//python/extensions:pip.bzl%pip": {
                        "general": {"generatedRepoSpecs": {"custom_hub": {}}}
                    },
                }
            }
            (root_path / "MODULE.bazel.lock").write_text(
                json.dumps(lock), encoding="utf-8"
            )

            self.assertEqual(COVERAGE.check_lock_coverage(root), [])

    def test_module_pip_missing_module_fails(self):
        self._assert_missing_module_fails(MODULE_PIP)

    def test_manifest_schema_validation_fails_closed_without_jsonschema(self):
        with mock.patch.dict(sys.modules, {"jsonschema": None}):
            problems, ran = MANIFEST.validate_with_jsonschema({}, {})
        self.assertFalse(ran)
        self.assertTrue(
            any("jsonschema is required" in problem for problem in problems)
        )

    def test_profile_guard_loads_private_absence_and_config_facts(self):
        manifest = {"profiles": [{"name": "cpu"}], "exceptions": []}
        private = {
            "profiles": {
                "cuda13": {
                    "absent": ["flash-attn"],
                    "config_settings": ["using_cuda13_x86"],
                },
                "cuda13_arm": {
                    "absent": [],
                    "config_settings": ["using_cuda13_arm"],
                },
                "ppu": {"absent": [], "config_settings": ["using_ppu"]},
            }
        }
        absent = PROFILE_GUARDS.load_absent(manifest, private)
        self.assertEqual(absent["flash-attn"], {"cuda13"})
        config_profiles = PROFILE_GUARDS.private_config_profiles(private)
        self.assertEqual(config_profiles["using_cuda13_x86"], ["cuda13"])
        self.assertEqual(config_profiles["using_ppu"], ["ppu"])
        # cuda13_arm must not be folded into cuda13: their absence facts differ.
        self.assertEqual(config_profiles["using_cuda13_arm"], ["cuda13_arm"])

    def test_profile_guard_rejects_private_profile_without_config_settings(self):
        private = {"profiles": {"ppu": {"lock": "internal_source/deps/x.txt"}}}
        problems = PROFILE_GUARDS.private_profile_problems(private)
        self.assertTrue(
            any("ppu is missing config_settings" in problem for problem in problems)
        )

    def test_mirror_coverage_scans_manifest_cc_view_urls(self):
        url = (
            "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/"
            "archives/example.whl"
        )
        with tempfile.TemporaryDirectory() as root:
            deps = Path(root) / "deps"
            deps.mkdir()
            (deps / "deps.json").write_text(
                json.dumps(
                    {
                        "packages": [
                            {
                                "name": "example",
                                "per_profile": {"rocm": {"cc_view": {"urls": [url]}}},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            declared = MIRROR.declared_urls(root)
            self.assertEqual(
                declared[url],
                ["deps/deps.json.packages[0].per_profile.rocm.cc_view.urls[0]"],
            )

    def test_flavor_missing_lock_fails(self):
        old_locks = FLAVOR.LOCKS
        try:
            FLAVOR.LOCKS = {"cpu": ("missing.txt", "")}
            with tempfile.TemporaryDirectory() as root:
                with self.assertRaises(SystemExit) as raised:
                    with contextlib.redirect_stdout(io.StringIO()):
                        old_argv = sys.argv
                        sys.argv = ["check_lock_flavor.py", root]
                        try:
                            FLAVOR.main()
                        finally:
                            sys.argv = old_argv
                self.assertEqual(raised.exception.code, 1)
        finally:
            FLAVOR.LOCKS = old_locks

    def test_flavor_zero_checked_pins_fails(self):
        old_locks = FLAVOR.LOCKS
        try:
            FLAVOR.LOCKS = {"cpu": ("empty.txt", "")}
            with tempfile.TemporaryDirectory() as root:
                Path(root, "empty.txt").write_text("numpy==1.0\n", encoding="utf-8")
                with self.assertRaises(SystemExit) as raised:
                    with contextlib.redirect_stdout(io.StringIO()):
                        old_argv = sys.argv
                        sys.argv = ["check_lock_flavor.py", root]
                        try:
                            FLAVOR.main()
                        finally:
                            sys.argv = old_argv
                self.assertEqual(raised.exception.code, 1)
        finally:
            FLAVOR.LOCKS = old_locks

    def test_cross_lock_compares_all_previous_hash_sets(self):
        manifest = {
            "profiles": [
                {"name": "a", "platform": "x86_64-unknown-linux-gnu", "lock": "a.txt"},
                {"name": "b", "platform": "x86_64-unknown-linux-gnu", "lock": "b.txt"},
                {"name": "c", "platform": "x86_64-unknown-linux-gnu", "lock": "c.txt"},
            ]
        }
        parsed = {
            "a.txt": {"pkg": ("1.0", {"a", "b"})},
            "b.txt": {"pkg": ("1.0", {"a"})},
            "c.txt": {"pkg": ("1.0", {"b"})},
        }
        problems, total = ARTIFACTS.check_cross_lock(manifest, parsed)
        self.assertEqual(total, 1)
        self.assertTrue(any("b.txt and c.txt" in p for p in problems))

    def test_artifact_missing_expected_lock_fails(self):
        manifest = {
            "profiles": [
                {
                    "name": "cpu",
                    "platform": "x86_64-unknown-linux-gnu",
                    "lock": "cpu.txt",
                }
            ],
            "packages": [],
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / "deps.json").write_text(json.dumps(manifest), encoding="utf-8")
            old_deps = ARTIFACTS.DEPS
            try:
                ARTIFACTS.DEPS = path
                with self.assertRaises(SystemExit) as raised:
                    with contextlib.redirect_stdout(io.StringIO()):
                        ARTIFACTS.main()
                self.assertEqual(raised.exception.code, 1)
            finally:
                ARTIFACTS.DEPS = old_deps

    def test_declared_artifact_missing_from_lock_fails(self):
        manifest = {
            "profiles": [
                {
                    "name": "cpu",
                    "platform": "x86_64-unknown-linux-gnu",
                    "lock": "cpu.txt",
                }
            ],
            "packages": [
                {
                    "name": "custom-wheel",
                    "per_profile": {"cpu": {"sha256": "a" * 64}},
                }
            ],
        }
        problems, count = ARTIFACTS.check_against_declared(manifest, {"cpu.txt": {}})
        self.assertEqual(count, 1)
        self.assertTrue(any("does not contain this package" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
