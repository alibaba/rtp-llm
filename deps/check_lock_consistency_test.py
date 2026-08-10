#!/usr/bin/env python3

import os
import tempfile
import unittest

from deps import check_lock_consistency as checker


class RequirementParserTest(unittest.TestCase):
    def parse(self, content):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "requirements.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            return checker.parse_source_pins(path, "x86_64")

    def test_compact_r_include(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, "base.txt")
            source = os.path.join(tmpdir, "requirements.txt")
            with open(base, "w", encoding="utf-8") as fh:
                fh.write("demo==1.2.0\n")
            with open(source, "w", encoding="utf-8") as fh:
                fh.write("-rbase.txt\n")
            pins = checker.parse_source_pins(source, "x86_64")
        self.assertEqual(pins["demo"][1], "1.2.0")

    def test_long_requirement_equals_include(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, "base.txt")
            source = os.path.join(tmpdir, "requirements.txt")
            with open(base, "w", encoding="utf-8") as fh:
                fh.write("demo==1.2.0\n")
            with open(source, "w", encoding="utf-8") as fh:
                fh.write("--requirement=base.txt\n")
            pins = checker.parse_source_pins(source, "x86_64")
        self.assertEqual(pins["demo"][1], "1.2.0")

    def test_extras_pin_is_checked_against_base_project(self):
        pins = self.parse("demo[http,security]==1.2.0\n")
        self.assertEqual(pins, {"demo": ("demo", "1.2.0")})

    def test_malformed_exact_pin_fails(self):
        with self.assertRaisesRegex(ValueError, "malformed exact pin"):
            self.parse("demo[broken==1.2.0\n")

    def test_missing_include_path_fails_cleanly(self):
        with self.assertRaisesRegex(ValueError, "missing a path"):
            self.parse("-r\n")


class LockParserTest(unittest.TestCase):
    def test_counts_http_and_https_direct_urls(self):
        content = (
            "plain @ http://example.test/plain.whl\n"
            "    --hash=sha256:" + "a" * 64 + "\n"
            "secure @ https://example.test/secure.whl\n"
            "    --hash=sha256:" + "b" * 64 + "\n"
        )
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as fh:
            fh.write(content)
            path = fh.name
        try:
            _, direct_urls, hashes = checker.parse_lock_versions(path)
        finally:
            os.unlink(path)
        self.assertEqual(direct_urls, 2)
        self.assertEqual(hashes["plain"], {"a" * 64})
        self.assertEqual(hashes["secure"], {"b" * 64})


class CrossPlatformInvariantTest(unittest.TestCase):
    def platform_pins(self):
        shared = {name: "1.2.3" for name in checker.CUDA_EXACT_VERSION_PARITY}
        return {
            "cuda12_9": dict(shared, **{"rtp-kernel": "0.1.0+125c29e5.1"}),
            "cuda12_arm": dict(shared, **{"rtp-kernel": "0.1.0+125c29e5.2"}),
        }

    def test_matching_cuda_versions_pass(self):
        self.assertEqual(
            checker.check_cross_platform_invariants(self.platform_pins()), []
        )

    def test_cuda_version_drift_fails(self):
        pins = self.platform_pins()
        pins["cuda12_arm"]["deep-gemm"] = "1.2.4"
        errors = checker.check_cross_platform_invariants(pins)
        self.assertTrue(any("deep-gemm x86/arm pins" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
