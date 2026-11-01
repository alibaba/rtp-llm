"""CPU-only boundary geometry, independent exact patterns and private files."""
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from uuid import uuid4

from tools.engram_sidecar.framework import large_offset_probe as p


class LargeOffsetContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.root.chmod(0o700)
        self.path = self.root / "probe.json"
        granularity = 2 << 20
        padded = p.LOGICAL_BYTES + (-p.LOGICAL_BYTES) % granularity
        self.document = p.metadata(padded, granularity, str(uuid4()))
        self.document["handle"] = base64.b64encode(b"private-test-capability").decode()

    def write(self):
        p.private_write(self.path, json.dumps(self.document).encode())

    def test_fixed_geometry_crosses_32_bit_byte_offset(self):
        self.assertEqual(p.ROWS, 16777473)
        self.assertEqual(p.LOGICAL_BYTES, 4429252872)
        self.assertGreater(p.WEIGHT_BYTES, 1 << 32)
        self.assertEqual(p.SAMPLE_ROWS[1] * p.DIM + p.DIM - 1, (1 << 32) - 1)
        self.assertEqual(p.SAMPLE_ROWS[2] * p.DIM, 1 << 32)
        self.assertEqual(p.SAMPLE_ROWS[-1] * p.DIM + p.DIM, p.WEIGHT_BYTES)
        self.assertEqual(p.WEIGHT_BYTES + p.SAMPLE_ROWS[-1] * p.SCALE_COLUMNS + p.SCALE_COLUMNS,
                         p.LOGICAL_BYTES)

    def test_four_rows_have_distinct_exact_fp8_scale_patterns(self):
        patterns = [p.row_pattern(slot) for slot in range(4)]
        self.assertEqual(len({weights + scales for weights, scales, _ in patterns}), 4)
        for weights, scales, expected in patterns:
            self.assertEqual((len(weights), len(scales), len(expected)), (256, 8, 256))
            self.assertTrue(all(value in p.FP8_BYTES for value in weights))
            self.assertTrue(all(125 <= value <= 129 for value in scales))
            # All reference values are small powers of two, exactly BF16.
            self.assertTrue(all(abs(value) in (0.125, 0.25, 0.5, 1, 2, 4, 8, 16) for value in expected))

    def test_metadata_validates_without_large_allocation(self):
        self.assertEqual(p.validate_metadata(self.document), b"private-test-capability")

    def test_geometry_and_patterns_are_not_peer_configurable(self):
        for key, value in (("rows", 1), ("weight_bytes", 1 << 32), ("dim", 128), ("schema_version", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                p.validate_metadata({**self.document, key: value})
        changed = json.loads(json.dumps(self.document))
        changed["samples"][0]["fp8_bytes"][0] ^= 1
        with self.assertRaisesRegex(ValueError, "PATTERN_MISMATCH"):
            p.validate_metadata(changed)

    def test_allocation_rounding_and_bounds_rejected(self):
        for key, value in (("allocation_bytes", 1 << 32), ("allocation_bytes", 129 << 30),
                           ("granularity", 0), ("granularity", True)):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "ALLOCATION"):
                p.validate_metadata({**self.document, key: value})

    def test_capability_errors_do_not_echo_payload(self):
        for value in ("", "?private-test-capability", "x" * 1025, 42):
            with self.subTest(value=str(value)[:30]):
                with self.assertRaises(ValueError) as raised:
                    p.validate_metadata({**self.document, "handle": value})
                self.assertNotIn("private-test-capability", str(raised.exception))

    def test_private_publication_is_readable_and_not_world_visible(self):
        self.write()
        metadata, handle = p.read_publication(self.path)
        self.assertEqual(metadata["logical_bytes"], p.LOGICAL_BYTES)
        self.assertEqual(handle, b"private-test-capability")
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_nonprivate_directory_and_file_rejected(self):
        self.write()
        self.root.chmod(0o755)
        with self.assertRaisesRegex(ValueError, "UNSAFE_PROBE_DIRECTORY"):
            p.read_publication(self.path)
        self.root.chmod(0o700)
        self.path.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "UNSAFE_BOOTSTRAP_FILE"):
            p.read_publication(self.path)

    def test_uid_and_symlink_rejected(self):
        self.write()
        with mock.patch.object(p.os, "getuid", return_value=self.root.stat().st_uid + 1):
            with self.assertRaisesRegex(ValueError, "UNSAFE_PROBE_DIRECTORY"):
                p.read_publication(self.path)
        alias = self.root / "alias.json"
        alias.symlink_to(self.path)
        with self.assertRaisesRegex(ValueError, "BOOTSTRAP_FILE_UNAVAILABLE"):
            p.read_publication(alias)

    def test_extra_fields_and_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, "INVALID_PROBE_FIELDS"):
            p.validate_metadata({**self.document, "budget": 999999})
        p.private_write(self.path, b'{"handle":"a","handle":"b"}')
        with self.assertRaisesRegex(ValueError, "INVALID_JSON"):
            p.read_publication(self.path)


if __name__ == "__main__":
    unittest.main()
