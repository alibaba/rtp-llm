"""CPU-only contracts for the real-checkpoint lookup probe; no fake GPU pass."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest


spec = importlib.util.spec_from_file_location("checkpoint_lookup_under_test", Path(__file__).with_name("checkpoint_lookup.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class OracleTest(unittest.TestCase):
    def setUp(self):
        self.layout = SimpleNamespace(layer_ids=(1, 14), num_embeddings=(100, 120), head_dim=32,
                                      n_hash_cols=2, max_ngram_size=4)
        self.metadata = {"config_sha256": "sha256:" + "a" * 64, "tokenizer_sha256": "sha256:" + "b" * 64}
        self.identity = {"dataset_id": "sha256:" + "c" * 64,
                         "generation": "d0cb151e-f230-4e3f-879f-5cf934cb692c", "model_revision": "actual-source-revision",
                         "hash_abi": "sha256:" + "d" * 64, "manifest_sha256": "sha256:" + "e" * 64}
        cases = [{"name": "boundary", "kind": "direct", "token_windows": None,
                  "ids": {"1": [[0, 50, 99, -1, 0]], "14": [[0, 60, 119, -1, 0]]}},
                 {"name": "real_tokenizer", "kind": "hash", "token_windows": [[5, -1, -1, -1]],
                  "ids": {"1": [[1, 2]], "14": [[3, 4]]}}]
        rows = {layer: [{"row_id": row, "fp8_hex": (b"\x38" * 32).hex(), "e8m0_hex": "7f"}
                        for row in sorted(values)] for layer, values in probe.selected_ids(cases, self.layout).items()}
        self.document = {"schema_version": 1, "kind": probe.KIND, "identity": self.identity,
                         "checkpoint_metadata": self.metadata, "layout": probe.layout_metadata(self.layout),
                         "cases": cases, "rows": rows,
                         "source_headers": {"model-00047-of-00048.safetensors": "sha256:" + "f" * 64}}

    def validate(self, document=None):
        return probe.validate_oracle(document or self.document, layout=self.layout, metadata=self.metadata,
                                     identity=self.identity, token_count=1000)

    def test_valid_document_contains_exact_requested_rows_only(self):
        parsed = self.validate()
        self.assertEqual(set(parsed["1"]), {0, 1, 2, 50, 99})
        self.assertEqual(set(parsed["14"]), {0, 3, 4, 60, 119})
        self.assertEqual(probe.decode_row(*parsed["1"][50]), [1.0] * 32)

    def test_fp8_decoder_covers_normal_subnormal_signed_zero_and_maximum(self):
        expected = {0x38: 1.0, 0xB8: -1.0, 0x40: 2.0, 0x30: 0.5,
                    0x01: 2 ** -9, 0x07: 7 * 2 ** -9, 0x08: 2 ** -6,
                    0x7E: 448.0, 0xFE: -448.0}
        for byte, value in expected.items():
            self.assertEqual(probe.fp8_e4m3_value(byte), value)
        self.assertEqual(math.copysign(1, probe.fp8_e4m3_value(0x80)), -1)
        for byte in (0x7F, 0xFF, -1, 256, True):
            with self.subTest(byte=byte), self.assertRaises(ValueError):
                probe.fp8_e4m3_value(byte)

    def test_e8m0_uses_ieee_exponent_bits_including_zero(self):
        actual = probe.decode_row(b"\x38" * 96, bytes([0, 127, 128]))
        self.assertEqual(actual, [0.0] * 32 + [1.0] * 32 + [2.0] * 32)
        with self.assertRaisesRegex(ValueError, "non-finite E8M0"):
            probe.decode_row(b"\x38" * 32, b"\xff")
        with self.assertRaisesRegex(ValueError, "geometry"):
            probe.decode_row(b"\x38" * 31, b"\x7f")

    def test_boundaries_include_4gib_and_tail_without_int32_truncation(self):
        rows, dim = 400_000_000, 256
        boundary = 2 ** 32 // dim
        ids = probe.boundary_ids(rows, dim)
        self.assertIn(rows - 1, ids)
        for row in (boundary - 1, boundary, boundary + 1):
            self.assertIn(row, ids)
        self.assertGreater((rows - 1) * dim, 2 ** 32)
        self.assertEqual(ids[-2:], [-1, 0])

    def test_token_windows_preserve_real_history_boundaries(self):
        windows = probe.windows_for_tokens([5, 6, 7, 8, 9], 4)
        self.assertEqual(windows[0], [5, -1, -1, -1])
        self.assertEqual(windows[3], [8, 7, 6, 5])
        self.assertEqual(windows[-1], [9, 8, 7, 6])
        for tokens in ([], [-1], [True], list(range(17))):
            with self.assertRaises(ValueError):
                probe.windows_for_tokens(tokens, 4)

    def test_rejects_generation_layout_and_config_drift(self):
        changes = [("identity", "generation", "7e25436e-7cdf-45f7-8520-d2f9475351e2"),
                   ("checkpoint_metadata", "config_sha256", "sha256:" + "0" * 64),
                   ("layout", "head_dim", 64)]
        for section, key, value in changes:
            document = copy.deepcopy(self.document);document[section][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "mismatch"):
                self.validate(document)
        document = copy.deepcopy(self.document);document["layout"]["layers"][0]["layer_id"] = True
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.validate(document)

    def test_rejects_duplicate_missing_extra_and_malformed_raw_rows(self):
        for mutation in ("duplicate", "missing", "extra", "short", "nan"):
            document = copy.deepcopy(self.document)
            rows = document["rows"]["1"]
            if mutation == "duplicate": rows.append(copy.deepcopy(rows[0]))
            if mutation == "missing": rows.pop()
            if mutation == "extra": rows.append(dict(rows[0], row_id=98))
            if mutation == "short": rows[0]["fp8_hex"] = "38"
            if mutation == "nan": rows[0]["fp8_hex"] = "7f" * 32
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate(document)

    def test_rejects_out_of_bounds_boolean_and_ragged_lookup_indices(self):
        for value in (-2, 100, True):
            document = copy.deepcopy(self.document)
            document["cases"][0]["ids"]["1"][0][0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "bounds"):
                self.validate(document)
        document = copy.deepcopy(self.document);document["cases"][0]["ids"]["1"].append([0])
        with self.assertRaisesRegex(ValueError, "ragged"):
            self.validate(document)

    def test_rejects_tokenizer_oob_before_hash_kernel(self):
        document = copy.deepcopy(self.document)
        document["cases"][1]["token_windows"][0][0] = 1000
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            self.validate(document)

    def test_bundle_samples_bind_raw_rows_to_published_owner(self):
        parsed = self.validate();bundle = {"samples": {}}
        for layer, count in zip(self.layout.layer_ids, self.layout.num_embeddings):
            selected = sorted({0, count // 2, count - 1})
            for slot, role in enumerate(("weight", "scale")):
                name = "layers.{}.engram.embed.{}".format(layer, role)
                bundle["samples"][name] = {"row_ids": selected, "sha256": probe.digest(b"".join(parsed[str(layer)][row][slot] for row in selected))}
        probe.verify_bundle_samples(parsed, bundle, self.layout)
        bundle["samples"]["layers.1.engram.embed.weight"]["sha256"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ValueError, "differs"):
            probe.verify_bundle_samples(parsed, bundle, self.layout)

    def test_bundle_identity_validates_digest_and_uuid(self):
        self.assertEqual(probe.bundle_identity(self.identity), self.identity)
        for name, value in (("generation", "not-a-uuid"), ("dataset_id", "wrong"), ("model_revision", "")):
            document = dict(self.identity, **{name: value})
            with self.subTest(name=name), self.assertRaises(ValueError):
                probe.bundle_identity(document)


class PrivateAndSourceTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.root.chmod(0o700)

    def test_private_roundtrip_refuses_overwrite_and_cleans_temporary(self):
        target = self.root / "oracle.json"
        probe.private_write_new(target, {"safe": 1})
        self.assertEqual(probe.private_read(target), {"safe": 1})
        self.assertEqual(target.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            probe.private_write_new(target, {"unsafe": 2})
        self.assertEqual(probe.private_read(target), {"safe": 1})
        self.assertFalse(list(self.root.glob("*.tmp-*")))

    def test_private_read_refuses_symlink_public_mode_and_oversize(self):
        target = self.root / "oracle.json"
        probe.private_write_new(target, {"safe": 1})
        link = self.root / "link.json";link.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            probe.private_read(link)
        target.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            probe.private_read(target)
        target.chmod(0o600)
        with target.open("wb") as stream: stream.truncate(probe.MAX_PRIVATE_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "oversized"):
            probe.private_read(target)

    def test_duplicate_keys_and_nonfinite_json_fail(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(ValueError):
                probe.unique_json(raw)

    def test_selected_source_rows_preserve_bytes_without_full_table(self):
        filename = "model.safetensors"
        layout = SimpleNamespace(layer_ids=(1,), num_embeddings=(4,), head_dim=32)
        weight = b"".join(bytes([0x38 + i]) * 32 for i in range(4))
        scale = bytes([125, 126, 127, 128])
        header = {"layers.1.engram.embed.weight": {"dtype": "F8_E4M3", "shape": [4, 32], "data_offsets": [0, 128]},
                  "layers.1.engram.embed.scale": {"dtype": "F8_E8M0", "shape": [4, 1], "data_offsets": [128, 132]}}
        raw = json.dumps(header).encode();raw += b" " * (-len(raw) % 8)
        (self.root / filename).write_bytes(struct.pack("<Q", len(raw)) + raw + weight + scale)
        index = {key: filename for key in header}
        rows, headers = probe.read_selected_rows(self.root, index, layout, {"1": {0, 3}})
        self.assertEqual([x["row_id"] for x in rows["1"]], [0, 3])
        self.assertEqual(bytes.fromhex(rows["1"][1]["fp8_hex"]), b"\x3b" * 32)
        self.assertEqual(rows["1"][1]["e8m0_hex"], "80")
        self.assertEqual(set(headers), {filename})
        with self.assertRaisesRegex(ValueError, "selection"):
            probe.read_selected_rows(self.root, index, layout, {"1": {4}})


if __name__ == "__main__":
    unittest.main()
