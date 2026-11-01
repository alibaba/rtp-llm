# SPDX-License-Identifier: Apache-2.0
"""Standalone S1a tests: python3 path/to/test_engram_manifest.py (stdlib only)."""

import copy
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from uuid import UUID

_PATH = Path(__file__).resolve().parents[1] / "engram_manifest.py"
_SPEC = importlib.util.spec_from_file_location("engram_manifest_under_test", _PATH)
m = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = m  # dataclasses resolves postponed annotations here.
_SPEC.loader.exec_module(m)


def uid(n):
    return str(UUID(int=n))


def fixture():
    result = dict(schema_version=1, model_revision="release-immutable-42",
                  generation=uid(1), layout_abi=m.LAYOUT_ABI,
                  hash_abi="sha256:" + "b" * 64, backend="fabric_egm",
                  state="READY", tensors=[])
    # Two layers, each packed into one allocation. Scale offset 96 is NOT
    # aligned to the allocation's 4096-byte VMM granularity, and is valid.
    for layer, rows in ((1, 3), (14, 5)):
        for role, columns in (("weight", 32), ("scale", 1)):
            weight = role == "weight"
            result["tensors"].append(dict(
                name=f"layers.{layer}.engram.embed.{role}", shape=[rows, columns],
                dtype="float8_e4m3fn" if weight else "float8_e8m0fnu",
                storage_dtype="uint8", row_bytes=columns, quant_block=32,
                shards=[dict(shard_id=uid(100 + layer * 2 + (not weight)),
                             row_begin=0, row_end=rows, logical_bytes=rows * columns,
                             byte_offset=0 if weight else rows * 32,
                             allocation_bytes=4096, allocation_granularity=4096,
                             owner_id="owner-a", owner_incarnation=uid(2),
                             allocation_id=uid(10 + layer), handle_ref=f"opaque-{layer}",
                             sha256="sha256:" + ("a" if weight else "c") * 64)]))
    result["dataset_id"] = m.compute_dataset_id(result)
    return result


def sharded_fixture():
    """Two layers, each weight/scale split into two 4096-row shards."""
    result = dict(schema_version=1, model_revision="release-immutable-42",
                  generation=uid(1), layout_abi=m.LAYOUT_ABI,
                  hash_abi="sha256:" + "b" * 64, backend="fabric_egm",
                  state="READY", tensors=[])
    rows, shard_rows = 8192, 4096
    for layer in (1, 14):
        for role, columns in (("weight", 32), ("scale", 1)):
            weight = role == "weight"
            shards = []
            for index in range(2):
                begin = index * shard_rows
                end = min(begin + shard_rows, rows)
                shards.append(dict(
                    shard_id=uid(1000 + layer * 10 + (not weight) * 2 + index),
                    row_begin=begin, row_end=end,
                    logical_bytes=(end - begin) * columns, byte_offset=0,
                    allocation_bytes=shard_rows * columns,
                    allocation_granularity=4096,
                    owner_id=f"shard-owner-{index}", owner_incarnation=uid(2000 + index),
                    allocation_id=uid(3000 + layer * 10 + (not weight) * 2 + index),
                    handle_ref=f"opaque-{layer}-{role}-{index}",
                    sha256="sha256:" + ("a" if weight else "c") * 64))
            result["tensors"].append(dict(
                name=f"layers.{layer}.engram.embed.{role}", shape=[rows, columns],
                dtype="float8_e4m3fn" if weight else "float8_e8m0fnu",
                storage_dtype="uint8", row_bytes=columns, quant_block=32,
                shards=shards))
    result["dataset_id"] = m.compute_dataset_id(result)
    return result


def encode(document):
    return json.dumps(document).encode()


def seal(document):
    document["dataset_id"] = m.compute_dataset_id(document)
    return document


def expected(**changes):
    values = dict(layers=((1, 3), (14, 5)), head_dim=32,
                  model_revision="release-immutable-42", hash_abi="sha256:" + "b" * 64)
    values.update(changes)
    return m.ExpectedLayout(**values)


class ManifestTest(unittest.TestCase):
    def assert_error(self, code, callable_, *args):
        with self.assertRaises(m.ManifestError) as error:
            callable_(*args)
        self.assertEqual(error.exception.code, code)
        return error.exception

    def test_complete_packed_generation_and_deduplicated_capacity(self):
        d = fixture()
        parsed = m.parse_manifest(encode(d))
        verified = m.validate_for_attach(parsed, expected(), d["dataset_id"], d["generation"])
        self.assertEqual(verified.allocation_count, 2)
        self.assertEqual(verified.physical_bytes, 8192)
        self.assertTrue(any(t.shards[0].byte_offset % 4096 for t in parsed.tensors))

    def test_frozen_objects_detach_from_input(self):
        d = fixture()
        parsed = m.parse_manifest(encode(d))
        d["tensors"][0]["shape"][0] = 999
        self.assertEqual(parsed.tensors[1].shape, (3, 32))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            parsed.state = "FAILED"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            parsed.tensors[0].shards[0].byte_offset = 0
        self.assertIsInstance(parsed.tensors, tuple)
        self.assertIsInstance(parsed.tensors[0].shape, tuple)
        layout = expected(layers=[[1, 3], [14, 5]])
        self.assertIsInstance(layout.layers[0], tuple)

    def test_identity_and_publication_digest_are_distinct(self):
        original = fixture()
        changed = copy.deepcopy(original)
        changed["generation"] = uid(99)
        for t in changed["tensors"]:
            s = t["shards"][0]
            s.update(owner_id="owner-b", owner_incarnation=uid(88),
                     allocation_id=uid(500 + int(t["name"].split(".")[1])),
                     handle_ref="new-opaque-reference")
        self.assertEqual(m.compute_dataset_id(changed), original["dataset_id"])
        self.assertNotEqual(m.manifest_digest(m.parse_manifest(encode(original))),
                            m.manifest_digest(m.parse_manifest(encode(changed))))

    def test_order_independent_digest(self):
        d = fixture()
        reordered = dict(reversed(list(d.items())))
        reordered["tensors"] = list(reversed(d["tensors"]))
        a, b = m.parse_manifest(encode(d)), m.parse_manifest(encode(reordered))
        self.assertEqual(a, b)
        self.assertEqual(m.manifest_digest(a), m.manifest_digest(b))

    def test_semantics_and_content_change_dataset_identity(self):
        for field, value in (("model_revision", "release-other"),
                             ("hash_abi", "sha256:" + "e" * 64)):
            d = fixture()
            d[field] = value
            self.assertNotEqual(m.compute_dataset_id(d), fixture()["dataset_id"])
        d = fixture()
        d["tensors"][0]["shards"][0]["sha256"] = "sha256:" + "f" * 64
        self.assertNotEqual(m.compute_dataset_id(d), fixture()["dataset_id"])
        self.assert_error("DATASET_DIGEST_MISMATCH", m.parse_manifest, encode(d))

    def test_missing_entire_layer_requires_trusted_layout(self):
        d = fixture()
        d["tensors"] = d["tensors"][:2]
        seal(d)
        parsed = m.parse_manifest(encode(d))  # Internally complete, globally incomplete.
        self.assert_error("INCOMPLETE_GENERATION", m.validate_for_attach,
                          parsed, expected(), d["dataset_id"], d["generation"])

    def test_expected_identity_generation_hash_shape_and_extra_layers(self):
        d = fixture()
        parsed = m.parse_manifest(encode(d))
        self.assert_error("DATASET_MISMATCH", m.validate_for_attach,
                          parsed, expected(), "sha256:" + "0" * 64, d["generation"])
        self.assert_error("GENERATION_MISMATCH", m.validate_for_attach,
                          parsed, expected(), d["dataset_id"], uid(999))
        self.assert_error("MODEL_CONTRACT_MISMATCH", m.validate_for_attach,
                          parsed, expected(hash_abi="sha256:" + "d" * 64), d["dataset_id"], d["generation"])
        for contract in (expected(layers=((1, 3),)), expected(head_dim=64),
                         expected(layers=((1, 3), (14, 6)))):
            self.assert_error("INCOMPLETE_GENERATION", m.validate_for_attach,
                              parsed, contract, d["dataset_id"], d["generation"])

    def test_non_ready_publication_cannot_attach(self):
        for state in ("LOADING", "VALIDATING", "DRAINING", "RETIRED", "FAILED"):
            d = fixture()
            d["state"] = state
            self.assert_error("NOT_READY", m.validate_for_attach,
                              m.parse_manifest(encode(d)), expected(), d["dataset_id"], d["generation"])

    def test_backend_is_an_explicit_attach_requirement(self):
        d = fixture()
        d["backend"] = "local_shm"
        # Storage transport does not change the semantic dataset identity.
        parsed = m.parse_manifest(encode(d))
        self.assert_error("BACKEND_MISMATCH", m.validate_for_attach,
                          parsed, expected(), d["dataset_id"], d["generation"])
        accepted = m.validate_for_attach(parsed, expected(backend="local_shm"),
                                         d["dataset_id"], d["generation"])
        self.assertEqual(accepted.allocation_count, 2)

    def test_missing_pair_and_duplicate_tensor(self):
        d = fixture()
        del d["tensors"][1]
        self.assert_error("INCOMPLETE_WEIGHT_SCALE", m.parse_manifest, encode(d))
        d = fixture()
        d["tensors"].append(copy.deepcopy(d["tensors"][0]))
        self.assert_error("DUPLICATE_TENSOR", m.parse_manifest, encode(d))

    def test_json_duplicates_invalid_constants_and_depth(self):
        self.assert_error("DUPLICATE_KEY", m.parse_manifest, b'{"x":1,"x":2}')
        for data in (b'{"x":NaN}', b'{"x":Infinity}', b'\xff', b'{} garbage',
                     b'[' * 2000 + b']' * 2000):
            self.assert_error("INVALID_JSON", m.parse_manifest, data)
        data = encode(fixture()).replace(b'"schema_version": 1', b'"schema_version": ' + b'9' * 5000)
        with self.assertRaises(m.ManifestError) as error:
            m.parse_manifest(data)
        # Python 3.11+ may reject at the JSON integer conversion limit; older
        # versions reach our signed-64-bit bound. Both must be structured.
        self.assertIn(error.exception.code, ("INVALID_JSON", "INVALID_INTEGER"))

    def test_numeric_types_overflow_and_negative_ranges(self):
        for value in (True, 3.0, "3", 0, -1, 1 << 63):
            d = fixture()
            d["tensors"][0]["shape"][0] = value
            self.assert_error("INVALID_INTEGER", m.parse_manifest, encode(d))
        d = fixture()
        d["tensors"][0]["shape"][0] = (1 << 63) - 1
        self.assert_error("SIZE_OVERFLOW", m.parse_manifest, encode(d))
        d = fixture()
        d["tensors"][0]["shards"][0]["byte_offset"] = -1
        self.assert_error("INVALID_INTEGER", m.parse_manifest, encode(d))

    def test_dtype_quantization_shape_and_stride(self):
        for field, value, code in (
            ("dtype", "float32", "UNSUPPORTED_DTYPE"),
            ("storage_dtype", "float8_e4m3fn", "UNSUPPORTED_DTYPE"),
            ("quant_block", 64, "UNSUPPORTED_QUANTIZATION"),
            ("row_bytes", 64, "INVALID_STRIDE"),
            ("shape", [3, 31], "INVALID_SHAPE"),
        ):
            d = fixture()
            d["tensors"][0][field] = value
            self.assert_error(code, m.parse_manifest, encode(d))
        d = fixture()
        d["tensors"][1].update(shape=[4, 1])
        d["tensors"][1]["shards"][0].update(row_end=4, logical_bytes=4)
        self.assert_error("INCOMPLETE_WEIGHT_SCALE", m.parse_manifest, encode(d))

    def test_range_bounds_granularity_and_overlap(self):
        for change, code in (
            ({"row_begin": 1}, "INCOMPLETE_RANGE"),
            ({"row_end": 4}, "INCOMPLETE_RANGE"),
            ({"logical_bytes": 95}, "INCOMPLETE_RANGE"),
            ({"byte_offset": 4095}, "OUT_OF_BOUNDS"),
            ({"byte_offset": (1 << 63) - 1}, "OUT_OF_BOUNDS"),
            ({"allocation_bytes": 4095}, "INVALID_ALLOCATION_SIZE"),
            ({"allocation_granularity": 0}, "INVALID_INTEGER"),
        ):
            d = fixture()
            d["tensors"][0]["shards"][0].update(change)
            self.assert_error(code, m.parse_manifest, encode(d))
        d = fixture()
        d["tensors"][1]["shards"][0]["byte_offset"] = 95
        self.assert_error("OVERLAPPING_TENSORS", m.parse_manifest, encode(d))

    def test_allocation_conflicts_and_multiple_owners(self):
        for key, value in (("allocation_bytes", 8192), ("allocation_granularity", 2048),
                           ("handle_ref", "different")):
            d = fixture()
            d["tensors"][1]["shards"][0][key] = value
            self.assert_error("ALLOCATION_CONFLICT", m.parse_manifest, encode(d))
        for key, value in (("owner_id", "second-owner"), ("owner_incarnation", uid(99))):
            d = fixture()
            for t in d["tensors"][2:]:
                t["shards"][0][key] = value
            self.assert_error("MULTIPLE_OWNERS", m.parse_manifest, encode(d))


    def test_two_shard_generation_and_validation(self):
        d = sharded_fixture()
        parsed = m.parse_manifest(encode(d))
        verified = m.validate_for_attach(
            parsed, expected(layers=((1, 8192), (14, 8192))), d["dataset_id"], d["generation"])
        self.assertEqual(verified.allocation_count, 8)
        self.assertEqual(verified.physical_bytes, 540672)
        for t in parsed.tensors:
            self.assertEqual(len(t.shards), 2)
            self.assertEqual(t.shards[0].row_begin, 0)
            self.assertEqual(t.shards[1].row_end, t.shape[0])

        d = sharded_fixture()
        d["tensors"][0]["shards"][1].update(row_begin=4097, logical_bytes=(8192 - 4097) * 32)
        self.assert_error("INCOMPLETE_RANGE", m.parse_manifest, encode(d))
        d = sharded_fixture()
        d["tensors"][0]["shards"][1]["allocation_bytes"] = 8192 * 32
        self.assert_error("UNEVEN_SHARDS", m.parse_manifest, encode(d))
        d = sharded_fixture()
        d["tensors"][0]["shards"][1]["byte_offset"] = 4096
        self.assert_error("INVALID_ALLOCATION_SIZE", m.parse_manifest, encode(d))

    def test_sharding_duplicate_shard_and_unknown_fields(self):
        d = fixture()
        d["tensors"][0]["shards"] *= 2
        self.assert_error("DUPLICATE_SHARD", m.parse_manifest, encode(d))
        d = fixture()
        d["tensors"][1]["shards"][0]["shard_id"] = d["tensors"][0]["shards"][0]["shard_id"]
        self.assert_error("DUPLICATE_SHARD", m.parse_manifest, encode(d))
        d = fixture()
        d["fabric_handle"] = "SECRET-HANDLE-BYTES"
        error = self.assert_error("INVALID_FIELDS", m.parse_manifest, encode(d))
        self.assertNotIn("SECRET", str(error))

    def test_version_abi_ids_and_no_sensitive_values_in_errors(self):
        for key, value, code in (("schema_version", 2, "UNSUPPORTED_SCHEMA"),
                                 ("schema_version", True, "INVALID_INTEGER"),
                                 ("layout_abi", "other", "UNSUPPORTED_SCHEMA"),
                                 ("backend", "other", "UNSUPPORTED_BACKEND"),
                                 ("generation", "SECRET-HANDLE-BYTES", "INVALID_UUID"),
                                 ("hash_abi", "invalid", "INVALID_DIGEST")):
            d = fixture()
            d[key] = value
            error = self.assert_error(code, m.parse_manifest, encode(d))
            self.assertNotIn("SECRET", str(error))
        d = fixture()
        d["tensors"][0]["shards"][0]["handle_ref"] = "\ud800"
        self.assert_error("INVALID_TEXT", m.parse_manifest, encode(d))
        self.assertNotIn("opaque-", repr(m.parse_manifest(encode(fixture()))))

    def test_limits_include_unique_allocation_sum_and_revalidation(self):
        d = fixture()
        data = encode(d)
        self.assert_error("INPUT_LIMIT", m.parse_manifest, data, m.Limits(max_manifest_bytes=10))
        self.assert_error("TENSOR_LIMIT", m.parse_manifest, data, m.Limits(max_tensors=3))
        self.assert_error("INVALID_ALLOCATION_SIZE", m.parse_manifest, data,
                          m.Limits(max_allocation_bytes=2048))
        self.assert_error("TOTAL_ALLOCATION_LIMIT", m.parse_manifest, data,
                          m.Limits(max_total_allocation_bytes=8191))
        parsed = m.parse_manifest(data, m.Limits(max_total_allocation_bytes=8192))
        self.assert_error("TOTAL_ALLOCATION_LIMIT", m.validate_for_attach,
                          parsed, expected(), d["dataset_id"], d["generation"],
                          m.Limits(max_total_allocation_bytes=8191))
        self.assert_error("INPUT_LIMIT", m.validate_for_attach,
                          parsed, expected(), d["dataset_id"], d["generation"],
                          m.Limits(max_manifest_bytes=10))
        forged = dataclasses.replace(parsed, state="FAILED")
        self.assert_error("NOT_READY", m.validate_for_attach,
                          forged, expected(), d["dataset_id"], d["generation"])
        malformed = dataclasses.replace(parsed, tensors=7)
        self.assert_error("TENSOR_LIMIT", m.validate_for_attach,
                          malformed, expected(), d["dataset_id"], d["generation"])
        unserializable = dataclasses.replace(parsed, model_revision=object())
        self.assert_error("INVALID_FIELDS", m.validate_for_attach,
                          unserializable, expected(), d["dataset_id"], d["generation"])

    def test_expected_layout_is_validated(self):
        for change, code in (({"layers": ((1, 3), (1, 5))}, "DUPLICATE_LAYER"),
                             ({"head_dim": 31}, "UNSUPPORTED_LAYOUT"),
                             ({"backend": "unknown"}, "UNSUPPORTED_BACKEND"),
                             ({"layers": ((True, 3),)}, "INVALID_INTEGER"),
                             ({"layers": ()}, "INVALID_LAYOUT")):
            with self.assertRaises(m.ManifestError) as error:
                expected(**change)
            self.assertEqual(error.exception.code, code)
        self.assert_error("INVALID_LIMIT", m.Limits, True)


if __name__ == "__main__":
    unittest.main()
