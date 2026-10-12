"""L20 protocol tests with real packed cache storage and independently mapped pages."""

import dataclasses
import json
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models.test.dsv41_cache_codec_test import (
    codec,
    load_codec_source,
    make_cache,
)

transfer = load_codec_source("_dsv41_transfer_test_source", "v41_cache_transfer.py")
Identity = transfer.V41CacheIdentity
Span = transfer.V41L20Span
Contract = transfer.V41L20Contract
Receiver = transfer.V41L20ReplicaReceiver
TAGS = ("hca_kv", "indexer_kv")
ROWS = ("v41_l20_global_delta", "v41_l20_index_delta")


def caches():
    return {tag: make_cache(tag, pages=6, gap=64)[0] for tag in TAGS}


def slots(blocks, start, count):
    return codec.logical_to_physical_slots(
        torch.tensor(blocks), torch.arange(start, start + count), "hca_kv", 1
    )


class V41L20TransferTest(unittest.TestCase):
    def setUp(self):
        self.contract = Contract("a" * 64, "b" * 64, "c" * 64, 4096)
        self.source = caches()
        self.target = caches()
        self.identity = Identity(101, 1, 12)
        self.producer_blocks = {"hca_kv": [2, 0], "indexer_kv": [3, 1]}
        self.replica_blocks = {"hca_kv": [1, 3], "indexer_kv": [4, 0]}
        self.receiver = Receiver(self.contract, 2, self.target, backend="eager")
        self.receiver.begin_request(self.identity, self.replica_blocks)

    def seed(self, blocks, start, count, marker=0):
        result = {}
        for i, tag in enumerate(TAGS):
            width = 288 if tag == "hca_kv" else 68
            rows = (
                (torch.arange(count * width) + marker + i * 31)
                .remainder(256)
                .to(torch.uint8)
                .view(count, width)
            )
            self.source[tag].scatter(
                rows, slots(blocks[tag], start, count), backend="eager"
            )
            result[tag] = rows
        return result

    def export(self, spans, round_id=0, tables=None, contract=None):
        return transfer.export_l20_delta(
            contract or self.contract,
            2,
            round_id,
            spans,
            self.source,
            tables or {self.identity: self.producer_blocks},
            backend="eager",
        )

    def snapshots(self):
        return {tag: c.raw.clone() for tag, c in self.target.items()}

    def assert_unchanged(self, snapshots):
        for tag in TAGS:
            torch.testing.assert_close(
                self.target[tag].raw, snapshots[tag], rtol=0, atol=0
            )

    def assert_rows(self, blocks, start, count, expected):
        for tag in TAGS:
            actual = self.target[tag].gather(
                slots(blocks[tag], start, count), backend="eager"
            )
            torch.testing.assert_close(actual, expected[tag], rtol=0, atol=0)

    def test_canonical_contract_covers_identity_and_layout(self):
        restored = Contract(**dataclasses.asdict(self.contract))
        self.assertEqual(restored.canonical_bytes(), self.contract.canonical_bytes())
        self.assertEqual(restored.fingerprint, self.contract.fingerprint)
        manifest = json.loads(self.contract.canonical_bytes())
        self.assertEqual(manifest["source_layer"], 20)
        self.assertEqual(manifest["cp_size"], 4)
        self.assertFalse(manifest["cp_cache_sharded"])
        self.assertEqual(
            [
                (r["tag"], r["entries"], r["scale_offset"], r["stride"])
                for r in manifest["regions"]
            ],
            [("hca_kv", 256, 65536, 73728), ("indexer_kv", 256, 16384, 17408)],
        )
        for field, value in (
            ("checkpoint_sha256", "d" * 64),
            ("config_sha256", "e" * 64),
            ("tokenizer_sha256", "f" * 64),
            ("max_sequence_length", 2048),
        ):
            self.assertNotEqual(
                dataclasses.replace(self.contract, **{field: value}).fingerprint,
                self.contract.fingerprint,
            )

    def test_rotary_origin_binds_contract_and_rejects_before_writes(self):
        legacy = self.contract.canonical_bytes()
        self.assertNotIn("rotary_compute_device", json.loads(legacy))
        contracts = [
            dataclasses.replace(self.contract, rotary_compute_device=value)
            for value in (None, "cpu", "cuda")
        ]
        self.assertEqual(contracts[0].canonical_bytes(), legacy)
        self.assertEqual(len({c.fingerprint for c in contracts}), 3)
        for contract in contracts:
            contract.require_rotary_config(
                {"rotary_compute_device": contract.rotary_compute_device}
            )
            for other in contracts:
                if other == contract:
                    continue
                with self.assertRaisesRegex(ValueError, "rotary origin"):
                    contract.require_rotary_config(
                        {"rotary_compute_device": other.rotary_compute_device}
                    )
        span = Span(self.identity, 0, 0, 3)
        self.seed(self.producer_blocks, 0, 3)
        before = self.snapshots()
        for contract in contracts[1:]:
            wire = self.export([span], contract=contract)
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                self.receiver.apply(wire, [span])
            self.assert_unchanged(before)
        self.receiver.apply(self.export([span]), [span])
        for value in (False, "cuda:0", "meta", []):
            with self.assertRaisesRegex(ValueError, "rotary_compute_device"):
                dataclasses.replace(self.contract, rotary_compute_device=value)

    def test_mhc_collapse_binds_contract_and_rejects_before_writes(self):
        legacy = self.contract.canonical_bytes()
        self.assertNotIn("mhc_collapse_mode", json.loads(legacy))
        omitted = dataclasses.replace(self.contract, mhc_collapse_mode=None)
        selected = dataclasses.replace(self.contract, mhc_collapse_mode="ordered_fp32")
        self.assertEqual(omitted.canonical_bytes(), legacy)
        self.assertNotEqual(selected.fingerprint, self.contract.fingerprint)
        selected.require_attention_config({"mhc_collapse_mode": "ordered_fp32"})
        for contract, config in (
            (selected, {}),
            (omitted, {"mhc_collapse_mode": "ordered_fp32"}),
        ):
            with self.assertRaisesRegex(ValueError, "mHC collapse mode"):
                contract.require_attention_config(config)
        span = Span(self.identity, 0, 0, 3)
        self.seed(self.producer_blocks, 0, 3)
        before = self.snapshots()
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            self.receiver.apply(self.export([span], contract=selected), [span])
        self.assert_unchanged(before)
        self.receiver.apply(self.export([span]), [span])
        for value in (False, True, 0, "", "native", "fp32", [], {}):
            with self.assertRaisesRegex(ValueError, "mhc_collapse_mode"):
                dataclasses.replace(self.contract, mhc_collapse_mode=value)

    def test_mhc_post_binds_contract_and_rejects_before_writes(self):
        legacy = self.contract.canonical_bytes()
        self.assertNotIn("mhc_post_mode", json.loads(legacy))
        omitted = dataclasses.replace(self.contract, mhc_post_mode=None)
        selected = dataclasses.replace(self.contract, mhc_post_mode="ordered_fp32")
        self.assertEqual(omitted.canonical_bytes(), legacy)
        self.assertNotEqual(selected.fingerprint, self.contract.fingerprint)
        selected.require_attention_config({"mhc_post_mode": "ordered_fp32"})
        for contract, config in (
            (selected, {}),
            (omitted, {"mhc_post_mode": "ordered_fp32"}),
        ):
            with self.assertRaisesRegex(ValueError, "mHC post mode"):
                contract.require_attention_config(config)
        span = Span(self.identity, 0, 0, 3)
        self.seed(self.producer_blocks, 0, 3)
        before = self.snapshots()
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            self.receiver.apply(self.export([span], contract=selected), [span])
        self.assert_unchanged(before)
        self.receiver.apply(self.export([span]), [span])
        for value in (False, True, 0, "", "native", "fp32", [], {}):
            with self.assertRaisesRegex(ValueError, "mhc_post_mode"):
                dataclasses.replace(self.contract, mhc_post_mode=value)

    def test_relocated_two_chunk_roundtrip_and_exact_readiness(self):
        first = Span(self.identity, 0, 0, 257)
        rows = self.seed(self.producer_blocks, 0, first.count)
        message = self.export([first])
        self.assertEqual(
            set(message),
            {"v41_delta_header", "v41_cache_fingerprint", "v41_delta_spans", *ROWS},
        )
        self.assertEqual(message["v41_delta_header"].tolist(), [1, 20, 2, 0, 1, 257])
        self.assertEqual(message["v41_delta_spans"].tolist(), [[101, 1, 12, 0, 0, 257]])
        self.receiver.apply(message, [first])
        self.receiver.require_ready(self.identity, 257, 0)
        self.assert_rows(self.replica_blocks, 0, 257, rows)
        second = Span(self.identity, 1, 257, 73)
        next_rows = self.seed(self.producer_blocks, 257, 73, marker=51)
        self.receiver.apply(self.export([second], 1), [second])
        self.receiver.require_ready(self.identity, 330, 1)
        self.assert_rows(self.replica_blocks, 257, 73, next_rows)
        self.assert_rows(self.replica_blocks, 0, 257, rows)
        for count, chunk in ((329, 1), (331, 1), (330, 0), (330, 2)):
            with self.assertRaises(ValueError):
                self.receiver.require_ready(self.identity, count, chunk)

    def test_export_rejects_shared_pages_even_for_disjoint_delta_slots(self):
        other = Identity(102, 2, 9)
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.export(
                [Span(self.identity, 0, 0, 1), Span(other, 1, 16, 1)],
                tables={
                    self.identity: self.producer_blocks,
                    other: self.producer_blocks,
                },
            )

    def test_multiple_requests_can_arrive_in_dispatch_order(self):
        other = Identity(102, 2, 9)
        other_source = {"hca_kv": [4], "indexer_kv": [2]}
        other_target = {"hca_kv": [5], "indexer_kv": [2]}
        self.receiver.begin_request(other, other_target)
        a = Span(self.identity, 0, 0, 19)
        b = Span(other, 0, 0, 7)
        a_rows = self.seed(self.producer_blocks, 0, 19, 11)
        b_rows = self.seed(other_source, 0, 7, 23)
        message = self.export(
            [b, a], tables={self.identity: self.producer_blocks, other: other_source}
        )
        message["hidden_states"] = torch.ones(1)
        self.receiver.apply(message, [b, a])
        self.assert_rows(self.replica_blocks, 0, 19, a_rows)
        self.assert_rows(other_target, 0, 7, b_rows)
        self.receiver.require_ready(other, 7, 0)

    def test_bad_envelope_and_fingerprint_never_write(self):
        span = Span(self.identity, 0, 0, 3)
        message = self.export([span])
        before = self.snapshots()
        for column in range(6):
            bad = dict(message)
            bad["v41_delta_header"] = message["v41_delta_header"].clone()
            bad["v41_delta_header"][column] += 1
            with self.subTest(column=column), self.assertRaisesRegex(
                ValueError, "envelope"
            ):
                self.receiver.apply(bad, [span])
            self.assert_unchanged(before)
        for field in ("checkpoint_sha256", "config_sha256", "tokenizer_sha256"):
            bad = self.export(
                [span], contract=dataclasses.replace(self.contract, **{field: "f" * 64})
            )
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                self.receiver.apply(bad, [span])
            self.assert_unchanged(before)
        self.receiver.apply(message, [span])

    def test_missing_or_malformed_payload_is_rejected_before_either_write(self):
        span = Span(self.identity, 0, 0, 3)
        message = self.export([span])
        before = self.snapshots()
        bad_values = [
            ("v41_delta_header", torch.zeros(6, dtype=torch.int32)),
            ("v41_cache_fingerprint", torch.zeros(31, dtype=torch.uint8)),
            ("v41_delta_spans", torch.zeros(1, 5, dtype=torch.int64)),
            (ROWS[0], torch.zeros(3, 288)),
            (ROWS[1], torch.zeros(2, 68, dtype=torch.uint8)),
            (ROWS[1], torch.zeros(68, 3, dtype=torch.uint8).T),
            (ROWS[1], self.target["hca_kv"].raw[0, :204].view(3, 68)),
        ]
        for key in message:
            bad = {k: v for k, v in message.items() if k != key}
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.receiver.apply(bad, [span])
            self.assert_unchanged(before)
        for key, value in bad_values:
            with self.subTest(key=key, shape=value.shape), self.assertRaises(
                ValueError
            ):
                self.receiver.apply({**message, key: value}, [span])
            self.assert_unchanged(before)
        with self.assertRaises(ValueError):
            self.receiver.apply({"hidden_states": torch.ones(1)}, [span])
        self.receiver.apply(message, [span])

    def test_spans_must_match_trusted_dispatch_exactly(self):
        span = Span(self.identity, 0, 0, 3)
        message = self.export([span])
        before = self.snapshots()
        for column in range(6):
            bad = dict(message)
            bad["v41_delta_spans"] = message["v41_delta_spans"].clone()
            bad["v41_delta_spans"][0, column] += 1
            with self.assertRaisesRegex(ValueError, "dispatch"):
                self.receiver.apply(bad, [span])
            self.assert_unchanged(before)

    def test_gaps_duplicates_and_initial_prefix_reuse_are_rejected(self):
        for chunk, start in ((1, 0), (0, 1), (9, 256)):
            span = Span(self.identity, chunk, start, 3)
            before = self.snapshots()
            with self.assertRaisesRegex(ValueError, "missing"):
                self.receiver.apply(self.export([span]), [span])
            self.assert_unchanged(before)
        span = Span(self.identity, 0, 0, 3)
        message = self.export([span])
        self.receiver.apply(message, [span])
        before = self.snapshots()
        with self.assertRaisesRegex(ValueError, "round"):
            self.receiver.apply(message, [span])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.receiver.apply(self.export([span], 1), [span])
        self.assert_unchanged(before)

    def test_wrong_identity_cannot_target_an_existing_allocation(self):
        before = self.snapshots()
        for identity in (
            Identity(999, 1, 12),
            Identity(101, 2, 12),
            Identity(101, 1, 13),
        ):
            span = Span(identity, 0, 0, 3)
            message = self.export([span], tables={identity: self.producer_blocks})
            with self.assertRaisesRegex(ValueError, "identity"):
                self.receiver.apply(message, [span])
            self.assert_unchanged(before)

    def test_cancelled_request_rejects_late_data_and_holds_pages_until_release(self):
        span = Span(self.identity, 0, 0, 3)
        message = self.export([span])
        self.receiver.cancel(self.identity)
        before = self.snapshots()
        with self.assertRaisesRegex(ValueError, "cancelled"):
            self.receiver.apply(message, [span])
        with self.assertRaises(ValueError):
            self.receiver.require_ready(self.identity, 3, 0)
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.receiver.begin_request(Identity(102, 2, 12), self.replica_blocks)
        self.assert_unchanged(before)
        self.receiver.release(self.identity)
        replacement = Identity(101, 2, 13)
        self.receiver.begin_request(replacement, self.replica_blocks)
        with self.assertRaisesRegex(ValueError, "identity"):
            self.receiver.apply(message, [span])
        with self.assertRaises(ValueError):
            self.receiver.require_ready(replacement, 3, 0)
        fresh = Span(replacement, 0, 0, 3)
        self.receiver.apply(
            self.export([fresh], tables={replacement: self.producer_blocks}), [fresh]
        )
        self.receiver.require_ready(replacement, 3, 0)

    def test_retired_generation_cannot_be_admitted_again(self):
        self.receiver.release(self.identity)
        for identity in (self.identity, Identity(201, 0, 99)):
            with self.assertRaisesRegex(ValueError, "generation"):
                self.receiver.begin_request(identity, self.replica_blocks)
        self.receiver.begin_request(Identity(101, 2, 13), self.replica_blocks)

    def test_empty_round_has_no_empty_tensors_and_advances_order(self):
        message = self.export([])
        self.assertEqual(set(message), {"v41_delta_header", "v41_cache_fingerprint"})
        self.assertTrue(all(t.numel() for t in message.values()))
        before = self.snapshots()
        with self.assertRaises(ValueError):
            self.receiver.apply(
                {**message, ROWS[0]: torch.empty(0, 288, dtype=torch.uint8)}, []
            )
        self.receiver.apply(message, [])
        self.assert_unchanged(before)
        span = Span(self.identity, 0, 0, 2)
        self.receiver.apply(self.export([span], 1), [span])
        self.receiver.require_ready(self.identity, 2, 0)

    def test_append_only_allocation_growth_preserves_imported_prefix(self):
        receiver = Receiver(self.contract, 2, self.target, backend="eager")
        prefix_blocks = {tag: pages[:1] for tag, pages in self.replica_blocks.items()}
        receiver.begin_request(self.identity, prefix_blocks)
        first = Span(self.identity, 0, 0, 250)
        first_rows = self.seed(self.producer_blocks, 0, 250)
        receiver.apply(self.export([first]), [first])
        second = Span(self.identity, 1, 250, 15)
        second_rows = self.seed(self.producer_blocks, 250, 15, 77)
        message = self.export([second], 1)
        before = self.snapshots()
        with self.assertRaisesRegex(ValueError, "allocation"):
            receiver.apply(message, [second])
        self.assert_unchanged(before)
        for bad in (
            {"hca_kv": [0, 3], "indexer_kv": [4, 0]},
            {tag: [] for tag in TAGS},
        ):
            with self.assertRaisesRegex(ValueError, "appended"):
                receiver.extend_request(self.identity, bad)
        receiver.extend_request(self.identity, self.replica_blocks)
        receiver.apply(message, [second])
        self.assert_rows(self.replica_blocks, 0, 250, first_rows)
        self.assert_rows(self.replica_blocks, 250, 15, second_rows)

    def test_failed_allocation_reservation_is_atomic(self):
        other = Identity(102, 2, 8)
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.receiver.begin_request(other, {"hca_kv": [0, 1], "indexer_kv": [1, 2]})
        # Pages checked before the conflicting page must not remain reserved.
        self.receiver.begin_request(other, {"hca_kv": [0], "indexer_kv": [1]})
        with self.assertRaisesRegex(ValueError, "already"):
            self.receiver.begin_request(other, {"hca_kv": [5], "indexer_kv": [5]})

    def test_validation_of_later_request_precedes_all_writes(self):
        other = Identity(102, 2, 8)
        span = Span(self.identity, 0, 0, 3)
        missing = Span(other, 0, 0, 3)
        message = self.export(
            [span, missing],
            tables={
                self.identity: self.producer_blocks,
                other: {"hca_kv": [4], "indexer_kv": [2]},
            },
        )
        before = self.snapshots()
        with self.assertRaisesRegex(ValueError, "identity"):
            self.receiver.apply(message, [span, missing])
        self.assert_unchanged(before)

    def test_write_failure_never_publishes_partially_imported_state(self):
        span = Span(self.identity, 0, 0, 3)
        self.seed(self.producer_blocks, 0, 3)
        message = self.export([span])
        original = codec.V41PagedCache.scatter

        def fail_second(cache, rows, positions, **kwargs):
            if cache.tag == "indexer_kv":
                raise RuntimeError("injected index copy failure")
            return original(cache, rows, positions, **kwargs)

        with patch.object(codec.V41PagedCache, "scatter", fail_second):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.receiver.apply(message, [span])
        with self.assertRaisesRegex(RuntimeError, "receiver failed"):
            self.receiver.require_ready(self.identity, 3, 0)
        with self.assertRaisesRegex(RuntimeError, "receiver failed"):
            self.receiver.apply(message, [span])
        self.receiver.release(self.identity)

    def test_backend_cannot_silently_fall_back_to_cpu(self):
        receiver = Receiver(self.contract, 2, self.target)
        receiver.begin_request(self.identity, self.replica_blocks)
        span = Span(self.identity, 0, 0, 3)
        before = self.snapshots()
        with self.assertRaisesRegex(ValueError, "CUDA"):
            receiver.apply(self.export([span]), [span])
        self.assert_unchanged(before)

    def test_rejects_invalid_contract_identity_spans_and_tables(self):
        for digest in ("", "a" * 63, "A" * 64, "g" * 64):
            with self.assertRaises(ValueError):
                Contract(digest, "b" * 64, "c" * 64, 4096)
        for args in ((True, 1, 1), (0, -1, 1), (0, 1, -1), (1 << 63, 1, 1)):
            with self.assertRaises(ValueError):
                Identity(*args)
        for chunk, start, count in (
            (-1, 0, 1),
            (0, -1, 1),
            (0, 0, 0),
            (0, 0, True),
            (0, (1 << 63) - 1, 1),
        ):
            with self.assertRaises(ValueError):
                Span(self.identity, chunk, start, count)
        other = Identity(102, 2, 9)
        for blocks in (
            {"hca_kv": [0]},
            {"hca_kv": [-1], "indexer_kv": [1]},
            {"hca_kv": [6], "indexer_kv": [1]},
            {"hca_kv": [0, 0], "indexer_kv": [1, 2]},
            {"hca_kv": [0], "indexer_kv": [1, 2]},
        ):
            with self.assertRaises(ValueError):
                self.receiver.begin_request(other, blocks)
        for cp_rank in (-1, 4, True):
            with self.assertRaises(ValueError):
                Receiver(self.contract, cp_rank, self.target)
        with self.assertRaises(ValueError):
            self.export([Span(self.identity, 0, 4096, 1)])
        span = Span(self.identity, 0, 0, 3)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.export([span, span])
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.export(
                [span, Span(other, 0, 0, 3)],
                tables={
                    self.identity: self.producer_blocks,
                    other: self.producer_blocks,
                },
            )


if __name__ == "__main__":
    unittest.main()
