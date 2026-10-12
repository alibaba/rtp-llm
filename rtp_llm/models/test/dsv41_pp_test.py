"""Complete PP frame admission with real L20 cache imports, without native ops."""

import dataclasses
import unittest

import torch

from rtp_llm.models.test.dsv41_cache_codec_test import load_codec_source
from rtp_llm.models.test.dsv41_cache_transfer_test import (
    Contract,
    Identity,
    Receiver,
    Span,
    caches,
    transfer,
)

pp = load_codec_source("_dsv41_pp_test_source", "v41_pp.py")


def frame_for(row_map):
    count = row_map.shape[0]
    hidden = torch.randn(count, 4, 5120, dtype=torch.bfloat16)
    pre_mix = torch.rand(count, 4)
    topk = torch.full((count, 512), -1, dtype=torch.int32)
    candidates = torch.full((count, 2048), -1, dtype=torch.int32)
    counts = torch.zeros(count, dtype=torch.int32)
    for row, (_, position, valid) in enumerate(row_map.tolist()):
        if valid:
            topk[row, : min(512, position + 1)] = torch.arange(
                max(0, position - 511), position + 1
            )
            nblocks = min(2048, position // 8 + 1)
            candidates[row, :nblocks] = torch.arange(
                position // 8 + 1 - nblocks, position // 8 + 1
            )
            counts[row] = nblocks
    return pp.V41PPFrame(hidden, pre_mix, row_map.clone(), topk, candidates, counts)


class V41PPFrameTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(413)
        self.source, self.target = caches(), caches()
        self.contract = Contract("a" * 64, "b" * 64, "c" * 64, 4096)
        self.identity = Identity(201, 1, 9)
        self.span = Span(self.identity, 0, 0, 17)
        self.rows = torch.tensor([[0, 0, 1], [0, 7, 1], [-1, -1, 0], [0, 16, 1]])
        self.frame = frame_for(self.rows)
        self.receiver = Receiver(self.contract, 0, self.target, backend="eager")
        self.receiver.begin_request(self.identity, {"hca_kv": [0], "indexer_kv": [1]})
        self.delta = transfer.export_l20_delta(
            self.contract,
            0,
            0,
            [self.span],
            self.source,
            {self.identity: {"hca_kv": [2], "indexer_kv": [3]}},
            backend="eager",
        )

    def wire(self, frame=None):
        return pp.pack_v41_pp_frame(
            self.delta, frame or self.frame, [self.span], self.rows
        )

    def assert_not_imported(self):
        for cache in self.target.values():
            self.assertTrue(bool((cache.raw == 173).all()))
        with self.assertRaises(ValueError):
            self.receiver.require_ready(self.identity, 17, 0)

    def test_roundtrip_preserves_delayed_mix_and_selection_order(self):
        message = self.wire()
        frame = pp.import_v41_pp_frame(message, self.receiver, [self.span], self.rows)
        self.receiver.require_ready(self.identity, 17, 0)
        for name in self.frame.__dict__:
            torch.testing.assert_close(
                getattr(frame, name), getattr(self.frame, name), rtol=0, atol=0
            )
        self.assertIs(frame.hidden_states, message["hidden_states"])
        self.assertIs(frame.prev_pre_mix, message["v41_prev_pre_mix"])

    def test_hidden_only_missing_mix_or_selection_cannot_commit_cache(self):
        message = self.wire()
        for key in (
            "v41_pp_header",
            "v41_prev_pre_mix",
            "v41_row_map",
            "v41_l20_topk",
            "v41_candidate_blocks",
            "v41_candidate_counts",
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                pp.import_v41_pp_frame(
                    {k: v for k, v in message.items() if k != key},
                    self.receiver,
                    [self.span],
                    self.rows,
                )
            self.assert_not_imported()

    def test_bad_shapes_dtypes_and_nonfinite_values_are_rejected_before_import(self):
        message = self.wire()
        wrong = [
            ("hidden_states", self.frame.hidden_states[:, 0]),
            ("v41_prev_pre_mix", self.frame.prev_pre_mix.bfloat16()),
            ("v41_prev_pre_mix", torch.full((4, 4), float("nan"))),
            ("hidden_states", torch.full_like(self.frame.hidden_states, float("inf"))),
            ("v41_row_map", self.rows.int()),
            ("v41_l20_topk", self.frame.topk[:, :511]),
            ("v41_candidate_counts", self.frame.candidate_counts.long()),
            ("v41_pp_header", torch.tensor([1, 4, 4, 4096])),
        ]
        for key, value in wrong:
            with self.subTest(key=key), self.assertRaises(ValueError):
                pp.import_v41_pp_frame(
                    {**message, key: value}, self.receiver, [self.span], self.rows
                )
            self.assert_not_imported()

    def test_wrong_cp_permutation_and_duplicate_rows_fail(self):
        message = self.wire()
        bad = self.rows[[1, 0, 2, 3]]
        with self.assertRaisesRegex(ValueError, "CP dispatch"):
            pp.import_v41_pp_frame(
                {**message, "v41_row_map": bad}, self.receiver, [self.span], self.rows
            )
        duplicates = self.rows.clone()
        duplicates[1] = duplicates[0]
        with self.assertRaisesRegex(ValueError, "repeats"):
            pp.pack_v41_pp_frame(
                self.delta, frame_for(duplicates), [self.span], duplicates
            )
        self.assert_not_imported()

    def test_padding_rows_cannot_carry_stale_selections(self):
        message = self.wire()
        for key in ("v41_l20_topk", "v41_candidate_blocks", "v41_candidate_counts"):
            changed = message[key].clone()
            changed[2] = 0 if changed.ndim == 2 else 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "padding"):
                pp.import_v41_pp_frame(
                    {**message, key: changed}, self.receiver, [self.span], self.rows
                )
        self.assert_not_imported()

    def test_candidate_count_causality_uniqueness_and_newest_block(self):
        message = self.wire()
        for replacement in ([0, 0, 2], [0, 1, 3], [0, 1, -1]):
            candidates = self.frame.candidate_blocks.clone()
            candidates[3, :3] = torch.tensor(replacement)
            with self.assertRaises(ValueError):
                pp.import_v41_pp_frame(
                    {**message, "v41_candidate_blocks": candidates},
                    self.receiver,
                    [self.span],
                    self.rows,
                )
        changed = self.frame.candidate_counts.clone()
        changed[3] = 2
        with self.assertRaisesRegex(ValueError, "count"):
            pp.import_v41_pp_frame(
                {**message, "v41_candidate_counts": changed},
                self.receiver,
                [self.span],
                self.rows,
            )
        self.assert_not_imported()

    def test_topk_requires_all_reachable_selections_without_duplicates(self):
        message = self.wire()
        for value in (-2, 100, 0, -1):
            changed = self.frame.topk.clone()
            changed[1, 1] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "top-k"):
                pp.import_v41_pp_frame(
                    {**message, "v41_l20_topk": changed},
                    self.receiver,
                    [self.span],
                    self.rows,
                )
        self.assert_not_imported()

    def test_query_positions_must_belong_to_the_current_request_chunk(self):
        for row in ([1, 7, 1], [0, 17, 1], [0, 7, 2], [0, -1, 0]):
            changed = self.rows.clone()
            changed[1] = torch.tensor(row)
            frame = dataclasses.replace(self.frame, row_map=changed)
            with self.subTest(row=row), self.assertRaises(ValueError):
                pp.pack_v41_pp_frame(self.delta, frame, [self.span], changed)

    def test_empty_local_frame_retains_full_cp_delta(self):
        rows = torch.empty(0, 3, dtype=torch.int64)
        message = pp.pack_v41_pp_frame(self.delta, frame_for(rows), [self.span], rows)
        self.assertNotIn("hidden_states", message)
        self.assertTrue(all(t.numel() for t in message.values()))
        frame = pp.import_v41_pp_frame(message, self.receiver, [self.span], rows)
        self.assertEqual(frame.hidden_states.shape, (0, 4, 5120))
        self.receiver.require_ready(self.identity, 17, 0)

    def test_frames_are_explicit_and_not_reused_across_forwards(self):
        first = pp.import_v41_pp_frame(
            self.wire(), self.receiver, [self.span], self.rows
        )
        saved_mix = first.prev_pre_mix.clone()
        span = Span(self.identity, 1, 17, 2)
        rows = torch.tensor([[0, 18, 1]])
        delta = transfer.export_l20_delta(
            self.contract,
            0,
            1,
            [span],
            self.source,
            {self.identity: {"hca_kv": [2], "indexer_kv": [3]}},
            backend="eager",
        )
        message = pp.pack_v41_pp_frame(delta, frame_for(rows), [span], rows)
        second = pp.import_v41_pp_frame(message, self.receiver, [span], rows)
        self.assertEqual(second.prev_pre_mix.shape, (1, 4))
        torch.testing.assert_close(first.prev_pre_mix, saved_mix, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
