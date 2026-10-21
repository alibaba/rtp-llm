"""CPU tests of UQI mask semantics, including middle segments and padding."""
import random
import unittest
from types import SimpleNamespace

import torch
from rtp_llm.models_py.modules.factory.attention import bert_uqi_schedule as module

build = module.build_uqi_graph_layout


class UqiLayoutTest(unittest.TestCase):
    def test_aggregate_bucket_exceeds_per_sequence_limit(self):
        capacity, requests, limit = 6144, 16, 512
        packed = module.BertUqiAiterGraphLayout(capacity, requests, limit)
        for lengths, starts, counts in (
            ([512] * 12 + [0] * 4, [128] * 12 + [-1] * 4, [64] * 12 + [0] * 4),
            ([7, 5] + [0] * 14, [2, 0] + [-1] * 14, [3, 5] + [0] * 14),
        ):
            packed.update(lengths, starts, counts)
            expected = module.build_uqi_aiter_layout(lengths, starts, counts, capacity, limit)
            for actual, reference in zip(
                (packed.perm, packed.inv_perm, packed.split_indptr, packed.full_indptr, packed.b_mask, packed.valid_mask),
                expected,
            ):
                self.assertEqual(list(actual), list(reference))
            self.assertTrue(all(0 <= b - a <= limit for a, b in zip(packed.full_indptr, packed.full_indptr[1:])))
            self.assertEqual(packed.full_indptr[-1], capacity)

    def test_visibility_matches_original_block_mask(self):
        for lengths, starts, counts in [
            ([5, 3, 0], [1, 0, -1], [2, 3, 0]),
            ([8, 0, 0], [-1, -1, -1], [0, 0, 0]),
            ([2, 3, 1], [1, -1, 0], [1, 0, 1]),
            ([0, 0, 0], [-1, -1, -1], [0, 0, 0]),
        ]:
            with self.subTest(lengths=lengths, counts=counts):
                layout = build(lengths, starts, counts, 8)
                self.assertEqual(len(layout), len(lengths) + 1)
                self.assertEqual(sum(n for n, _, _ in layout), 8)
                for (n, start, count), original_n, original_start, original_count in zip(
                    layout, lengths, starts, counts
                ):
                    self.assertEqual(n, original_n)
                    for q in range(n):
                        for k in range(n):
                            expected = (
                                original_start <= q < original_start + original_count
                                or not original_start <= k < original_start + original_count
                            )
                            actual = start <= q < start + count or not start <= k < start + count
                            self.assertEqual(actual, expected)
                self.assertEqual(layout[-1], (8 - sum(lengths), 8 - sum(lengths), 0))

    def test_aiter_layout_permutation_and_profiles(self):
        for lengths, starts, counts in [([5, 3, 0], [1, 0, -1], [2, 3, 0]),
                                         ([8, 0, 0], [-1]*3, [0]*3),
                                         ([0, 0, 0], [-1]*3, [0]*3)]:
            perm, inv, first, full, mask, valid = module.build_uqi_aiter_layout(
                lengths, starts, counts, 32, 8)
            self.assertEqual(sorted(perm), list(range(32)))
            self.assertEqual([perm[i] for i in inv], list(range(32)))
            self.assertEqual((len(first), len(full)), (11, 8))
            for ptr in (first, full):
                self.assertEqual(ptr[-1], 32)
                self.assertTrue(all(0 <= b-a <= 8 for a,b in zip(ptr, ptr[1:])))
            self.assertEqual(sum(mask), sum(counts))
            self.assertEqual(valid, [True]*sum(lengths)+[False]*(32-sum(lengths)))
            offset = 0
            for n, start, count in zip(lengths, starts, counts):
                for local in range(n):
                    original = offset + local
                    self.assertEqual(mask[inv[original]], start <= local < start + count)
                offset += n

    def test_invalid_metadata(self):
        for args in [([2], [1], [2], 4), ([5], [-1], [0], 4),
                     ([2], [], [], 4), ([-1], [-1], [0], 4)]:
            with self.assertRaises(ValueError):
                build(*args)

    def test_packed_layout_matches_reference_across_changing_requests(self):
        rng = random.Random(20261019)
        for capacity, requests, limit in ((32, 3, 8), (4096, 32, 512), (6144, 32, 512)):
            packed = module.BertUqiAiterGraphLayout(capacity, requests, limit)
            buffer = packed.buffer
            cases = [
                ([0] * requests, [-1] * requests, [0] * requests),
                ([limit] + [0] * (requests - 1), [0] + [-1] * (requests - 1),
                 [limit] + [0] * (requests - 1)),
            ]
            for _ in range(100):
                lengths, starts, counts = [], [], []
                remaining = capacity
                for _ in range(requests):
                    n = rng.randint(0, min(limit, remaining))
                    count = rng.randint(0, n)
                    lengths.append(n)
                    starts.append(rng.randint(0, n - count) if count else -1)
                    counts.append(count)
                    remaining -= n
                cases.append((lengths, starts, counts))
            # Alternate unrelated layouts, then return to empty and pure-B to
            # catch stale permutation, mask and indptr values after reuse.
            for lengths, starts, counts in cases + cases[:2]:
                with self.subTest(capacity=capacity, lengths=lengths):
                    expected = module.build_uqi_aiter_layout(
                        lengths, starts, counts, capacity, limit
                    )
                    packed.update(lengths, starts, counts)
                    actual = tuple(list(view) for view in (
                        packed.perm, packed.inv_perm, packed.split_indptr, packed.full_indptr,
                        packed.b_mask, packed.valid_mask,
                    ))
                    self.assertEqual(actual, expected)
                    self.assertIs(packed.buffer, buffer)
                    self.assertEqual(packed.split_indptr_offset % 8, 0)
                    self.assertEqual(packed.full_indptr_offset % 4, 0)

    def test_packed_layout_rejects_invalid_requests_without_overwriting_plan(self):
        packed = module.BertUqiAiterGraphLayout(8, 2, 4)
        packed.update([3, 2], [1, -1], [1, 0])
        previous = bytes(packed.buffer)
        for args in (
            ([3], [1], [1]),
            ([5, 0], [-1, -1], [0, 0]),
            ([4, 4], [3, -1], [2, 0]),
            ([4, 4], [-1], [0]),
            ([-1, 0], [-1, -1], [0, 0]),
        ):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    packed.update(*args)
                self.assertEqual(bytes(packed.buffer), previous)

    def test_packed_layout_full_bucket_middle_and_edge_segments(self):
        packed = module.BertUqiAiterGraphLayout(12, 3, 5)
        lengths, starts, counts = [5, 5, 2], [1, 0, -1], [3, 5, 0]
        packed.update(lengths, starts, counts)
        self.assertEqual(list(packed.perm), [0, 4, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11])
        self.assertEqual([packed.perm[i] for i in packed.inv_perm], list(range(12)))
        self.assertEqual(list(packed.b_mask), [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        self.assertTrue(all(packed.valid_mask))
        self.assertEqual(list(packed.split_indptr), [0, 2, 5, 5, 10, 12, 12, 12, 12, 12])
        self.assertEqual(list(packed.full_indptr), [0, 5, 10, 12, 12, 12, 12])


class UqiHostMetadataTest(unittest.TestCase):
    def test_scan_matches_dense_segments_with_visual_markers_and_missing_sep(self):
        tokens = torch.tensor(
            [101, 2, 9, 102, 7, 2, 10, 101, 2, 8, 102], dtype=torch.int32
        )
        lengths = torch.tensor([7, 4], dtype=torch.int32)
        # The first marker belongs to a visual row and must be ignored.
        mask = torch.tensor([1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32)
        starts, counts = module.derive_bert_uqi_bounds(tokens, lengths, mask)
        self.assertEqual(starts.tolist(), [5, 1])
        self.assertEqual(counts.tolist(), [2, 3])
        attn = SimpleNamespace(
            input_lengths=lengths, uqi_b_starts=starts, uqi_b_lens=counts
        )
        self.assertTrue(module.valid_bert_uqi_bounds(attn))
        attn.uqi_b_lens[0] = 3
        self.assertFalse(module.valid_bert_uqi_bounds(attn))

    def test_qi_and_empty_sequences_have_no_user_segment(self):
        starts, counts = module.derive_bert_uqi_bounds(
            torch.tensor([101, 9, 102], dtype=torch.int32),
            torch.tensor([0, 3, 0], dtype=torch.int32),
        )
        self.assertEqual(starts.tolist(), [-1, -1, -1])
        self.assertEqual(counts.tolist(), [0, 0, 0])

    def test_malformed_host_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            module.derive_bert_uqi_bounds(torch.tensor([1, 2]), torch.tensor([3]))


if __name__ == "__main__":
    unittest.main()
