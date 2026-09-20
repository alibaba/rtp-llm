"""Chunk mapping and cross-layer contracts for the prefill selection chain."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_candidates as candidates
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_topk as topk
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import (
    _apply_prefill_candidates,
    mask_candidate_logits,
    select_candidate_blocks,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PrefillSelectionIntegrationTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(
            os.environ,
            {
                "DSV41_FUSED_PREFILL_CANDIDATES": "1",
                "DSV41_FUSED_PREFILL_TOPK": "1",
                "DSV4_TOPK_V3": "1",
                "DSV41_PREFILL_CANDIDATE_FLAGS_MAX_BYTES": str(256 * 1024**2),
            },
        )
        env.start()
        self.addCleanup(env.stop)
        torch.manual_seed(923)

    def _check_chain(self, ragged, cache_bitmap):
        rows, block_size, candidate_k = 11, 8, 71
        if ragged:
            # Request rows are interleaved, and requests have unequal key counts.
            chunks = [
                (torch.tensor([7, 0, 9], device="cuda"), 769),
                (torch.tensor([4, 2], device="cuda"), 769),
                (torch.tensor([1, 8, 3], device="cuda"), 1021),
                (torch.tensor([6, 10, 5], device="cuda"), 1021),
            ]
        else:
            chunks = [(slice(i, min(i + 3, rows)), 1021) for i in range(0, rows, 3)]
        output = torch.full((rows, candidate_k), -1, device="cuda", dtype=torch.int32)
        expected = output.clone()
        shared = {"candidates": output}
        if cache_bitmap:
            # Poison the cache; source chunks must overwrite all owned words.
            flags = torch.full(
                (rows, candidates.bitmap_words(1021, block_size) + 3),
                -1,
                device="cuda",
                dtype=torch.int32,
            )
            shared["prefill_candidate_mask"] = (output, flags, block_size)
        prepared = []
        for mapping, width in chunks:
            count = output[mapping].shape[0]
            visible = torch.linspace(0, width, count, device="cuda").long()
            columns = torch.arange(width, device="cuda")
            backing = torch.randn((count, width + 259), device="cuda")
            logits = backing[:, 1 : width + 1]
            logits.masked_fill_(columns[None] >= visible[:, None], -torch.inf)
            expected[mapping] = select_candidate_blocks(
                logits, visible, block_size, candidate_k
            )
            _apply_prefill_candidates(
                shared, logits, visible, mapping, block_size, candidate_k, True
            )
            prepared.append((mapping, logits, visible))
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

        rebuild = candidates.build_flags
        with patch.object(candidates, "build_flags", wraps=rebuild) as build:
            for layer in range(4):
                for mapping, source, visible in prepared:
                    # Keep a padded stride at the producer/selector boundary.
                    backing = torch.empty(
                        (source.shape[0], source.shape[1] + 257), device="cuda"
                    )
                    logits = backing[:, : source.shape[1]]
                    logits.copy_(source * (layer + 1))
                    reference = mask_candidate_logits(
                        logits.clone(), expected[mapping], block_size
                    )
                    _apply_prefill_candidates(
                        shared, logits, visible, mapping, block_size, candidate_k, False
                    )
                    torch.testing.assert_close(logits, reference, rtol=0, atol=0)
                    selected = topk.try_select_tokens(logits, visible)
                    self.assertIsNotNone(selected)
                    scores, ids = reference.topk(512)
                    reference_ids = torch.where(scores.isfinite(), ids, -1)
                    actual_scores = logits.gather(1, selected.long().clamp_min(0))
                    actual_scores.masked_fill_(selected < 0, -torch.inf)
                    scores.masked_fill_(reference_ids < 0, -torch.inf)
                    torch.testing.assert_close(
                        actual_scores.sort(-1).values,
                        scores.sort(-1).values,
                        rtol=0,
                        atol=0,
                    )
            self.assertEqual(build.call_count, 0 if cache_bitmap else 4 * len(chunks))

    def test_contiguous_chunks_reuse_one_bitmap_for_four_consumers(self):
        self._check_chain(ragged=False, cache_bitmap=True)

    def test_ragged_chunks_scatter_bitmap_and_keep_request_widths(self):
        self._check_chain(ragged=True, cache_bitmap=True)

    def test_uncached_chunks_rebuild_from_exact_candidate_ids(self):
        self._check_chain(ragged=True, cache_bitmap=False)

    def test_source_fallback_invalidates_partial_bitmap(self):
        logits = torch.randn((5, 769), device="cuda")
        visible = torch.full((5,), 769, device="cuda", dtype=torch.long)
        output = torch.empty((5, 71), device="cuda", dtype=torch.int32)
        flags = torch.empty((5, 4), device="cuda", dtype=torch.int32)
        shared = {"candidates": output, "prefill_candidate_mask": (output, flags, 8)}
        with patch.dict(os.environ, {"DSV41_FUSED_PREFILL_CANDIDATES": "0"}):
            _apply_prefill_candidates(shared, logits, visible, slice(None), 8, 71, True)
        self.assertNotIn("prefill_candidate_mask", shared)
        expected = select_candidate_blocks(logits, visible, 8, 71)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        reference = mask_candidate_logits(logits.clone(), expected, 8)
        _apply_prefill_candidates(shared, logits, visible, slice(None), 8, 71, False)
        torch.testing.assert_close(logits, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
