"""Chunk mapping and cross-layer contracts for the prefill selection chain."""

import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_deepselect as deepselect
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_candidates as candidates
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_topk as topk
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import (
    _apply_prefill_candidates,
    mask_candidate_logits,
    select_candidate_blocks,
)


def reference_topk(logits, visible, topk=512, *, bounds=None, out=None):
    """Test oracle: score descending, then original key index ascending.

    Native selectors permit different membership/order within exact ties.
    Canonicalize only in tests that need repeatable downstream precision;
    retain causal/candidate masks and the selected-nonfinite -> -1 contract.
    """
    columns = torch.arange(logits.shape[1], device=logits.device)
    starts, ends = (
        bounds if bounds is not None else (torch.zeros_like(visible), visible)
    )
    valid = (columns[None] >= starts[:, None]) & (columns[None] < ends[:, None])
    scores = logits.masked_fill(~valid, -torch.inf)
    ids = scores.argsort(dim=-1, descending=True, stable=True)[:, :topk]
    selected = ids.int().masked_fill(~scores.gather(1, ids).isfinite(), -1)
    return selected if out is None else out.copy_(selected)


@contextmanager
def reference_prefill_selection():
    """Scoped selector mocks for tests; restore native functions on exit."""
    with patch.object(
        topk, "try_select_tokens", side_effect=reference_topk
    ), patch.object(deepselect, "is_available", return_value=False):
        yield


class PrefillSelectionReferenceTest(unittest.TestCase):
    def test_ties_masks_and_destination_match_across_batch_shapes(self):
        scores = torch.tensor([[4.0, 4.0, -torch.inf, 4.0, 9.0]])
        visible = torch.tensor([4])
        original = scores.clone()
        expected = torch.tensor([[0, 1, 3, -1]], dtype=torch.int32)
        with reference_prefill_selection():
            self.assertFalse(deepselect.is_available(scores.device))
            for rows in (1, 3, 768):
                out = torch.empty(rows, 4, dtype=torch.int32)
                result = topk.try_select_tokens(
                    scores.expand(rows, -1), visible.expand(rows), 4, out=out
                )
                self.assertIs(result, out)
                torch.testing.assert_close(result, expected.expand(rows, -1))
        torch.testing.assert_close(scores, original)

    def test_reference_patch_is_restored_on_failure(self):
        native_topk, native_available = topk.try_select_tokens, deepselect.is_available
        with self.assertRaisesRegex(RuntimeError, "probe failed"):
            with reference_prefill_selection():
                raise RuntimeError("probe failed")
        self.assertIs(topk.try_select_tokens, native_topk)
        self.assertIs(deepselect.is_available, native_available)

    def test_bounds_and_nonfinite_entries_keep_the_selector_contract(self):
        scores = torch.tensor([[9.0, torch.inf, 3.0, 3.0, 9.0]])
        selected = reference_topk(
            scores, torch.tensor([5]), 3, bounds=(torch.tensor([1]), torch.tensor([4]))
        )
        torch.testing.assert_close(
            selected, torch.tensor([[-1, 2, 3]], dtype=torch.int32)
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PrefillSelectionIntegrationTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(
            os.environ,
            {
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
        # Exercise the unsupported-helper fallback without a product A/B flag.
        with patch.object(candidates, "select_candidates", return_value=None):
            _apply_prefill_candidates(shared, logits, visible, slice(None), 8, 71, True)
        self.assertNotIn("prefill_candidate_mask", shared)
        expected = select_candidate_blocks(logits, visible, 8, 71)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        reference = mask_candidate_logits(logits.clone(), expected, 8)
        _apply_prefill_candidates(shared, logits, visible, slice(None), 8, 71, False)
        torch.testing.assert_close(logits, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
