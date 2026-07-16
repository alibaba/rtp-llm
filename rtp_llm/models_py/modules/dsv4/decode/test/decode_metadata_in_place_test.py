"""Stage 3A in-place metadata builder tests.

Verify that ``allocate_decode_metadata + update_decode_metadata_in_place``:

1. Produces tensors that are bit-equal to ``build_decode_metadata`` for the
   same ``start_pos`` and the prefix ``[:bs]``.
2. Does NOT reallocate buffers across calls (data_ptr() unchanged).
3. Tail rows beyond the active prefix retain their (-1 sentinel) state
   so a runtime BS smaller than the alloc size doesn't smear.

Stage 3A foundation only — no CUDA graph capture is exercised here
(no CUDA on dev box). The captured-graph integration test lives in the
SM100_ARM smoke tier.
"""

import os
import sys
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.models_py.modules.dsv4.decode.decode_attn_metadata import (
    allocate_decode_metadata,
    build_decode_metadata,
    update_decode_metadata_in_place,
)

_DEVICE = torch.device("cpu")


def _alloc(max_bs: int, ratios=None):
    return allocate_decode_metadata(
        max_batch_size=max_bs,
        q_len=1,
        window_size=8,
        head_dim=32,
        max_seq_len=64,
        compress_ratios=ratios or [4, 128],
        index_topk=4,
        device=_DEVICE,
    )


def _build(start_pos: torch.Tensor, ratios=None):
    return build_decode_metadata(
        start_pos=start_pos,
        q_len=1,
        window_size=8,
        head_dim=32,
        max_seq_len=64,
        compress_ratios=ratios or [4, 128],
        index_topk=4,
        device=_DEVICE,
    )


class TestUpdateInPlace(unittest.TestCase):

    def test_matches_build_for_swa_csa_hca(self):
        """update_in_place vs build_decode_metadata: prefix bit-equal for
        a mix of SWA-only / CSA / HCA layers (ratios [4, 128])."""
        for start_pos_list in [[0], [3], [4, 7, 11], [0, 1, 2, 3, 4, 5, 6, 7]]:
            start_pos = torch.tensor(start_pos_list, dtype=torch.int32)
            bs = len(start_pos_list)
            ref = _build(start_pos)
            meta = _alloc(max_bs=8)
            update_decode_metadata_in_place(meta, start_pos)

            self.assertTrue(
                torch.equal(meta.start_pos[:bs], ref.start_pos),
                f"start_pos mismatch for bs={bs}",
            )
            self.assertTrue(
                torch.equal(meta.slot_mapping_swa[:bs], ref.slot_mapping_swa),
                f"slot_mapping_swa mismatch for bs={bs}",
            )
            self.assertTrue(
                torch.equal(meta.topk_window_idxs[:bs], ref.topk_window_idxs),
                f"topk_window_idxs mismatch for bs={bs}",
            )
            for r in [4, 128]:
                self.assertTrue(
                    torch.equal(
                        meta.slot_mapping_compressed[r][:bs],
                        ref.slot_mapping_compressed[r],
                    ),
                    f"slot_mapping_compressed[{r}] mismatch for bs={bs}",
                )
                self.assertTrue(
                    torch.equal(meta.compressed_lens[r][:bs], ref.compressed_lens[r]),
                    f"compressed_lens[{r}] mismatch for bs={bs}",
                )
                self.assertTrue(
                    torch.equal(
                        meta.topk_total_by_ratio[r][:bs], ref.topk_total_by_ratio[r]
                    ),
                    f"topk_total_by_ratio[{r}] mismatch for bs={bs}",
                )

    def test_no_realloc_across_calls(self):
        """Multiple update_in_place calls must reuse storage. forbid_realloc=True
        asserts this internally; we also double-check via data_ptr()."""
        meta = _alloc(max_bs=4)

        # Snapshot pointers BEFORE first call (alloc-state)
        before = {
            "start_pos": meta.start_pos.data_ptr(),
            "slot_swa": meta.slot_mapping_swa.data_ptr(),
            "topk_window": meta.topk_window_idxs.data_ptr(),
            "topk_buf": meta.topk_buffer_compressed.data_ptr(),
        }
        for r, t in meta.slot_mapping_compressed.items():
            before[f"slot_cmp[{r}]"] = t.data_ptr()
        for r, t in meta.compressed_lens.items():
            before[f"cmp_lens[{r}]"] = t.data_ptr()
        for r, t in meta.topk_total_by_ratio.items():
            before[f"topk_total[{r}]"] = t.data_ptr()

        # Call 1 with bs=2
        update_decode_metadata_in_place(
            meta,
            torch.tensor([3, 5], dtype=torch.int32),
            forbid_realloc=True,
        )
        # Call 2 with bs=4 (different start_pos, different bs)
        update_decode_metadata_in_place(
            meta,
            torch.tensor([0, 1, 7, 11], dtype=torch.int32),
            forbid_realloc=True,
        )
        # Call 3 with bs=1
        update_decode_metadata_in_place(
            meta,
            torch.tensor([20], dtype=torch.int32),
            forbid_realloc=True,
        )

        after = {
            "start_pos": meta.start_pos.data_ptr(),
            "slot_swa": meta.slot_mapping_swa.data_ptr(),
            "topk_window": meta.topk_window_idxs.data_ptr(),
            "topk_buf": meta.topk_buffer_compressed.data_ptr(),
        }
        for r, t in meta.slot_mapping_compressed.items():
            after[f"slot_cmp[{r}]"] = t.data_ptr()
        for r, t in meta.compressed_lens.items():
            after[f"cmp_lens[{r}]"] = t.data_ptr()
        for r, t in meta.topk_total_by_ratio.items():
            after[f"topk_total[{r}]"] = t.data_ptr()

        for k in before:
            self.assertEqual(before[k], after[k], f"buffer {k} reallocated")

    def test_tail_kept_at_sentinel_after_smaller_bs(self):
        """If we update with bs<max_bs, the tail rows must NOT contain
        leftover data from a previous larger-bs update — they should
        stay at -1 sentinel from the original allocation."""
        meta = _alloc(max_bs=4)
        # Tail before any update
        self.assertTrue((meta.slot_mapping_swa == -1).all())
        self.assertTrue((meta.topk_window_idxs == -1).all())

        # Update with bs=2 — tail rows should remain at sentinel
        update_decode_metadata_in_place(
            meta,
            torch.tensor([3, 5], dtype=torch.int32),
            forbid_realloc=True,
        )
        # Slots beyond bs=2 (which is bs*q_len=2 entries for q_len=1) still -1
        self.assertTrue(
            (meta.slot_mapping_swa[2:] == -1).all(),
            f"slot_mapping_swa tail polluted: {meta.slot_mapping_swa}",
        )
        # topk_window_idxs[2:] still -1
        self.assertTrue(
            (meta.topk_window_idxs[2:] == -1).all(), f"topk_window_idxs tail polluted"
        )

    def test_bs_equals_max_bs_works(self):
        """Stage 3A common case: captured graph at fixed BS == alloc BS."""
        meta = _alloc(max_bs=4)
        update_decode_metadata_in_place(
            meta,
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            forbid_realloc=True,
        )
        ref = _build(torch.tensor([0, 1, 2, 3], dtype=torch.int32))
        self.assertTrue(torch.equal(meta.slot_mapping_swa, ref.slot_mapping_swa))
        self.assertTrue(torch.equal(meta.topk_window_idxs, ref.topk_window_idxs))


if __name__ == "__main__":
    unittest.main()
