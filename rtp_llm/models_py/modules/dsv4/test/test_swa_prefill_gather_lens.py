"""UT: ``compute_prefill_gather_lens`` Triton kernel.

Validates the per-prefill-request gather length used by FP8 SWA
continuation prefill (``_swa_prefill_ops_triton.compute_prefill_gather_lens``):

    query_len  = query_start_loc[num_decodes+i+1] - query_start_loc[num_decodes+i]
    prefix_len = seq_lens[num_decodes+i] - query_len
    gather_len = query_len + min(prefix_len, window_size - 1)

The ``window_size - 1`` cap is the SWA invariant: the earliest
prefill query at sequence position p reuses at most ``win-1``
prefix tokens (positions ``p-win+1 .. p-1``).

Compared against a Python reference. Coverage:
  * cold prefill (prefix=0)
  * continuation prefill, prefix < win-1 (no truncation)
  * continuation prefill, prefix > win-1 (truncated to win-1)
  * decodes preceding prefills (num_decodes>0): kernel must skip the
    decode segment of seq_lens / query_start_loc
  * num_prefills==0 (early-out)
  * multi-request batch with mixed prefix lengths

Run:
  CUDA_VISIBLE_DEVICES=7 /opt/conda310/bin/python3 -m unittest \\
    rtp_llm.models_py.modules.dsv4.test.test_swa_prefill_gather_lens
"""

from __future__ import annotations

import unittest
from typing import List

import torch

from rtp_llm.models_py.modules.dsv4._swa_prefill_ops_triton import (
    compute_prefill_gather_lens,
)


def _ref_gather_lens(
    seq_lens: List[int],  # [num_decodes + num_prefills]
    query_start_loc: List[int],  # [num_decodes + num_prefills + 1]
    num_prefills: int,
    num_decodes: int,
    window_size: int,
) -> List[int]:
    out = []
    for i in range(num_prefills):
        idx = num_decodes + i
        query_len = query_start_loc[idx + 1] - query_start_loc[idx]
        prefix_len = seq_lens[idx] - query_len
        out.append(query_len + min(prefix_len, window_size - 1))
    return out


class SwaPrefillGatherLensTest(unittest.TestCase):

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")

    def _check(
        self,
        seq_lens: List[int],
        query_start_loc: List[int],
        num_prefills: int,
        num_decodes: int,
        window_size: int,
    ):
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=self.device)
        qsl_t = torch.tensor(query_start_loc, dtype=torch.int32, device=self.device)
        got = compute_prefill_gather_lens(
            seq_lens=seq_lens_t,
            query_start_loc=qsl_t,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            window_size=window_size,
        )
        ref = _ref_gather_lens(
            seq_lens, query_start_loc, num_prefills, num_decodes, window_size
        )
        self.assertEqual(got.shape, (num_prefills,))
        self.assertEqual(got.dtype, torch.int32)
        self.assertEqual(
            got.cpu().tolist(),
            ref,
            msg=f"gather_lens got={got.cpu().tolist()} expected={ref}",
        )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_cold_prefill_single_request(self):
        """B=1, sp=0 → gather_len = query_len + min(0, win-1) = query_len."""
        self._check(
            seq_lens=[200],
            query_start_loc=[0, 200],
            num_prefills=1,
            num_decodes=0,
            window_size=512,
        )

    def test_continuation_prefix_below_win(self):
        """prefix_len=100 < win-1=511 → gather_len = 50 + 100 = 150."""
        self._check(
            seq_lens=[150],  # sp=100, query_len=50
            query_start_loc=[0, 50],
            num_prefills=1,
            num_decodes=0,
            window_size=512,
        )

    def test_continuation_prefix_above_win(self):
        """prefix_len=2000 > win-1=511 → gather_len = 50 + 511 = 561."""
        self._check(
            seq_lens=[2050],  # sp=2000, query_len=50
            query_start_loc=[0, 50],
            num_prefills=1,
            num_decodes=0,
            window_size=512,
        )

    def test_continuation_prefix_exact_boundary(self):
        """prefix_len = win-1 exactly → no clipping."""
        self._check(
            seq_lens=[100 + 511],
            query_start_loc=[0, 100],
            num_prefills=1,
            num_decodes=0,
            window_size=512,
        )

    def test_with_decodes_preceding(self):
        """num_decodes>0: kernel must offset both seq_lens and qsl by num_decodes.

        Layout: 2 decodes, 2 prefills (total 4 reqs).
        decode reqs have seq_lens 1000, 1500 and query_len=1 each;
        prefill reqs: req 0 has sp=200 query=50, req 1 cold sp=0 query=300.
        """
        self._check(
            seq_lens=[1000, 1500, 250, 300],
            query_start_loc=[0, 1, 2, 52, 352],
            num_prefills=2,
            num_decodes=2,
            window_size=512,
        )

    def test_multi_request_mixed(self):
        """B=3 with very different prefix lengths: 0 / mid / huge."""
        self._check(
            seq_lens=[100, 600, 10000],
            query_start_loc=[0, 100, 200, 300],  # all query_len=100
            num_prefills=3,
            num_decodes=0,
            window_size=128,
        )

    def test_num_prefills_zero(self):
        """Early-out path: returns empty int32 tensor without launch."""
        seq_lens_t = torch.tensor([100, 200], dtype=torch.int32, device=self.device)
        qsl_t = torch.tensor([0, 1, 2], dtype=torch.int32, device=self.device)
        got = compute_prefill_gather_lens(
            seq_lens=seq_lens_t,
            query_start_loc=qsl_t,
            num_prefills=0,
            num_decodes=2,
            window_size=512,
        )
        self.assertEqual(got.shape, (0,))
        self.assertEqual(got.dtype, torch.int32)

    def test_window_size_1(self):
        """win=1 ⇒ win-1=0 ⇒ never reuse any prefix; gather_len = query_len."""
        self._check(
            seq_lens=[1000],
            query_start_loc=[0, 50],
            num_prefills=1,
            num_decodes=0,
            window_size=1,
        )


if __name__ == "__main__":
    unittest.main()
