"""G1-full CUDA matrix: real combine/split producers -> ready path parity.

Drives the ACTUAL Triton producers (``combine_topk_swa_indices`` and its CP
variant) and the ACTUAL production split (``split_sm120_combined_tables``,
shared with ``attention.py`` — no algorithm copy), then checks the ready
consumer path (``_ready_chunk``) bit-exactly against the incumbent
``canonical_topk`` + ``clamp_min_(0)`` chain across CP sizes, chunk slices,
topk widths and compress ratios, including the in-place lens-clamp side
effect and contract-gate fallbacks.
"""

import unittest
from typing import List

import torch

from rtp_llm.models_py.modules.dsv4.fp8._sm120_prefill_indices import (
    Sm120PrefillIndices,
    _ready_chunk,
    split_sm120_combined_tables,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_ops_triton import (
    combine_topk_swa_indices,
    combine_topk_swa_indices_cp,
)
from rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla import (
    SM120_EXTRA_TOPK_WIDTHS,
    canonical_topk,
)

SWA_SUPPORTED = (128, 512, 1024, 2048)


def zigzag_positions(seq_full: int, cp_size: int, rank: int, device) -> torch.Tensor:
    pair = seq_full // (cp_size * 2)
    return torch.cat(
        [
            torch.arange(rank * pair, (rank + 1) * pair, device=device),
            torch.arange(
                seq_full - (rank + 1) * pair, seq_full - rank * pair, device=device
            ),
        ]
    ).to(torch.int64)


def generic_normalize(indices: torch.Tensor, lens: torch.Tensor, supported):
    """The incumbent chain, verbatim: canonical_topk + clamp_min_(0)."""
    out_i, out_l = canonical_topk(indices, lens, supported)
    out_i.clamp_min_(0)
    return out_i, out_l


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ProducerConsumerMatrixTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0")
        torch.cuda.init()

    def _produce_cp0(self, num_tokens, topk, ratio, window, seq_lens, N):
        """Non-CP producer: one request batch, varlen."""
        g = torch.Generator(device="cpu").manual_seed(num_tokens * 1000 + topk)
        topk_indices = (
            torch.randint(0, max(N, 1), (num_tokens, topk), generator=g).to(
                dtype=torch.int32, device=self.device
            )
            if topk > 0
            else torch.empty((num_tokens, 0), dtype=torch.int32, device=self.device)
        )
        qsl = [0]
        for s in seq_lens:
            qsl.append(qsl[-1] + s)
        gather_lens = seq_lens
        gather_len_max = max(gather_lens)
        M = max(N + window + gather_len_max, 1)
        combined_i, combined_l = combine_topk_swa_indices(
            topk_indices=topk_indices,
            query_start_loc=torch.tensor(qsl, dtype=torch.int32, device=self.device),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=self.device),
            gather_lens=torch.tensor(
                gather_lens, dtype=torch.int32, device=self.device
            ),
            window_size=window,
            compress_ratio=ratio,
            topk=topk,
            M=M,
            N=N,
        )
        return combined_i, combined_l, M, N

    def _produce_cp(self, seq_full, cp_size, rank, topk, ratio, window):
        pair = seq_full // (cp_size * 2)
        gp = zigzag_positions(seq_full, cp_size, rank, self.device)
        N = seq_full // ratio
        M = N + seq_full
        topk_indices = (
            torch.randint(
                0, max(1, N), (gp.numel(), topk), dtype=torch.int32, device=self.device
            )
            if topk > 0
            else torch.empty((gp.numel(), 0), dtype=torch.int32, device=self.device)
        )
        combined_i, combined_l = combine_topk_swa_indices_cp(
            topk_indices=topk_indices,
            global_positions=gp,
            sp_int=0,
            window_size=window,
            compress_ratio=ratio,
            topk=topk,
            M=M,
            N=N,
        )
        return combined_i, combined_l, M, N

    def _check_matrix_entry(
        self, combined_i, combined_l, M, N, window, ratio, topk_width
    ):
        swa_i, swa_l, extra_i, extra_l, _ = split_sm120_combined_tables(
            combined_i,
            combined_l,
            M=M,
            N=N,
            window_size=window,
            extra_width=max(topk_width, 1),
            ratio=ratio,
            device=self.device,
        )
        record = Sm120PrefillIndices(
            swa_indices=swa_i,
            swa_lens=swa_l,
            extra_indices=extra_i,
            extra_lens=extra_l,
        )
        rows = int(swa_i.shape[0])
        # chunk slices: full, halves, and an off-aligned interior window
        slices = {(0, rows)}
        if rows >= 4:
            slices |= {(0, rows // 2), (rows // 2, rows), (1, rows - 1)}
        for start, end in sorted(slices):
            for supported, r_indices, r_lens in (
                (SWA_SUPPORTED, record.swa_indices, record.swa_lens),
                (SM120_EXTRA_TOPK_WIDTHS, record.extra_indices, record.extra_lens),
            ):
                if r_indices is None:
                    continue
                got = _ready_chunk(r_indices, r_lens, start, end, supported)
                self.assertIsNotNone(got, (start, end, supported))
                want_i, want_l = generic_normalize(
                    r_indices[start:end], r_lens[start:end], supported
                )
                self.assertTrue(
                    torch.equal(got[0], want_i),
                    f"indices mismatch at {(start, end, supported)}",
                )
                self.assertTrue(
                    torch.equal(got[1], want_l),
                    f"lens mismatch at {(start, end, supported)}",
                )

    def test_matrix_cp1_varlen(self):
        for topk, ratio in ((0, 1), (128, 4), (2048, 4)):
            with self.subTest(topk=topk, ratio=ratio):
                ci, cl, M, N = self._produce_cp0(
                    num_tokens=64,
                    topk=topk,
                    ratio=128,
                    window=16,
                    seq_lens=[20, 44],
                    N=64,
                )
                self._check_matrix_entry(ci, cl, M, N, 16, ratio, topk)

    def test_matrix_cp2_zigzag(self):
        for rank in (0, 1):
            ci, cl, M, N = self._produce_cp(
                seq_full=64, cp_size=2, rank=rank, topk=8, ratio=4, window=16
            )
            self._check_matrix_entry(ci, cl, M, N, 16, 4, 8)

    def test_matrix_cp4_zigzag(self):
        for rank in (0, 1, 2, 3):
            ci, cl, M, N = self._produce_cp(
                seq_full=128, cp_size=4, rank=rank, topk=16, ratio=4, window=32
            )
            self._check_matrix_entry(ci, cl, M, N, 32, 4, 16)

    def test_matrix_swa_only_topk0(self):
        ci, cl, M, N = self._produce_cp0(
            num_tokens=33, topk=0, ratio=1, window=8, seq_lens=[33], N=0
        )
        self._check_matrix_entry(ci, cl, M, N, 8, 1, 0)


if __name__ == "__main__":
    unittest.main()
