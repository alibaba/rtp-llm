"""UT: ``combine_topk_swa_indices`` Triton kernel.

Validates the per-query combined index row used by FP8 SWA prefill
sparse_fwd (``_swa_prefill_ops_triton.combine_topk_swa_indices``).

Per-query (query at absolute sequence position ``pos`` in batch ``b``):

    topk_len = min((pos+1) // compress_ratio, TOP_K)
    swa_len  = min(pos+1, WINDOW_SIZE)

    row[:topk_len]                  = topk_indices[token, :topk_len] + M * b
    row[topk_len : topk_len+swa_len] = M*b + N + (pos - swa_len + 1
                                                  - gather_start) + arange
        where gather_start = seq_len - gather_len

    combined_lens[token] = topk_len + swa_len
    row tail (>= topk_len + swa_len) stays at sentinel (-1) from the
    pre-allocation.

Compared against a Python reference. Coverage:
  * SWA-only (TOP_K=0) — used by V4-Flash layer 0
  * mixed mode (TOP_K>0) — used by C4A/C128A layers
  * combined_topk = align(TOP_K + WINDOW_SIZE, 128) padding correctness
  * gather_start offset (continuation prefill)
  * tail sentinel (-1) preserved past combined_lens

Run:
  CUDA_VISIBLE_DEVICES=7 /opt/conda310/bin/python3 -m unittest \\
    rtp_llm.models_py.modules.dsv4.test.test_swa_combine_topk
"""

from __future__ import annotations

import unittest
from typing import List

import torch

from rtp_llm.models_py.modules.dsv4.fp8._swa_ops_triton import (
    _SPARSE_PREFILL_TOPK_ALIGNMENT,
    combine_topk_swa_indices,
    combine_topk_swa_indices_cp,
)


def _ref_combine(
    topk_indices: torch.Tensor,  # [num_tokens, topk] int32 (chunk-local)
    query_start_loc: torch.Tensor,  # [num_reqs+1] int32 (chunk-local)
    seq_lens: torch.Tensor,  # [num_reqs] int32
    gather_lens: torch.Tensor,  # [num_reqs] int32
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
):
    """Pure-torch reference matching the Triton kernel."""
    num_tokens = int(topk_indices.shape[0])
    aligned = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    out = torch.full(
        (num_tokens, aligned),
        -1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    lens = torch.zeros(num_tokens, dtype=torch.int32, device=topk_indices.device)

    qsl = query_start_loc.tolist()
    base = qsl[0]
    seq_lens_l = seq_lens.tolist()
    gather_lens_l = gather_lens.tolist()
    num_reqs = int(seq_lens.shape[0])

    for b in range(num_reqs):
        qs = qsl[b] - base
        qe = qsl[b + 1] - base
        query_len = qe - qs
        seq_len = seq_lens_l[b]
        gather_len = gather_lens_l[b]
        start_pos = seq_len - query_len
        gather_start = seq_len - gather_len
        for tok_local in range(query_len):
            tok = qs + tok_local
            pos = start_pos + tok_local
            topk_len = min((pos + 1) // compress_ratio, topk)
            swa_len = min(pos + 1, window_size)
            # Compressed-attn part
            for j in range(topk_len):
                out[tok, j] = int(topk_indices[tok, j].item()) + M * b
            # SWA part
            base_swa = M * b + N + (pos - swa_len + 1 - gather_start)
            for j in range(swa_len):
                out[tok, topk_len + j] = base_swa + j
            lens[tok] = topk_len + swa_len
    return out, lens


class SwaCombineTopkTest(unittest.TestCase):

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(0)

    def _check(
        self,
        num_tokens: int,
        topk: int,
        compress_ratio: int,
        window_size: int,
        seq_lens: List[int],
        query_start_loc: List[int],
        gather_lens: List[int],
        N: int,
        M_extra: int = 0,
    ):
        if topk > 0:
            topk_indices = torch.randint(
                0,
                max(N, 1),
                (num_tokens, topk),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices = torch.empty(
                (num_tokens, 0), dtype=torch.int32, device=self.device
            )
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=self.device)
        qsl_t = torch.tensor(query_start_loc, dtype=torch.int32, device=self.device)
        gather_lens_t = torch.tensor(gather_lens, dtype=torch.int32, device=self.device)
        # M is the per-batch workspace stride. SWA-only: gather_len_max + M_extra
        # (caller in production uses gather_len_max).
        gather_len_max = max(gather_lens)
        M = max(N + window_size + gather_len_max + M_extra, 1)

        got_indices, got_lens = combine_topk_swa_indices(
            topk_indices=topk_indices,
            query_start_loc=qsl_t,
            seq_lens=seq_lens_t,
            gather_lens=gather_lens_t,
            window_size=window_size,
            compress_ratio=compress_ratio,
            topk=topk,
            M=M,
            N=N,
        )
        ref_indices, ref_lens = _ref_combine(
            topk_indices,
            qsl_t,
            seq_lens_t,
            gather_lens_t,
            window_size,
            compress_ratio,
            topk,
            M,
            N,
        )
        # Lens
        self.assertEqual(got_lens.shape, ref_lens.shape)
        self.assertTrue(
            torch.equal(got_lens, ref_lens),
            msg=(
                f"combined_lens mismatch:\n"
                f"  got={got_lens.cpu().tolist()}\n"
                f"  ref={ref_lens.cpu().tolist()}"
            ),
        )
        # Indices: only compare valid positions [0, len). Tail must be -1.
        self.assertEqual(got_indices.shape, ref_indices.shape)
        for t in range(num_tokens):
            n = int(got_lens[t].item())
            got_row = got_indices[t, :n].cpu().tolist()
            ref_row = ref_indices[t, :n].cpu().tolist()
            self.assertEqual(
                got_row,
                ref_row,
                msg=f"token {t} prefix mismatch:\n  got={got_row}\n  ref={ref_row}",
            )
            tail = got_indices[t, n:].cpu().tolist()
            for v in tail:
                self.assertEqual(
                    v,
                    -1,
                    msg=f"token {t} tail at offset {n}+ must be -1 sentinel, got {v}",
                )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_swa_only_cold_prefill(self):
        """V4 layer-0 use case: TOP_K=0, sp=0, single request."""
        S = 200
        win = 128
        self._check(
            num_tokens=S,
            topk=0,
            compress_ratio=1,
            window_size=win,
            seq_lens=[S],
            query_start_loc=[0, S],
            gather_lens=[S],  # cold prefill: query_len + 0
            N=0,
        )

    def test_swa_only_continuation_prefill(self):
        """sp>0: gather_start = seq_len - gather_len shifts SWA indices."""
        S, sp, win = 80, 100, 128
        # gather_len = query_len + min(sp, win-1) = 80 + min(100, 127) = 180
        gather_len = S + min(sp, win - 1)
        self._check(
            num_tokens=S,
            topk=0,
            compress_ratio=1,
            window_size=win,
            seq_lens=[sp + S],
            query_start_loc=[0, S],
            gather_lens=[gather_len],
            N=0,
        )

    def test_mixed_mode_c4a(self):
        """TOP_K>0: simulates C4A compressed attention combined with SWA."""
        S = 100
        topk = 16
        compress_ratio = 4
        win = 64
        self._check(
            num_tokens=S,
            topk=topk,
            compress_ratio=compress_ratio,
            window_size=win,
            seq_lens=[S],
            query_start_loc=[0, S],
            gather_lens=[S],
            N=128,  # compressed region of 128 cols
        )

    def test_mixed_mode_c128a(self):
        """compress_ratio=128: topk_len ramps very slowly with pos.

        For small pos < 128 ⇒ topk_len = 0; once pos >= 128 ⇒ topk_len = 1; etc.
        Verifies the (pos+1) // compress_ratio cap is correct.
        """
        S = 300
        self._check(
            num_tokens=S,
            topk=4,
            compress_ratio=128,
            window_size=64,
            seq_lens=[S],
            query_start_loc=[0, S],
            gather_lens=[S],
            N=64,
        )

    def test_alignment_padding(self):
        """combined_topk = align(TOP_K + WINDOW_SIZE, 128). Verify the
        kernel pads tail past topk+win to the alignment with -1 sentinel."""
        S = 30
        topk = 17  # TOP_K + win = 17 + 50 = 67 → align(67,128) = 128
        win = 50
        self._check(
            num_tokens=S,
            topk=topk,
            compress_ratio=2,
            window_size=win,
            seq_lens=[S],
            query_start_loc=[0, S],
            gather_lens=[S],
            N=32,
        )

    def test_window_size_caps_swa_len(self):
        """For pos+1 > WINDOW_SIZE, swa_len caps at WINDOW_SIZE."""
        S = 200
        win = 32
        self._check(
            num_tokens=S,
            topk=0,
            compress_ratio=1,
            window_size=win,
            seq_lens=[S],
            query_start_loc=[0, S],
            gather_lens=[S],
            N=0,
        )

    def test_empty_input(self):
        """num_tokens=0 returns empty (combined_indices, combined_lens)."""
        out_idx, out_lens = combine_topk_swa_indices(
            topk_indices=torch.empty((0, 0), dtype=torch.int32, device=self.device),
            query_start_loc=torch.tensor([0], dtype=torch.int32, device=self.device),
            seq_lens=torch.empty(0, dtype=torch.int32, device=self.device),
            gather_lens=torch.empty(0, dtype=torch.int32, device=self.device),
            window_size=128,
            compress_ratio=1,
            topk=0,
            M=128,
            N=0,
        )
        self.assertEqual(out_idx.shape[0], 0)
        self.assertEqual(out_lens.shape[0], 0)

    def _check_cp_against_ref(
        self,
        topk_indices: torch.Tensor,
        global_positions: torch.Tensor,
        sp_int: int,
        window_size: int,
        compress_ratio: int,
        topk: int,
        M: int,
        N: int,
        req_id_per_token: torch.Tensor | None = None,
        prefix_lengths: torch.Tensor | None = None,
    ):
        from rtp_llm.models_py.modules.dsv4.cp import (
            combine_topk_swa_indices_cp_b1,
            combine_topk_swa_indices_cp_varlen,
        )

        got_idx, got_lens = combine_topk_swa_indices_cp(
            topk_indices=topk_indices,
            global_positions=global_positions,
            sp_int=sp_int,
            window_size=window_size,
            compress_ratio=compress_ratio,
            topk=topk,
            M=M,
            N=N,
            req_id_per_token=req_id_per_token,
            prefix_lengths=prefix_lengths,
        )
        if req_id_per_token is None:
            ref_idx, ref_lens = combine_topk_swa_indices_cp_b1(
                topk_indices=topk_indices,
                global_positions=global_positions,
                sp_int=sp_int,
                window_size=window_size,
                compress_ratio=compress_ratio,
                topk=topk,
                M=M,
                N=N,
            )
        else:
            ref_idx, ref_lens = combine_topk_swa_indices_cp_varlen(
                topk_indices=topk_indices,
                global_positions=global_positions,
                sp_int=sp_int,
                window_size=window_size,
                compress_ratio=compress_ratio,
                topk=topk,
                M=M,
                N=N,
                req_id_per_token=req_id_per_token,
                prefix_lengths=prefix_lengths,
            )
        self.assertTrue(torch.equal(got_lens, ref_lens))
        self.assertTrue(torch.equal(got_idx, ref_idx))

    def test_cp_fused_b1_zigzag(self):
        seq_full = 64
        cp_size = 2
        rank = 0
        pair = seq_full // (cp_size * 2)
        gp = torch.cat(
            [
                torch.arange(rank * pair, (rank + 1) * pair, device=self.device),
                torch.arange(
                    seq_full - (rank + 1) * pair,
                    seq_full - rank * pair,
                    device=self.device,
                ),
            ]
        ).to(torch.int64)
        topk = 8
        ratio = 4
        win = 16
        N = seq_full // ratio
        M = N + seq_full
        topk_indices = torch.randint(
            0, max(1, N), (gp.numel(), topk), dtype=torch.int32, device=self.device
        )
        self._check_cp_against_ref(
            topk_indices=topk_indices,
            global_positions=gp,
            sp_int=0,
            window_size=win,
            compress_ratio=ratio,
            topk=topk,
            M=M,
            N=N,
        )

    def test_cp_fused_varlen_b2_offsets(self):
        win = 8
        ratio = 4
        topk = 4
        N = 6
        M = 32
        prefix = torch.tensor([0, 10], dtype=torch.int64, device=self.device)
        req = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=self.device)
        gp = torch.tensor([0, 7, 10, 13], dtype=torch.int64, device=self.device)
        topk_indices = torch.tensor(
            [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
            dtype=torch.int32,
            device=self.device,
        )
        self._check_cp_against_ref(
            topk_indices=topk_indices,
            global_positions=gp,
            sp_int=0,
            window_size=win,
            compress_ratio=ratio,
            topk=topk,
            M=M,
            N=N,
            req_id_per_token=req,
            prefix_lengths=prefix,
        )

    def test_cp_fused_swa_only_topk0(self):
        seq = 33
        win = 8
        gp = torch.arange(seq, dtype=torch.int64, device=self.device)
        topk_indices = torch.empty((seq, 0), dtype=torch.int32, device=self.device)
        self._check_cp_against_ref(
            topk_indices=topk_indices,
            global_positions=gp,
            sp_int=0,
            window_size=win,
            compress_ratio=1,
            topk=0,
            M=seq,
            N=0,
            req_id_per_token=torch.zeros(seq, dtype=torch.int32, device=self.device),
            prefix_lengths=torch.zeros(1, dtype=torch.int64, device=self.device),
        )


if __name__ == "__main__":
    unittest.main()
