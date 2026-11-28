"""CUDA matrix: real combine producers -> generic split oracles.

Drives the ACTUAL Triton producers (``combine_topk_swa_indices`` and its CP
variant) and the ACTUAL production split (``split_sm120_combined_tables``,
shared with ``attention.py`` — no algorithm copy), then checks the raw
combined live prefix/-1 tail and the generic split output bit-exactly
against INDEPENDENT Python oracles across CP sizes, mixed prefixes, topk
widths and compress ratios.
"""

import math
import unittest
from typing import List

import torch

from rtp_llm.models_py.modules.dsv4.fp8._sm120_prefill_indices import (
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
INT32_MAX = 2**31 - 1


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


def split_oracle(live: List[int], M: int, N: int, window: int, ratio: int, k: int):
    """Independent Python oracle for split_sm120_combined_tables.

    ``live`` is the raw combined row's live prefix. Compressed entries
    (``local < N`` within each request's stride ``M``) map to
    ``req * N + local``; SWA entries map to ``req * (M - N) + local - N``.
    """
    extra, swa = [], []
    for v in live:
        req, local = divmod(int(v), M)
        if local < N:
            extra.append(req * N + local)
        else:
            swa.append(req * (M - N) + local - N)
    extra_width = math.ceil(max(k, 1) / 64) * 64
    return (
        swa + [0] * (window - len(swa)),
        len(swa),
        extra + [0] * (extra_width - len(extra)),
        len(extra),
        64 if ratio == 4 else 2,
    )


def _normalize_oracle(host_table, host_lens, supported):
    """Independent canonical_topk + clamp_min_(0) oracle over host lists.

    Never calls the production normalization; expected values are built only
    from the sliced host rows.
    """
    e_indices, e_lens = [], []
    for row, length in zip(host_table, host_lens):
        width = len(row)
        padded = next(w for w in supported if w >= width)
        limit = min(max(int(length), 0), padded)
        live = [v for j, v in enumerate(row) if j < limit and v >= 0]
        e_indices.append(live + [0] * (padded - len(live)))
        e_lens.append(len(live))
    return e_indices, e_lens


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
        # Independent cold-prefill visibility oracle, before normalization.
        host_indices = combined_i.cpu().view(num_tokens, -1)
        host_lengths = combined_l.cpu().tolist()
        host_topk = topk_indices.cpu()
        for batch, length in enumerate(seq_lens):
            for pos in range(length):
                row = qsl[batch] + pos
                compressed = min((pos + 1) // ratio, topk)
                expected = [int(v) + batch * M for v in host_topk[row, :compressed]]
                expected += [
                    batch * M + N + p for p in range(max(0, pos - window + 1), pos + 1)
                ]
                self.assertEqual(host_lengths[row], len(expected))
                self.assertEqual(host_indices[row, : len(expected)].tolist(), expected)
        return combined_i, combined_l, M, N

    def _produce_cp(self, seq_full, cp_size, rank, topk, ratio, window):
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
        # Independent per-row visibility oracle on the raw combined output.
        host_indices = combined_i.cpu()
        host_lengths = combined_l.cpu().tolist()
        host_topk = topk_indices.cpu()
        host_gp = gp.cpu().tolist()
        for row, pos in enumerate(host_gp):
            compressed = min((pos + 1) // ratio, topk)
            expected = [int(v) for v in host_topk[row, :compressed]]
            expected += [N + p for p in range(max(0, pos - window + 1), pos + 1)]
            self.assertEqual(host_lengths[row], len(expected))
            self.assertEqual(host_indices[row, : len(expected)].tolist(), expected)
        return combined_i, combined_l, M, N

    def _produce_cp_mixed(self, seq_lens, prefixes, cp_size, rank, topk, ratio, window):
        """CP4 producer over requests with different continuation prefixes."""
        local_positions = []
        req_ids = []
        for req, s in enumerate(seq_lens):
            rows = list(range(rank * s // 8, (rank + 1) * s // 8)) + list(
                range(s - (rank + 1) * s // 8, s - rank * s // 8)
            )
            local_positions += [prefixes[req] + row for row in rows]
            req_ids += [req] * len(rows)
        gp = torch.tensor(local_positions, dtype=torch.int64, device=self.device)
        req_i32 = torch.tensor(req_ids, dtype=torch.int32, device=self.device)
        prefix_i64 = torch.tensor(prefixes, dtype=torch.int64, device=self.device)
        N = max((p + s) // ratio for p, s in zip(prefixes, seq_lens))
        M = N + max(s + min(p, 15) for p, s in zip(prefixes, seq_lens))
        host_gp = gp.cpu().tolist()
        topk_rows = [
            [j % max(1, (pos + 1) // ratio) for j in range(topk)] for pos in host_gp
        ]
        topk_indices = torch.tensor(topk_rows, dtype=torch.int32, device=self.device)
        combined_i, combined_l = combine_topk_swa_indices_cp(
            topk_indices,
            gp,
            0,
            window,
            ratio,
            topk,
            M,
            N,
            req_id_per_token=req_i32,
            prefix_lengths=prefix_i64,
        )
        # Independent visibility oracle: compressed prefix + SWA window.
        host_indices = combined_i.cpu()
        host_lengths = combined_l.cpu().tolist()
        for row, (req, pos) in enumerate(zip(req_ids, host_gp)):
            p = prefixes[req]
            compressed = min((pos + 1) // ratio, topk)
            expected = [req * M + t for t in topk_rows[row][:compressed]]
            expected += [
                req * M + N + q - (p - min(p, window - 1))
                for q in range(max(0, pos - window + 1), pos + 1)
            ]
            self.assertEqual(host_lengths[row], len(expected))
            self.assertEqual(host_indices[row, : len(expected)].tolist(), expected)
        return combined_i, combined_l, M, N

    def _check_matrix_entry(
        self, combined_i, combined_l, M, N, window, ratio, topk_width
    ):
        rows = int(combined_i.shape[0])
        host_indices = combined_i.cpu().view(rows, -1)
        host_lengths = combined_l.cpu().tolist()
        # Raw combined contract BEFORE the split: live prefix, exact length,
        # and every tail entry is -1.
        for row in range(rows):
            length = host_lengths[row]
            self.assertTrue(
                bool((host_indices[row, length:] == -1).all()),
                f"row {row}: non-(-1) entry past length {length}",
            )
        swa_i, swa_l, extra_i, extra_l, extra_page = split_sm120_combined_tables(
            combined_i,
            combined_l,
            M=M,
            N=N,
            window_size=window,
            extra_width=max(topk_width, 1),
            ratio=ratio,
            device=self.device,
        )
        self.assertEqual(extra_page, 64 if ratio == 4 else 2)
        self.assertEqual(int(swa_i.shape[1]), window)
        self.assertEqual(int(extra_i.shape[1]), math.ceil(max(topk_width, 1) / 64) * 64)
        swa_i, swa_l = swa_i.cpu(), swa_l.cpu().tolist()
        extra_i, extra_l = extra_i.cpu(), extra_l.cpu().tolist()
        for row in range(rows):
            live = host_indices[row, : host_lengths[row]].tolist()
            e_swa, e_swa_l, e_extra, e_extra_l, e_page = split_oracle(
                live, M, N, window, ratio, topk_width
            )
            self.assertEqual(extra_page, e_page)
            self.assertEqual(swa_l[row], e_swa_l, f"swa_len row {row}")
            self.assertEqual(extra_l[row], e_extra_l, f"extra_len row {row}")
            self.assertEqual(swa_i[row].tolist(), e_swa, f"swa row {row}")
            self.assertEqual(extra_i[row].tolist(), e_extra, f"extra row {row}")

        # Consumer chain: slice the ACTUAL split tables at full/half/interior
        # boundaries, run the production canonical_topk + clamp_min_(0), and
        # compare against the independent list oracle over the sliced hosts.
        slices = {(0, rows)}
        if rows >= 4:
            slices |= {(0, rows // 2), (rows // 2, rows), (1, rows - 1)}
        for start, end in sorted(slices):
            for supported, table, lens in (
                (SWA_SUPPORTED, swa_i, swa_l),
                (SM120_EXTRA_TOPK_WIDTHS, extra_i, extra_l),
            ):
                if table is None:
                    continue
                got_i, got_l = canonical_topk(
                    table[start:end].to(self.device),
                    torch.tensor(
                        lens[start:end], dtype=torch.int32, device=self.device
                    ),
                    supported,
                )
                got_i = got_i.clamp_min_(0)
                e_i, e_l = _normalize_oracle(
                    table[start:end].tolist(), lens[start:end], supported
                )
                self.assertTrue(
                    torch.equal(
                        got_i,
                        torch.tensor(e_i, dtype=torch.int32, device=self.device),
                    ),
                    f"normalized indices mismatch {(start, end, supported)}",
                )
                self.assertTrue(
                    torch.equal(
                        got_l.cpu(),
                        torch.tensor(e_l, dtype=torch.int32),
                    ),
                    f"normalized lens mismatch {(start, end, supported)}",
                )

    def test_matrix_cp1_varlen(self):
        for topk, ratio in ((0, 1), (128, 4), (2048, 4)):
            with self.subTest(topk=topk, ratio=ratio):
                ci, cl, M, N = self._produce_cp0(
                    num_tokens=64,
                    topk=topk,
                    ratio=ratio,
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

    def test_matrix_pdfusion_mixed_prefix_requests(self):
        for ratio in (4, 128):
            for rank in (0, 1, 2, 3):
                with self.subTest(ratio=ratio, rank=rank):
                    ci, cl, M, N = self._produce_cp_mixed(
                        seq_lens=[64, 96],
                        prefixes=[0, 256],
                        cp_size=4,
                        rank=rank,
                        topk=16,
                        ratio=ratio,
                        window=16,
                    )
                    self._check_matrix_entry(ci, cl, M, N, 16, ratio, 16)

    def test_cp4_positions_above_int32(self):
        # Global positions are int64 above INT32 while output coordinates
        # remain int32-representable; every CP4 rank is exercised.
        prefix = 2**31 + 32
        for rank in (0, 1, 2, 3):
            with self.subTest(rank=rank):
                ci, cl, M, N = self._produce_cp_mixed(
                    seq_lens=[64],
                    prefixes=[prefix],
                    cp_size=4,
                    rank=rank,
                    topk=16,
                    ratio=128,
                    window=16,
                )
                self._check_matrix_entry(ci, cl, M, N, 16, 128, 16)

    def test_generic_split_exact_large_coordinates(self):
        # Exact divmod coordinates at request-stride extremes — page 0 and
        # the largest page whose combined entries still fit int32. Expected
        # values come from the Python divmod oracle only; no production
        # split or normalization computes an expected value.
        for m in (1, 2, 3, 7, 64, 257, 40960, 65521, 2**30 - 1, 2**31 - 1):
            n = m // 3
            extra_count = 8 if n else 0
            largest_page = (INT32_MAX - (m - 1)) // m
            ci = torch.full((2, 128), -1, dtype=torch.int32)
            for row, page in enumerate((0, largest_page)):
                if n:
                    ci[row, :8] = page * m + n - 1
                ci[row, extra_count : extra_count + 8] = page * m + m - 1
            cl = torch.tensor([extra_count + 8] * 2, dtype=torch.int32)
            swa_i, swa_l, extra_i, extra_l, page_size = split_sm120_combined_tables(
                ci.to(self.device),
                cl.to(self.device),
                M=m,
                N=n,
                window_size=8,
                extra_width=8,
                ratio=4,
                device=self.device,
            )
            for row, page in enumerate((0, largest_page)):
                live = ci[row, : extra_count + 8].tolist()
                e_swa, e_swa_l, e_extra, e_extra_l, e_page = split_oracle(
                    live, m, n, 8, 4, 8
                )
                self.assertEqual(page_size, e_page)
                self.assertEqual(int(swa_l[row]), e_swa_l, (m, page))
                self.assertEqual(int(extra_l[row]), e_extra_l, (m, page))
                self.assertEqual(swa_i[row].tolist(), e_swa, (m, page))
                self.assertEqual(extra_i[row].tolist(), e_extra, (m, page))

        # int64 request stride: M above INT32_MAX with combined values that
        # still divide exactly. extra = 1*N+7, SWA = 1*(M-N)+5.
        m, n = 2**31 + 17, 2**30
        ci = torch.tensor([[m + 7, m + n + 5]], dtype=torch.int64)
        cl = torch.tensor([2], dtype=torch.int32)
        swa_i, swa_l, extra_i, extra_l, _ = split_sm120_combined_tables(
            ci.to(self.device),
            cl.to(self.device),
            M=m,
            N=n,
            window_size=1,
            extra_width=1,
            ratio=4,
            device=self.device,
        )
        self.assertEqual(extra_i[0].tolist(), [1073741831] + [0] * 63)
        self.assertEqual(swa_i[0].tolist(), [1073741846])
        self.assertEqual(int(extra_l[0]), 1)
        self.assertEqual(int(swa_l[0]), 1)


def _producer_table(rows, width, lens, device, seed):
    g = torch.Generator().manual_seed(seed)
    table = torch.zeros((rows, width), dtype=torch.int32, device=device)
    for r in range(rows):
        n = int(lens[r])
        if n > 0:
            table[r, :n] = torch.randint(0, 4096, (n,), generator=g).to(
                device=device, dtype=torch.int32
            )
    return table


def _normalize_actual(indices, lens, supported):
    i, n = canonical_topk(indices, lens, supported)
    return i.clamp_min_(0), n


def _expected_rows(host_table, host_lens, supported):
    e_indices, e_lens = [], []
    for row, length in zip(host_table, host_lens):
        width = len(row)
        padded = next(w for w in supported if w >= width)
        limit = min(max(length, 0), padded)
        live = [v for j, v in enumerate(row) if j < limit and v >= 0]
        e_indices.append(live + [0] * (padded - len(live)))
        e_lens.append(len(live))
    return e_indices, e_lens


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ReadyChunkCudaTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0")
        torch.cuda.init()

    def check(self, rows, width, lens, supported, seed):
        table = _producer_table(rows, width, lens, self.device, seed)
        lens_t = torch.tensor(lens, dtype=torch.int32, device=self.device)
        e_i, e_l = _expected_rows(table.cpu().tolist(), lens, supported)
        got_i, got_l = _normalize_actual(table.clone(), lens_t.clone(), supported)
        self.assertTrue(
            torch.equal(got_i, torch.tensor(e_i, dtype=torch.int32, device=self.device))
        )
        self.assertTrue(
            torch.equal(got_l, torch.tensor(e_l, dtype=torch.int32, device=self.device))
        )
        self.assertEqual(got_i.device.type, "cuda")

    def test_widths_on_cuda(self):
        window = 128
        for width, supported in (
            (window, SWA_SUPPORTED),
            (64, SM120_EXTRA_TOPK_WIDTHS),
            (192, SM120_EXTRA_TOPK_WIDTHS),
            (256, SM120_EXTRA_TOPK_WIDTHS),
            (2048, SM120_EXTRA_TOPK_WIDTHS),
        ):
            for rows in (1, 1024):
                lens = [min(2 * r + 1, width) for r in range(rows)]
                self.check(rows, width, lens, supported, seed=rows + width)

        table = torch.tensor([[7, -1, 9, 99]], dtype=torch.int32, device=self.device)
        table_copy = table.clone()
        lens_t = torch.tensor([3], dtype=torch.int32, device=self.device)
        lens_copy = lens_t.clone()
        got_i, got_l = _normalize_actual(table, lens_t, SWA_SUPPORTED)
        self.assertEqual(got_i[0].tolist()[:3], [7, 9, 0])
        self.assertEqual(got_l.tolist(), [2])
        self.assertTrue(torch.equal(table, table_copy))
        self.assertTrue(torch.equal(lens_t, lens_copy))

        for length, e_live in ((-3, []), (10**6, [7, 9, 99])):
            got_i, got_l = _normalize_actual(
                table,
                torch.tensor([length], dtype=torch.int32, device=self.device),
                SWA_SUPPORTED,
            )
            self.assertEqual(got_i[0].tolist()[: len(e_live) + 1], e_live + [0])
            self.assertEqual(got_l.tolist(), [len(e_live)])

        with self.assertRaises(RuntimeError):
            _normalize_actual(
                torch.zeros((1, 8193), dtype=torch.int32, device=self.device),
                torch.ones(1, dtype=torch.int32, device=self.device),
                SWA_SUPPORTED,
            )

    def test_non_default_stream_and_events(self):
        rows, width = 256, 128
        lens = [min(r + 1, width) for r in range(rows)]
        stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(stream):
            table = _producer_table(rows, width, lens, self.device, 11)
            lens_t = torch.tensor(lens, dtype=torch.int32, device=self.device)
            got_i, _ = _normalize_actual(table, lens_t, SWA_SUPPORTED)
            ev = torch.cuda.Event()
            ev.record(stream)
        torch.cuda.current_stream(self.device).wait_event(ev)
        e_i, _ = _expected_rows(table.cpu().tolist(), lens, SWA_SUPPORTED)
        self.assertTrue(
            torch.equal(got_i, torch.tensor(e_i, dtype=torch.int32, device=self.device))
        )

    def test_two_forwards_same_shape_different_contents(self):
        rows, width = 1024, 128
        lens = [width] * rows
        for seed in (101, 202):
            table = _producer_table(rows, width, lens, self.device, seed)
            lens_t = torch.tensor(lens, dtype=torch.int32, device=self.device)
            e_i, _ = _expected_rows(table.cpu().tolist(), lens, SWA_SUPPORTED)
            got_i, _ = _normalize_actual(table.clone(), lens_t.clone(), SWA_SUPPORTED)
            self.assertTrue(
                torch.equal(
                    got_i, torch.tensor(e_i, dtype=torch.int32, device=self.device)
                )
            )
            del table, lens_t, got_i, e_i

    def test_inference_mode(self):
        rows, width = 64, 128
        lens = [min(r + 3, width) for r in range(rows)]
        with torch.inference_mode():
            table = _producer_table(rows, width, lens, self.device, 5)
            lens_t = torch.tensor(lens, dtype=torch.int32, device=self.device)
            got_i, _ = _normalize_actual(table, lens_t, SWA_SUPPORTED)
        e_i, _ = _expected_rows(table.cpu().tolist(), lens, SWA_SUPPORTED)
        self.assertTrue(
            torch.equal(got_i, torch.tensor(e_i, dtype=torch.int32, device=self.device))
        )

    def test_graph_replay_preserves_input_lengths(self):
        window = 128
        table = torch.arange(window, dtype=torch.int32, device=self.device).repeat(2, 1)
        lengths = torch.tensor([-1, window + 5], dtype=torch.int32, device=self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            _normalize_actual(table, lengths, SWA_SUPPORTED)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            norm_i, norm_l = _normalize_actual(table, lengths, SWA_SUPPORTED)
        for values, e_counts in (([-1, window + 5], [0, 128]), ([17, -3], [17, 0])):
            lengths.copy_(torch.tensor(values, dtype=torch.int32, device=self.device))
            graph.replay()
            self.assertEqual(lengths.tolist(), values)
            self.assertEqual(norm_l.tolist(), e_counts)
            for row, count in enumerate(e_counts):
                live = norm_i[row].tolist()
                self.assertEqual(live[:count], list(range(window))[:count])
                self.assertTrue(all(v == 0 for v in live[count:]))


if __name__ == "__main__":
    unittest.main()
