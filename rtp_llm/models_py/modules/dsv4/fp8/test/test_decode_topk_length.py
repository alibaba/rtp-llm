"""CPU tests for opt_flash_mla decode FlashMLA effective-length plumbing.

Covers the ``topk_length`` / ``extra_topk_length`` design that lets the DSv4 FP8
sparse decode kernel scan only the real ``seq_len``-derived width instead of the
CUDA-graph capture width (e.g. HCA 8192 -> ``seq_len // 128``):

  * value/clamp correctness (SWA window, CSA->index_topk cap, HCA dense),
  * MTP q_len>1 last-token semantics,
  * graph-stability (in-place ``.copy_`` keeps buffer addresses stable),
  * SWA-only / warmup None-safety,
  * eager (build) vs CUDA-graph (allocate+update helper) numerical parity,
  * the dispatch helpers actually forward the lengths to flash_mla.

All CPU-only: ``build_decode_metadata_fp8`` (no paged tables) and
``_update_topk_lengths_in_place`` are pure-torch; the FlashMLA call is faked.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
    _update_topk_lengths_in_place,
    allocate_decode_metadata_fp8,
    build_decode_metadata_fp8,
)

WINDOW = 128
HEAD_DIM = 512
INDEX_TOPK = 1024
RATIOS = [0, 4, 128]


def _hca_dense_width(max_seq_len: int) -> int:
    return ((max_seq_len // 128) + 63) // 64 * 64


class DecodeTopkLengthEagerTest(unittest.TestCase):
    """build_decode_metadata_fp8 (eager path) length values."""

    def _build(self, start_pos, q_len, max_seq_len):
        sp = torch.tensor(start_pos, dtype=torch.int32)
        return build_decode_metadata_fp8(
            sp,
            q_len=q_len,
            window_size=WINDOW,
            head_dim=HEAD_DIM,
            max_seq_len=max_seq_len,
            compress_ratios=RATIOS,
            index_topk=INDEX_TOPK,
            device=torch.device("cpu"),
        )

    def test_long_context_q_len_1(self) -> None:
        # 16K / 128K / 1M current-token positions (q_len=1).
        cases = (
            # (start_pos, max_seq_len, hca_len, swa_len)
            (16 * 1024 - 1, 1024 * 1024, 128, WINDOW),
            (128 * 1024 - 1, 1024 * 1024, 1024, WINDOW),
            (64 * 1024 - 1, 1024 * 1024, 512, WINDOW),
        )
        for start_pos, max_seq_len, hca_len, swa_len in cases:
            with self.subTest(start_pos=start_pos):
                meta = self._build([start_pos], q_len=1, max_seq_len=max_seq_len)
                self.assertEqual(int(meta.swa_topk_length[0]), swa_len)
                # HCA effective length == seq_len // 128, well under capture width.
                self.assertEqual(int(meta.compressed_topk_length_by_ratio[128][0]), hca_len)
                self.assertLess(hca_len, _hca_dense_width(max_seq_len))
                # CSA capped at index_topk for long context.
                self.assertEqual(
                    int(meta.compressed_topk_length_by_ratio[4][0]), INDEX_TOPK
                )

    def test_short_context_csa_below_cap_and_swa_partial(self) -> None:
        # seq=404 tokens (start_pos=403, q_len=1): CSA len = 404//4 = 101 < cap,
        # HCA len = 404//128 = 3, SWA len = min(128, 404) = 128.
        meta = self._build([403], q_len=1, max_seq_len=1024 * 1024)
        self.assertEqual(int(meta.compressed_topk_length_by_ratio[4][0]), 101)
        self.assertEqual(int(meta.compressed_topk_length_by_ratio[128][0]), 3)
        self.assertEqual(int(meta.swa_topk_length[0]), WINDOW)

    def test_early_decode_swa_partial_window(self) -> None:
        # position 50 (q_len=1): SWA len = min(128, 51) = 51 (partial window).
        meta = self._build([50], q_len=1, max_seq_len=4096)
        self.assertEqual(int(meta.swa_topk_length[0]), 51)
        self.assertEqual(int(meta.compressed_topk_length_by_ratio[4][0]), 51 // 4)
        self.assertEqual(int(meta.compressed_topk_length_by_ratio[128][0]), 0)

    def test_mtp_q_len_4_uses_last_token(self) -> None:
        # q_len=4, first-token pos 16380 -> last token pos 16383.
        # SWA len = min(128, 16380+4) = 128; HCA len = (16383+1)//128 = 128;
        # CSA len = min(1024, 16384//4) = 1024.
        meta = self._build([16380], q_len=4, max_seq_len=1024 * 1024)
        self.assertEqual(int(meta.swa_topk_length[0]), WINDOW)
        self.assertEqual(int(meta.compressed_topk_length_by_ratio[128][0]), 128)
        self.assertEqual(int(meta.compressed_topk_length_by_ratio[4][0]), INDEX_TOPK)

    def test_per_request_lengths_independent(self) -> None:
        # batch with mixed positions -> per-request lengths differ.
        meta = self._build([16 * 1024 - 1, 403, 50], q_len=1, max_seq_len=1024 * 1024)
        self.assertEqual(
            [int(x) for x in meta.compressed_topk_length_by_ratio[128]],
            [128, 3, 0],
        )
        self.assertEqual(
            [int(x) for x in meta.swa_topk_length],
            [WINDOW, WINDOW, 51],
        )


class DecodeTopkLengthInPlaceTest(unittest.TestCase):
    """allocate + _update_topk_lengths_in_place (CUDA-graph path emulation)."""

    def _alloc(self, max_batch, q_len, max_seq_len, ratios=RATIOS):
        return allocate_decode_metadata_fp8(
            max_batch_size=max_batch,
            q_len=q_len,
            window_size=WINDOW,
            head_dim=HEAD_DIM,
            max_seq_len=max_seq_len,
            compress_ratios=ratios,
            index_topk=INDEX_TOPK,
            device=torch.device("cpu"),
        )

    def test_buffers_allocated(self) -> None:
        meta = self._alloc(16, 4, 1024 * 1024)
        self.assertIsNotNone(meta.swa_topk_length)
        self.assertEqual(tuple(meta.swa_topk_length.shape), (16,))
        self.assertEqual(meta.swa_topk_length.dtype, torch.int32)
        self.assertIn(4, meta.compressed_topk_length_by_ratio)
        self.assertIn(128, meta.compressed_topk_length_by_ratio)
        self.assertEqual(tuple(meta.compressed_topk_length_by_ratio[128].shape), (16,))

    def test_values_match_formula(self) -> None:
        meta = self._alloc(4, 1, 1024 * 1024)
        bs = 2
        start_pos = torch.tensor([16 * 1024 - 1, 403], dtype=torch.int32)
        # Emulate fused_update_decode_meta_pure filling compressed_lens.
        meta.compressed_lens[4][:bs].copy_((start_pos + 1) // 4)
        meta.compressed_lens[128][:bs].copy_((start_pos + 1) // 128)
        _update_topk_lengths_in_place(meta, start_pos, bs)
        self.assertEqual([int(x) for x in meta.swa_topk_length[:bs]], [WINDOW, WINDOW])
        # HCA = compressed_lens (no clamp): 128, 3
        self.assertEqual(
            [int(x) for x in meta.compressed_topk_length_by_ratio[128][:bs]], [128, 3]
        )
        # CSA = min(index_topk, compressed_lens[4]): 4096->1024, 101->101
        self.assertEqual(
            [int(x) for x in meta.compressed_topk_length_by_ratio[4][:bs]],
            [INDEX_TOPK, 101],
        )

    def test_in_place_no_realloc(self) -> None:
        # The redteam blocker: in-place .copy_ must keep buffer addresses stable
        # so a captured CUDA graph keeps reading from the same storage.
        meta = self._alloc(4, 1, 1024 * 1024)
        ptr_swa = meta.swa_topk_length.data_ptr()
        ptr_cmp = {r: t.data_ptr() for r, t in meta.compressed_topk_length_by_ratio.items()}
        for sp in ([16 * 1024 - 1, 403], [128 * 1024 - 1, 50]):
            bs = 2
            start_pos = torch.tensor(sp, dtype=torch.int32)
            meta.compressed_lens[4][:bs].copy_((start_pos + 1) // 4)
            meta.compressed_lens[128][:bs].copy_((start_pos + 1) // 128)
            _update_topk_lengths_in_place(meta, start_pos, bs)
        self.assertEqual(meta.swa_topk_length.data_ptr(), ptr_swa)
        for r, t in meta.compressed_topk_length_by_ratio.items():
            self.assertEqual(t.data_ptr(), ptr_cmp[r])

    def test_swa_only_no_compressed_ratios(self) -> None:
        # SWA-only model (compress_ratios=[0]) / warmup: no compressed length
        # buffers, helper must not crash and must still fill swa_topk_length.
        meta = self._alloc(4, 1, 4096, ratios=[0])
        self.assertEqual(dict(meta.compressed_topk_length_by_ratio), {})
        bs = 2
        start_pos = torch.tensor([300, 50], dtype=torch.int32)
        _update_topk_lengths_in_place(meta, start_pos, bs)
        self.assertEqual([int(x) for x in meta.swa_topk_length[:bs]], [WINDOW, 51])


class DecodeTopkLengthCaptureFullWidthTest(unittest.TestCase):
    """full_width=True (CUDA-graph capture/warmup) max-provisions the lengths.

    Regression for the smoke crash: FlashMLA freezes its sparse tile schedule on
    the warmup forward keyed on the topk_length VALUES, then reuses it for every
    replay (sparse_decode.h:420, cuda_graph_runner.cc:880). Capturing with the
    full width makes the frozen schedule a valid superset for any real replay
    length; real (smaller) lengths are written before each replay.
    """

    def _alloc(self, max_batch, q_len, max_seq_len, ratios=RATIOS):
        return allocate_decode_metadata_fp8(
            max_batch_size=max_batch,
            q_len=q_len,
            window_size=WINDOW,
            head_dim=HEAD_DIM,
            max_seq_len=max_seq_len,
            compress_ratios=ratios,
            index_topk=INDEX_TOPK,
            device=torch.device("cpu"),
        )

    def test_full_width_ignores_start_pos(self) -> None:
        max_seq_len = 1024 * 1024
        meta = self._alloc(4, 1, max_seq_len)
        bs = 3
        # Small start positions that would give SHORT real lengths.
        start_pos = torch.tensor([7, 50, 403], dtype=torch.int32)
        meta.compressed_lens[4][:bs].copy_((start_pos + 1) // 4)
        meta.compressed_lens[128][:bs].copy_((start_pos + 1) // 128)
        _update_topk_lengths_in_place(meta, start_pos, bs, full_width=True)
        # SWA -> full window for every request, regardless of start_pos.
        self.assertEqual([int(x) for x in meta.swa_topk_length[:bs]], [WINDOW] * bs)
        # Compressed -> the full extra-index width (== indices tensor width),
        # i.e. the same scan width the topk_length=None baseline implies.
        for r in (4, 128):
            width_r = meta.topk_total_by_ratio[r].shape[-1] - WINDOW
            self.assertEqual(
                [int(x) for x in meta.compressed_topk_length_by_ratio[r][:bs]],
                [width_r] * bs,
            )

    def test_full_width_is_superset_of_real(self) -> None:
        # The frozen (full_width) lengths must be >= the real lengths for every
        # request and ratio, so the schedule built at capture never under-runs.
        max_seq_len = 1024 * 1024
        bs = 4
        start_pos = torch.tensor(
            [128 * 1024 - 1, 16 * 1024 - 1, 403, 7], dtype=torch.int32
        )
        cap = self._alloc(bs, 1, max_seq_len)
        real = self._alloc(bs, 1, max_seq_len)
        for m in (cap, real):
            m.compressed_lens[4][:bs].copy_((start_pos + 1) // 4)
            m.compressed_lens[128][:bs].copy_((start_pos + 1) // 128)
        _update_topk_lengths_in_place(cap, start_pos, bs, full_width=True)
        _update_topk_lengths_in_place(real, start_pos, bs, full_width=False)
        self.assertTrue(bool((cap.swa_topk_length >= real.swa_topk_length).all()))
        for r in (4, 128):
            self.assertTrue(
                bool(
                    (
                        cap.compressed_topk_length_by_ratio[r]
                        >= real.compressed_topk_length_by_ratio[r]
                    ).all()
                )
            )

    def test_full_width_no_realloc(self) -> None:
        meta = self._alloc(4, 1, 1024 * 1024)
        ptr_swa = meta.swa_topk_length.data_ptr()
        ptr_cmp = {
            r: t.data_ptr() for r, t in meta.compressed_topk_length_by_ratio.items()
        }
        start_pos = torch.tensor([7, 50], dtype=torch.int32)
        _update_topk_lengths_in_place(meta, start_pos, 2, full_width=True)
        self.assertEqual(meta.swa_topk_length.data_ptr(), ptr_swa)
        for r, t in meta.compressed_topk_length_by_ratio.items():
            self.assertEqual(t.data_ptr(), ptr_cmp[r])

    def test_full_width_swa_only(self) -> None:
        meta = self._alloc(4, 1, 4096, ratios=[0])
        start_pos = torch.tensor([300, 50], dtype=torch.int32)
        _update_topk_lengths_in_place(meta, start_pos, 2, full_width=True)
        self.assertEqual([int(x) for x in meta.swa_topk_length[:2]], [WINDOW, WINDOW])


class DecodeTopkLengthParityTest(unittest.TestCase):
    """Eager build vs allocate+update produce identical length values."""

    def test_eager_graph_parity(self) -> None:
        max_seq_len = 1024 * 1024
        q_len = 4
        start_pos = torch.tensor([128 * 1024 - 4, 16 * 1024 - 4, 200, 7], dtype=torch.int32)
        bs = int(start_pos.shape[0])

        eager = build_decode_metadata_fp8(
            start_pos,
            q_len=q_len,
            window_size=WINDOW,
            head_dim=HEAD_DIM,
            max_seq_len=max_seq_len,
            compress_ratios=RATIOS,
            index_topk=INDEX_TOPK,
            device=torch.device("cpu"),
        )

        graph = allocate_decode_metadata_fp8(
            max_batch_size=bs,
            q_len=q_len,
            window_size=WINDOW,
            head_dim=HEAD_DIM,
            max_seq_len=max_seq_len,
            compress_ratios=RATIOS,
            index_topk=INDEX_TOPK,
            device=torch.device("cpu"),
        )
        for r in (4, 128):
            graph.compressed_lens[r][:bs].copy_((start_pos + q_len) // r)
        _update_topk_lengths_in_place(graph, start_pos, bs)

        self.assertTrue(
            torch.equal(eager.swa_topk_length, graph.swa_topk_length[:bs])
        )
        for r in (4, 128):
            self.assertTrue(
                torch.equal(
                    eager.compressed_topk_length_by_ratio[r],
                    graph.compressed_topk_length_by_ratio[r][:bs],
                ),
                f"ratio {r} mismatch: eager={eager.compressed_topk_length_by_ratio[r]} "
                f"graph={graph.compressed_topk_length_by_ratio[r][:bs]}",
            )


class DecodeTopkLengthPlumbingTest(unittest.TestCase):
    """The dispatch helpers forward lengths to flash_mla_with_kvcache."""

    def _with_fake_flash_mla(self, fn):
        calls = []
        fake = types.ModuleType("flash_mla")
        fake.get_mla_metadata = lambda *a, **k: (object(), None)

        def fake_kv(**kwargs):
            calls.append(
                {
                    "topk_length": kwargs.get("topk_length"),
                    "extra_topk_length": kwargs.get("extra_topk_length"),
                }
            )
            q = kwargs["q"]
            hd = int(kwargs["head_dim_v"])
            out = torch.zeros(q.shape[0], q.shape[1], q.shape[2], hd, dtype=q.dtype)
            lse = torch.zeros(q.shape[0], q.shape[2], q.shape[1])
            return out, lse

        fake.flash_mla_with_kvcache = fake_kv
        old = sys.modules.get("flash_mla")
        sys.modules["flash_mla"] = fake
        try:
            module = importlib.import_module(
                "rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op"
            )
            module._FLASH_MLA_AVAILABLE = True
            op = module.SparseAttnV4DecodeFp8Op(
                n_heads=2, head_dim=HEAD_DIM, softmax_scale=HEAD_DIM**-0.5
            )
            fn(op, calls)
        finally:
            if old is None:
                sys.modules.pop("flash_mla", None)
            else:
                sys.modules["flash_mla"] = old
        return calls

    def test_dual_paged_forwards_both_lengths(self) -> None:
        from rtp_llm.models_py.modules.dsv4.fp8.decode.attention_kernels import (
            attn_fp8_dual_paged,
        )

        B, q_len = 3, 1
        q = torch.zeros(B, q_len, 2, HEAD_DIM, dtype=torch.bfloat16)
        swa_pool = torch.zeros(2, 128, 584, dtype=torch.uint8)
        cmp_pool = torch.zeros(2, 64, 584, dtype=torch.uint8)
        swa_idx = torch.full((B, q_len, WINDOW), -1, dtype=torch.int32)
        cmp_idx = torch.full((B, q_len, 8192), -1, dtype=torch.int32)
        topk_len = torch.tensor([128, 128, 51], dtype=torch.int32)
        extra_len = torch.tensor([128, 3, 0], dtype=torch.int32)

        calls = self._with_fake_flash_mla(
            lambda op, _calls: attn_fp8_dual_paged(
                q=q,
                swa_pool_3d=swa_pool,
                cmp_pool_3d=cmp_pool,
                attn_sink=torch.zeros(2, dtype=torch.float32),
                swa_topk_3d=swa_idx,
                cmp_topk_3d=cmp_idx,
                swa_block_table=torch.zeros(B, 4, dtype=torch.int32),
                sched_meta=object(),
                fp8_op=op,
                topk_length=topk_len,
                extra_topk_length=extra_len,
            )
        )
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(calls[0]["topk_length"], topk_len))
        self.assertTrue(torch.equal(calls[0]["extra_topk_length"], extra_len))

    def test_swa_paged_forwards_topk_length(self) -> None:
        from rtp_llm.models_py.modules.dsv4.fp8.decode.attention_kernels import (
            attn_fp8_swa_paged,
        )

        B, q_len = 2, 1
        q = torch.zeros(B, q_len, 2, HEAD_DIM, dtype=torch.bfloat16)
        swa_pool = torch.zeros(2, 128, 584, dtype=torch.uint8)
        swa_idx = torch.full((B, q_len, WINDOW), -1, dtype=torch.int32)
        topk_len = torch.tensor([128, 51], dtype=torch.int32)

        calls = self._with_fake_flash_mla(
            lambda op, _calls: attn_fp8_swa_paged(
                q=q,
                swa_pool_3d=swa_pool,
                attn_sink=torch.zeros(2, dtype=torch.float32),
                swa_topk_3d=swa_idx,
                swa_block_table=torch.zeros(B, 4, dtype=torch.int32),
                sched_meta=object(),
                fp8_op=op,
                topk_length=topk_len,
            )
        )
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(calls[0]["topk_length"], topk_len))
        self.assertIsNone(calls[0]["extra_topk_length"])


if __name__ == "__main__":
    unittest.main()
