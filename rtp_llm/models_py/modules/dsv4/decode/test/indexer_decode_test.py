"""Unit tests for ``IndexerDecodeV4Op``.

Validates the reference path bit-exactly against a pure-Python einsum
oracle and (when ``deep_gemm.fp8_paged_mqa_logits`` is available) checks
the FP8 fast path's topk against the reference via topk-IoU.

Note: the fast-path FP8 paged layout integration is currently scaffolded
but not numerically verified end-to-end; its IoU test is marked
``@unittest.skip`` until the model integration site lands. See the
top-of-file comment in ``indexer_decode_op.py`` for context.
"""

import os
import sys
import types
import unittest
from unittest.mock import patch

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.models_py.modules.dsv4.decode.indexer_decode_op import (
    IndexerDecodeV4Op,
    _fast_path_available,
)


def _oracle_topk(
    q: torch.Tensor,  # [B, q_len, H, D] bf16
    kv: torch.Tensor,  # [B, T_max, D]   bf16
    weights: torch.Tensor,  # [B, q_len, H]   bf16 (already scaled)
    cmp_lens: torch.Tensor,  # [B] int32
    K: int,
) -> torch.Tensor:
    """Pure-Python einsum-then-topk oracle. Returns int32 [B, q_len, K]
    with -1 padding for invalid positions."""
    B, q_len, H, D = q.shape
    T_max = kv.shape[1]
    device = q.device
    score = torch.einsum("bshd,btd->bsht", q.float(), kv.float())
    score = (score.relu_() * weights.float().unsqueeze(-1)).sum(
        dim=2
    )  # [B, q_len, T_max]
    t_range = torch.arange(T_max, device=device, dtype=torch.int32)
    valid = t_range.view(1, 1, T_max) < cmp_lens.view(B, 1, 1).to(torch.int32)
    score = torch.where(valid, score, torch.full_like(score, float("-inf")))
    k_eff = min(K, T_max)
    idxs = score.topk(k_eff, dim=-1).indices.to(torch.int32)
    if k_eff < K:
        pad = torch.full((B, q_len, K - k_eff), -1, dtype=torch.int32, device=device)
        idxs = torch.cat([idxs, pad], dim=-1)
    cmp_len_b = cmp_lens.view(B, 1, 1).to(torch.int32)
    idxs = torch.where(idxs < cmp_len_b, idxs, torch.full_like(idxs, -1))
    return idxs


def _topk_iou(
    a: torch.Tensor,  # [B, q_len, K] int32 (-1 = invalid)
    b: torch.Tensor,  # same shape
) -> float:
    """Per-(B, q_len) row IoU between two topk index sets, averaged."""
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    B, S, K = a.shape
    a_list = a.view(-1, K).cpu().tolist()
    b_list = b.view(-1, K).cpu().tolist()
    ious = []
    for ai, bi in zip(a_list, b_list):
        sa = {x for x in ai if x >= 0}
        sb = {x for x in bi if x >= 0}
        if not sa and not sb:
            ious.append(1.0)
            continue
        inter = len(sa & sb)
        union = len(sa | sb)
        ious.append(inter / union if union else 1.0)
    return sum(ious) / max(len(ious), 1)


class TestIndexerDecodeOp(unittest.TestCase):
    # Common shapes — V4 uses H_idx=64, D_idx=128, K=index_topk (here 32 to
    # exercise both topk dim and length-masking edge cases without making
    # the oracle slow).
    B = 4
    Q_LEN = 1
    H_IDX = 64
    D_IDX = 128
    T_COMPRESSED = 64
    K = 32

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("Indexer decode op is CUDA-only")
        torch.manual_seed(0)

    # ----------------------------------------------------------------
    # 1) Reference path == pure-Python oracle (exact match expected)
    # ----------------------------------------------------------------
    def test_reference_path_matches_indexer_einsum(self):
        device = torch.device("cuda:0")
        B, Q, H, D = self.B, self.Q_LEN, self.H_IDX, self.D_IDX
        T = self.T_COMPRESSED

        q = (torch.randn(B, Q, H, D, device=device) * 0.1).to(torch.bfloat16)
        kv = (torch.randn(B, T, D, device=device) * 0.1).to(torch.bfloat16)
        weights = (torch.randn(B, Q, H, device=device) * 0.5).to(torch.bfloat16)
        # All requests use the full T as their valid compressed length.
        cmp_lens = torch.full((B,), T, dtype=torch.int32, device=device)

        op = IndexerDecodeV4Op(
            index_n_heads=H,
            index_head_dim=D,
            index_topk=self.K,
            softmax_scale=1.0,
        )
        out_buffer = torch.full((B, Q, self.K), -1, dtype=torch.int32, device=device)
        result = op.forward(
            q,
            kv,
            weights,
            cmp_lens,
            out_buffer,
            force_reference=True,
        )
        self.assertIs(result, out_buffer)

        oracle = _oracle_topk(q, kv, weights, cmp_lens, self.K)

        # topk indices may differ in ORDER (topk is unsorted). Compare as sets per row.
        # For the reference path we expect identical set membership.
        for b in range(B):
            for s in range(Q):
                got = sorted(int(x) for x in result[b, s].tolist() if int(x) >= 0)
                want = sorted(int(x) for x in oracle[b, s].tolist() if int(x) >= 0)
                self.assertEqual(got, want, f"mismatch at b={b}, s={s}")

    # ----------------------------------------------------------------
    # 2) Fast path topk-IoU vs reference  -- WIP: paged FP8 layout
    # ----------------------------------------------------------------
    @unittest.skip(
        "fast-path WIP: paged FP8 (data || scale) packed layout "
        "needs end-to-end validation against deep_gemm. Reference "
        "path is the source of truth for now; integration into "
        "deepseek_v4_model.py will toggle the env flag once the "
        "fast-path layout is verified."
    )
    @unittest.skipUnless(torch.cuda.is_available(), "no cuda")
    @unittest.skipUnless(_fast_path_available(), "no deep_gemm fp8_paged_mqa_logits")
    def test_fast_path_topk_iou(self):
        device = torch.device("cuda:0")
        B, Q, H, D = self.B, self.Q_LEN, self.H_IDX, self.D_IDX
        T = self.T_COMPRESSED

        q = (torch.randn(B, Q, H, D, device=device) * 0.1).to(torch.bfloat16)
        kv = (torch.randn(B, T, D, device=device) * 0.1).to(torch.bfloat16)
        weights = (torch.randn(B, Q, H, device=device) * 0.5).to(torch.bfloat16)
        cmp_lens = torch.full((B,), T, dtype=torch.int32, device=device)

        op = IndexerDecodeV4Op(
            index_n_heads=H,
            index_head_dim=D,
            index_topk=self.K,
            softmax_scale=1.0,
        )
        out_ref = torch.full((B, Q, self.K), -1, dtype=torch.int32, device=device)
        op.forward(q, kv, weights, cmp_lens, out_ref, force_reference=True)

        out_fast = torch.full((B, Q, self.K), -1, dtype=torch.int32, device=device)
        op.forward(q, kv, weights, cmp_lens, out_fast, force_reference=False)

        iou = _topk_iou(out_ref, out_fast)
        self.assertGreaterEqual(
            iou,
            0.95,
            f"fast-path topk-IoU={iou:.3f} below threshold 0.95",
        )

    # ----------------------------------------------------------------
    # 3) compressed_len_per_req < T_max masks invalid range
    # ----------------------------------------------------------------
    def test_short_compressed_len(self):
        device = torch.device("cuda:0")
        B, Q, H, D = self.B, self.Q_LEN, self.H_IDX, self.D_IDX
        T = self.T_COMPRESSED  # 64
        valid_len = 10

        q = (torch.randn(B, Q, H, D, device=device) * 0.1).to(torch.bfloat16)
        kv = (torch.randn(B, T, D, device=device) * 0.1).to(torch.bfloat16)
        weights = (torch.randn(B, Q, H, device=device) * 0.5).to(torch.bfloat16)
        cmp_lens = torch.full((B,), valid_len, dtype=torch.int32, device=device)

        op = IndexerDecodeV4Op(
            index_n_heads=H,
            index_head_dim=D,
            index_topk=self.K,
            softmax_scale=1.0,
        )
        out_buffer = torch.full((B, Q, self.K), -1, dtype=torch.int32, device=device)
        op.forward(q, kv, weights, cmp_lens, out_buffer, force_reference=True)

        # Every non-(-1) entry must be < valid_len. Slots beyond valid_len
        # should be -1 (since K=32 > valid_len=10, the last 22 entries per row
        # must be -1).
        for b in range(B):
            for s in range(Q):
                row = out_buffer[b, s].tolist()
                non_neg = [x for x in row if x >= 0]
                self.assertEqual(
                    len(non_neg),
                    valid_len,
                    f"row b={b},s={s} has {len(non_neg)} valid entries; expected {valid_len}",
                )
                for x in non_neg:
                    self.assertLess(
                        x,
                        valid_len,
                        f"row b={b},s={s} has out-of-range index {x} >= {valid_len}",
                    )
                # Set membership = {0..valid_len-1}.
                self.assertEqual(
                    sorted(non_neg),
                    list(range(valid_len)),
                    f"row b={b},s={s} valid indices != [0..{valid_len-1}]",
                )

    # ----------------------------------------------------------------
    # 4) shape & dtype sanity
    # ----------------------------------------------------------------
    def test_returns_same_buffer_with_correct_shape_dtype(self):
        device = torch.device("cuda:0")
        B, Q, H, D = 2, 1, 4, 128
        T = 16
        K = 8
        q = torch.randn(B, Q, H, D, device=device).to(torch.bfloat16)
        kv = torch.randn(B, T, D, device=device).to(torch.bfloat16)
        weights = torch.randn(B, Q, H, device=device).to(torch.bfloat16)
        cmp_lens = torch.tensor([T, T // 2], dtype=torch.int32, device=device)
        op = IndexerDecodeV4Op(
            index_n_heads=H,
            index_head_dim=D,
            index_topk=K,
            softmax_scale=1.0,
        )
        out_buffer = torch.full((B, Q, K), -1, dtype=torch.int32, device=device)
        ret = op.forward(q, kv, weights, cmp_lens, out_buffer, force_reference=True)
        self.assertIs(ret, out_buffer)
        self.assertEqual(ret.shape, (B, Q, K))
        self.assertEqual(ret.dtype, torch.int32)


class TestIndexerDecodeFastPathAbi(unittest.TestCase):
    def test_fast_path_passes_2d_lengths_and_striped_block_layout(self):
        from rtp_llm.models_py.modules.dsv4.decode import indexer_decode_op as dsv4_mod
        from rtp_llm.models_py.modules.dsv41.decode import (
            indexer_decode_op as dsv41_mod,
        )

        def fake_unpack_ue8m0_scale(sf_packed: torch.Tensor) -> torch.Tensor:
            return (((sf_packed & 0xFF).to(torch.int32)) << 23).view(torch.float32)

        fake_indexer_op = types.ModuleType(
            "rtp_llm.models_py.modules.base.cuda.indexer_op"
        )
        fake_indexer_op._unpack_ue8m0_scale = fake_unpack_ue8m0_scale

        B, H, D, T, K = 2, 4, 128, 8, 4
        block_kv = 4

        for mod in (dsv4_mod, dsv41_mod):
            for q_len in (1, 2, 3):
                for scale_fmt in ("ue8m0", "fp32"):
                    with self.subTest(
                        module=mod.__name__, q_len=q_len, scale_fmt=scale_fmt
                    ):
                        recorded = {}

                        def fake_quant(
                            x,
                            *,
                            group_size,
                            eps,
                            column_major_scales,
                            scale_tma_aligned,
                            scale_ue8m0,
                        ):
                            if scale_ue8m0:
                                self.assertTrue(
                                    column_major_scales and scale_tma_aligned
                                )
                            rows = x.shape[0]
                            token_vals = (
                                (torch.arange(rows, dtype=torch.int64) % 9 + 1)
                                .view(rows, 1)
                                .expand(rows, D)
                                .to(torch.float8_e4m3fn)
                            )
                            exponents = 127 + (
                                torch.arange(rows, dtype=torch.int32) % 4
                            )
                            if scale_ue8m0:
                                aligned_mn = (rows + 3) // 4 * 4
                                packed = torch.zeros((1, aligned_mn), dtype=torch.int32)
                                packed[0, :rows] = exponents
                                return token_vals, packed.transpose(-1, -2)[:rows, :]
                            return token_vals, torch.exp2(
                                (exponents - 127).float()
                            ).view(rows, 1)

                        fake_fp8_kernel = types.ModuleType(
                            "rtp_llm.models_py.kernels.cuda.fp8_kernel"
                        )
                        fake_fp8_kernel.sgl_per_token_group_quant_fp8 = fake_quant

                        def fake_meta(context_lens, blk_sz, num_sms):
                            recorded["meta_lens"] = context_lens.clone()
                            recorded["meta_contiguous"] = context_lens.is_contiguous()
                            return object()

                        def fake_logits(
                            q_fp8,
                            kv_cache,
                            weights,
                            context_lens,
                            block_table,
                            schedule_meta,
                            max_seq_len,
                            clean_logits=False,
                        ):
                            recorded["logits_lens"] = context_lens.clone()
                            recorded["logits_contiguous"] = context_lens.is_contiguous()
                            recorded["kv_cache"] = kv_cache.clone()
                            recorded["weights_contiguous"] = weights.is_contiguous()
                            return torch.zeros(
                                (B * q_len, max_seq_len), dtype=torch.float32
                            )

                        fake_dg = types.SimpleNamespace(
                            get_num_sms=lambda: 8,
                            get_legacy_paged_mqa_logits_metadata=fake_meta,
                            fp8_paged_mqa_logits=fake_logits,
                        )

                        op = mod.IndexerDecodeV4Op(
                            index_n_heads=H,
                            index_head_dim=D,
                            index_topk=K,
                            softmax_scale=0.5,
                            scale_fmt=scale_fmt,
                            page_block_size=block_kv,
                        )
                        q = torch.zeros((B, q_len, H, D), dtype=torch.bfloat16)
                        kv = torch.zeros((B, T, D), dtype=torch.bfloat16)
                        weights = torch.ones((B, q_len, H), dtype=torch.bfloat16)
                        cmp_lens = torch.tensor([5, 7], dtype=torch.int32)
                        out_buf = torch.full((B, q_len, K), -1, dtype=torch.int32)

                        with (
                            patch.object(mod, "deep_gemm", fake_dg),
                            patch("torch.cuda.is_available", return_value=True),
                            patch(
                                "torch.cuda.get_device_capability", return_value=(10, 0)
                            ),
                            patch.dict(
                                sys.modules,
                                {
                                    "rtp_llm.models_py.kernels.cuda.fp8_kernel": fake_fp8_kernel,
                                    "rtp_llm.models_py.modules.base.cuda.indexer_op": fake_indexer_op,
                                },
                            ),
                        ):
                            op._forward_fast(q, kv, weights, cmp_lens, out_buf)

                        expected_lens = torch.tensor(
                            [[5] * q_len, [7] * q_len], dtype=torch.int32
                        )
                        self.assertTrue(recorded["meta_contiguous"])
                        self.assertTrue(recorded["logits_contiguous"])
                        self.assertTrue(recorded["weights_contiguous"])
                        torch.testing.assert_close(recorded["meta_lens"], expected_lens)
                        torch.testing.assert_close(
                            recorded["logits_lens"], expected_lens
                        )

                        kv_u8 = recorded["kv_cache"]
                        num_blocks = kv_u8.shape[0]
                        self.assertEqual(kv_u8.shape, (num_blocks, block_kv, 1, D + 4))
                        blk_bytes = kv_u8.view(num_blocks, block_kv * (D + 4))
                        total_tokens = num_blocks * block_kv
                        expected_k = (
                            (torch.arange(total_tokens, dtype=torch.int64) % 9 + 1)
                            .view(total_tokens, 1)
                            .expand(total_tokens, D)
                            .to(torch.float8_e4m3fn)
                            .view(torch.uint8)
                            .reshape(num_blocks, block_kv * D)
                        )
                        torch.testing.assert_close(
                            blk_bytes[:, : block_kv * D], expected_k
                        )
                        decoded_scales = (
                            blk_bytes[:, block_kv * D :]
                            .contiguous()
                            .view(torch.float32)
                        )
                        self.assertEqual(decoded_scales.shape, (num_blocks, block_kv))
                        expected_scales = torch.exp2(
                            (torch.arange(total_tokens, dtype=torch.int32) % 4).float()
                        ).view(num_blocks, block_kv)
                        torch.testing.assert_close(decoded_scales, expected_scales)

    def test_legacy_1d_deepgemm_abi_uses_per_request_lengths_for_unequal_mtp(self):
        from rtp_llm.models_py.modules.dsv4.decode import indexer_decode_op as dsv4_mod
        from rtp_llm.models_py.modules.dsv4.fp8 import _indexer_score as dsv4_score
        from rtp_llm.models_py.modules.dsv41.decode import (
            indexer_decode_op as dsv41_mod,
        )
        from rtp_llm.models_py.utils import arch

        B, H, D, T, K, block_kv = 2, 4, 128, 256, 4, 64
        cmp_lens = torch.tensor([64, 256], dtype=torch.int32)

        def fake_quant(x, **kwargs):
            rows = x.shape[0]
            return (
                torch.zeros((rows, D), dtype=torch.float8_e4m3fn),
                torch.ones((rows, 1), dtype=torch.float32),
            )

        fake_fp8_kernel = types.ModuleType("rtp_llm.models_py.kernels.cuda.fp8_kernel")
        fake_fp8_kernel.sgl_per_token_group_quant_fp8 = fake_quant
        fake_indexer_op = types.ModuleType(
            "rtp_llm.models_py.modules.base.cuda.indexer_op"
        )
        fake_indexer_op._unpack_ue8m0_scale = None

        try:
            arch._get_sm_for_device.cache_clear()
            for mod in (dsv4_mod, dsv41_mod):
                for q_len in (2, 3):
                    with self.subTest(module=mod.__name__, q_len=q_len):
                        recorded = {}

                        def fake_meta(lens, blk_sz, num_sms):
                            recorded["meta_lens"] = lens.clone()
                            return object()

                        def fake_logits(
                            q_fp8,
                            kv_cache,
                            weights,
                            context_lens,
                            block_table,
                            schedule_meta,
                            max_seq_len,
                            clean_logits=False,
                        ):
                            recorded["q_shape"] = tuple(q_fp8.shape)
                            recorded["logits_lens"] = context_lens.clone()
                            recorded["bt"] = block_table.clone()
                            return torch.zeros(
                                (B * q_len, max_seq_len), dtype=torch.float32
                            )

                        fake_dg_cu129 = types.SimpleNamespace(
                            __version__="2.1.1+ee8d9c7.cu129",
                            get_num_sms=lambda: 8,
                            get_paged_mqa_logits_metadata=fake_meta,
                            fp8_paged_mqa_logits=fake_logits,
                        )
                        op = mod.IndexerDecodeV4Op(
                            index_n_heads=H,
                            index_head_dim=D,
                            index_topk=K,
                            softmax_scale=0.5,
                            scale_fmt="fp32",
                            page_block_size=block_kv,
                        )
                        q = torch.zeros((B, q_len, H, D), dtype=torch.bfloat16)
                        kv = torch.zeros((B, T, D), dtype=torch.bfloat16)
                        weights = torch.ones((B, q_len, H), dtype=torch.bfloat16)
                        out_buf = torch.full((B, q_len, K), -1, dtype=torch.int32)
                        with (
                            patch.object(mod, "deep_gemm", fake_dg_cu129),
                            patch.object(
                                mod,
                                "prepare_fp8_paged_mqa_inputs",
                                side_effect=lambda *a, **kw: dsv4_score.prepare_fp8_paged_mqa_inputs(
                                    *a, device=torch.device("cuda", 0), **kw
                                ),
                            ),
                            patch("torch.cuda.is_available", return_value=True),
                            patch(
                                "torch.cuda.get_device_capability", return_value=(9, 0)
                            ),
                            patch.dict(
                                sys.modules,
                                {
                                    "rtp_llm.models_py.kernels.cuda.fp8_kernel": fake_fp8_kernel,
                                    "rtp_llm.models_py.modules.base.cuda.indexer_op": fake_indexer_op,
                                },
                            ),
                        ):
                            op._forward_fast(q, kv, weights, cmp_lens, out_buf)

                        if q_len == 2:
                            self.assertEqual(recorded["q_shape"], (B, 2, H, D))
                            self.assertEqual(tuple(recorded["meta_lens"].shape), (B,))
                            self.assertEqual(tuple(recorded["logits_lens"].shape), (B,))
                            torch.testing.assert_close(recorded["meta_lens"], cmp_lens)
                            torch.testing.assert_close(
                                recorded["logits_lens"], cmp_lens
                            )
                        else:
                            expected_flat = cmp_lens.repeat_interleave(q_len)
                            self.assertEqual(recorded["q_shape"], (B * q_len, 1, H, D))
                            self.assertEqual(
                                tuple(recorded["meta_lens"].shape), (B * q_len,)
                            )
                            torch.testing.assert_close(
                                recorded["meta_lens"], expected_flat
                            )
                            torch.testing.assert_close(
                                recorded["logits_lens"], expected_flat
                            )

            # Direct 2D unequal lengths on 1D backend flattens next_n>1 to keep per-token lengths exact.
            fake_dg_1d = types.SimpleNamespace(__version__="2.1.1+ee8d9c7.cu129")
            q_2d = torch.zeros((2, 2, H, D), dtype=torch.float8_e4m3fn)
            lens_2d = torch.tensor([[64, 65], [256, 257]], dtype=torch.int32)
            bt_2d = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            ):
                q_out, lens_out, bt_out = dsv4_score.prepare_fp8_paged_mqa_inputs(
                    q_2d,
                    lens_2d,
                    bt_2d,
                    device=torch.device("cuda", 0),
                    backend=fake_dg_1d,
                )
            self.assertEqual(tuple(q_out.shape), (4, 1, H, D))
            self.assertEqual(tuple(lens_out.shape), (4,))
            torch.testing.assert_close(
                lens_out, torch.tensor([64, 65, 256, 257], dtype=torch.int32)
            )
            torch.testing.assert_close(bt_out, bt_2d.repeat_interleave(2, dim=0))
        finally:
            arch._get_sm_for_device.cache_clear()

    def test_sm90_next_n_gt_2_flattens_paged_mqa_inputs(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import _indexer_score as dsv4_score
        from rtp_llm.models_py.modules.dsv41.fp8 import _indexer_score as dsv41_score
        from rtp_llm.models_py.utils import arch

        B, H, D, block_kv, max_blocks, max_ctx = 2, 64, 128, 64, 3, 192
        kv_pool = torch.zeros((max_blocks * block_kv, 132), dtype=torch.uint8)
        block_table = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32)
        cuda_dev = torch.device("cuda", 0)

        def make_distinct_q(next_n: int) -> torch.Tensor:
            return (
                (torch.arange(B * next_n, dtype=torch.int64) % 7 + 1)
                .view(B, next_n, 1, 1)
                .expand(B, next_n, H, D)
                .to(torch.float8_e4m3fn)
                .contiguous()
            )

        try:
            pin_2d = types.SimpleNamespace(
                get_legacy_paged_mqa_logits_metadata=lambda *_: None
            )
            arch._get_sm_for_device.cache_clear()
            for cap, next_n, expected_flatten in (
                ((9, 0), 1, False),
                ((9, 0), 2, False),
                ((9, 0), 3, True),
                ((10, 0), 3, False),
            ):
                q_fp8 = make_distinct_q(next_n)
                ctx_lens = torch.arange(1, B * next_n + 1, dtype=torch.int32).view(
                    B, next_n
                )
                with (
                    patch("torch.cuda.is_available", return_value=True),
                    patch("torch.cuda.get_device_capability", return_value=cap),
                ):
                    q_out, ctx_out, bt_out = dsv4_score.prepare_fp8_paged_mqa_inputs(
                        q_fp8, ctx_lens, block_table, device=cuda_dev, backend=pin_2d
                    )
                if expected_flatten:
                    self.assertEqual(tuple(q_out.shape), (B * next_n, 1, H, D))
                    self.assertEqual(tuple(ctx_out.shape), (B * next_n, 1))
                    torch.testing.assert_close(
                        q_out.to(torch.float32),
                        q_fp8.reshape(B * next_n, 1, H, D).to(torch.float32),
                    )
                    torch.testing.assert_close(ctx_out, ctx_lens.reshape(B * next_n, 1))
                    torch.testing.assert_close(
                        bt_out, block_table.repeat_interleave(next_n, dim=0)
                    )
                else:
                    self.assertEqual(tuple(q_out.shape), (B, next_n, H, D))
                    self.assertEqual(tuple(ctx_out.shape), (B, next_n))
                    torch.testing.assert_close(
                        q_out.to(torch.float32), q_fp8.to(torch.float32)
                    )
                    torch.testing.assert_close(ctx_out, ctx_lens)
                    torch.testing.assert_close(bt_out, block_table)

            # Verify both dsv4 and dsv41 fp8_paged_indexer_score pass adapted inputs
            # to metadata schedule and fp8_paged_mqa_logits.
            next_n = 3
            q_fp8 = make_distinct_q(next_n)
            w_fold = torch.ones((B * next_n, H), dtype=torch.float32)
            ctx_lens = torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.int32)
            for mod in (dsv4_score, dsv41_score):
                recorded = {}

                def fake_meta(lens, blk_sz, num_sms):
                    recorded["meta_lens_shape"] = tuple(lens.shape)
                    recorded["meta_lens"] = lens.clone()
                    return object()

                def fake_logits(q, kv, w, lens, bt, sched, max_ctx_len):
                    recorded["q_shape"] = tuple(q.shape)
                    recorded["q"] = q.clone()
                    recorded["lens_shape"] = tuple(lens.shape)
                    recorded["lens"] = lens.clone()
                    recorded["bt"] = bt.clone()
                    return torch.zeros((B * next_n, max_ctx_len), dtype=torch.float32)

                fake_dg = types.SimpleNamespace(
                    get_num_sms=lambda: 8,
                    get_legacy_paged_mqa_logits_metadata=fake_meta,
                    fp8_paged_mqa_logits=fake_logits,
                )
                fake_q = types.SimpleNamespace(
                    shape=q_fp8.shape,
                    dtype=q_fp8.dtype,
                    is_cuda=True,
                    device=cuda_dev,
                    dim=q_fp8.dim,
                    contiguous=lambda q=q_fp8: q,
                )
                with (
                    patch.object(mod, "_deep_gemm", fake_dg),
                    patch.object(dsv4_score, "_deep_gemm", fake_dg),
                    patch.object(dsv4_score, "_HAS_DEEP_GEMM", True),
                    patch("torch.cuda.is_available", return_value=True),
                    patch("torch.cuda.get_device_capability", return_value=(9, 0)),
                ):
                    out = mod.fp8_paged_indexer_score(
                        fake_q,
                        w_fold,
                        kv_pool,
                        block_table,
                        ctx_lens,
                        block_kv,
                        max_ctx,
                    )
                self.assertEqual(tuple(out.shape), (B * next_n, max_ctx))
                self.assertEqual(recorded["meta_lens_shape"], (B * next_n, 1))
                torch.testing.assert_close(
                    recorded["meta_lens"].cpu(), ctx_lens.reshape(B * next_n, 1)
                )
                self.assertEqual(recorded["q_shape"], (B * next_n, 1, H, D))
                torch.testing.assert_close(
                    recorded["q"].cpu().to(torch.float32),
                    q_fp8.reshape(B * next_n, 1, H, D).to(torch.float32),
                )
                self.assertEqual(recorded["lens_shape"], (B * next_n, 1))
                torch.testing.assert_close(
                    recorded["lens"].cpu(), ctx_lens.reshape(B * next_n, 1)
                )
                torch.testing.assert_close(
                    recorded["bt"].cpu(), block_table.repeat_interleave(next_n, dim=0)
                )
        finally:
            arch._get_sm_for_device.cache_clear()

    def test_cuda129_abi_and_base_indexer_fallback(self):
        from rtp_llm.models_py.modules.base.cuda import indexer_op as base_op
        from rtp_llm.models_py.modules.dsv4.fp8 import _indexer_score as dsv4_score
        from rtp_llm.models_py.modules.dsv41.fp8 import _indexer_score as dsv41_score

        fake_dg_1d = types.SimpleNamespace(
            __version__="2.1.1+ee8d9c7.cu129",
            fp8_mqa_logits=lambda *a: torch.zeros((a[0].shape[0], a[1][0].shape[0])),
            fp8_paged_mqa_logits=lambda *a, **kw: torch.zeros((a[0].shape[0], 64)),
            get_paged_mqa_logits_metadata=lambda *a: torch.empty(0, dtype=torch.int32),
        )
        cuda_dev = torch.device("cuda", 0)
        q = torch.zeros((2, 3, 64, 128), dtype=torch.float8_e4m3fn)
        bt = torch.tensor([[0], [1]], dtype=torch.int32)
        lens_1d = torch.tensor([32, 48], dtype=torch.int32)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        ):
            self.assertFalse(
                dsv4_score.supports_paged_mqa_logits(cuda_dev, 32, backend=fake_dg_1d)
            )
            self.assertTrue(
                dsv4_score.supports_paged_mqa_logits(cuda_dev, 64, backend=fake_dg_1d)
            )
            q_out, ctx_out, bt_out = dsv4_score.prepare_fp8_paged_mqa_inputs(
                q, lens_1d, bt, device=cuda_dev, backend=fake_dg_1d
            )
            self.assertEqual(tuple(q_out.shape), (6, 1, 64, 128))
            self.assertEqual(tuple(ctx_out.shape), (6,))
            self.assertEqual(tuple(bt_out.shape), (6, 1))

        # SM120 exports fp8_mqa_logits/fp8_paged_mqa_logits on 2.8.2; base/cuda
        # must route CUDA calls on SM120 to legacy fallback rather than calling it.
        fake_sm120_dg = types.SimpleNamespace(
            __version__="2.8.2+a5165ad.cu132",
            fp8_mqa_logits=lambda *a, **kw: self.fail(
                "should not call SM120 fp8_mqa_logits"
            ),
            fp8_paged_mqa_logits=lambda *a, **kw: self.fail(
                "should not call SM120 fp8_paged_mqa_logits"
            ),
            get_legacy_paged_mqa_logits_metadata=lambda *a: None,
            get_paged_mqa_logits_metadata=lambda *a: None,
        )
        fake_q_cuda = types.SimpleNamespace(is_cuda=True, device=cuda_dev)
        with (
            patch.object(base_op, "deep_gemm", fake_sm120_dg),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(12, 0)),
        ):
            self.assertFalse(base_op._can_use_deepgemm_mqa(fake_q_cuda))
            self.assertFalse(base_op._can_use_deepgemm_paged_mqa(fake_q_cuda, 64))

        # Pre-2.8 six-arg fp8_mqa_logits call omits max_seqlen_k.
        recorded_args = {}

        def fake_6arg_mqa(q_in, k_pair, w_in, ks, ke, clean):
            recorded_args["argc"] = 6
            recorded_args["clean"] = clean
            return torch.arange(16, dtype=torch.float32).view(2, 8)

        fake_dg_mqa = types.SimpleNamespace(
            __version__="2.1.1+ee8d9c7.cu129",
            fp8_mqa_logits=fake_6arg_mqa,
        )
        q3 = torch.zeros((2, 64, 128), dtype=torch.float8_e4m3fn)
        kq = torch.zeros((8, 128), dtype=torch.float8_e4m3fn)
        ks_f = torch.ones((8,), dtype=torch.float32)
        wf = torch.ones((2, 64), dtype=torch.float32)
        ks = torch.tensor([1, 2], dtype=torch.int32)
        ke = torch.tensor([5, 7], dtype=torch.int32)
        fake_q3 = types.SimpleNamespace(
            is_cuda=True,
            device=cuda_dev,
            dtype=q3.dtype,
            shape=q3.shape,
            dim=q3.dim,
            contiguous=lambda: q3,
        )
        from contextlib import nullcontext

        with (
            patch.object(dsv41_score, "_deep_gemm", fake_dg_mqa),
            patch.object(dsv4_score, "_deep_gemm", fake_dg_mqa),
            patch.object(dsv4_score, "_HAS_DEEP_GEMM_MQA", True),
            patch.object(dsv4_score, "_DEEP_GEMM_SM100_LEGACY_OK", True),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            patch("torch.cuda.device", lambda *_: nullcontext()),
            patch.object(torch.Tensor, "to", lambda self, *a, **kw: self),
        ):
            out = dsv41_score.fp8_mqa_indexer_score(
                fake_q3, wf, kq, ks_f, ks, ke, clean_logits=True, max_seqlen_k=4
            )
            self.assertEqual(recorded_args["argc"], 6)
            self.assertFalse(recorded_args["clean"])
            self.assertEqual(tuple(out.shape), (2, 4))

    def test_sm100_relative_clean_logits_dsv41(self):
        from contextlib import nullcontext

        from rtp_llm.models_py.modules.dsv4.fp8 import _indexer_score as dsv4_score
        from rtp_llm.models_py.modules.dsv41.fp8 import _indexer_score as dsv41_score

        cuda_dev = torch.device("cuda", 0)
        # DeepGEMM 2.8.2 (legacy 2D API) asserts `not clean_logits` when max_seqlen_k > 0.
        # Verify dsv41 passes clean_logits=False and masks tail columns to -inf.
        seen_clean = []

        def fake_282_mqa(q_in, k_pair, w_in, ks_t, ke_t, clean, max_k):
            seen_clean.append((clean, max_k))
            self.assertFalse(clean)
            return torch.arange(16, dtype=torch.float32).view(4, 4)

        fake_dg_282 = types.SimpleNamespace(
            __version__="2.8.2+a5165ad.cu132",
            fp8_mqa_logits=fake_282_mqa,
            get_legacy_paged_mqa_logits_metadata=lambda *a: None,
        )
        q4 = torch.zeros((4, 64, 128), dtype=torch.float8_e4m3fn)
        kq = torch.zeros((8, 128), dtype=torch.float8_e4m3fn)
        ks_f = torch.ones((8,), dtype=torch.float32)
        wf = torch.ones((4, 64), dtype=torch.float32)
        # Rows: [2,4) normal nonzero start, [1,3) short, [3,3) empty interval, [6,10) ke > N=8.
        ks = torch.tensor([2, 1, 3, 6], dtype=torch.int32)
        ke = torch.tensor([4, 3, 3, 10], dtype=torch.int32)
        fake_q4 = types.SimpleNamespace(
            is_cuda=True,
            device=cuda_dev,
            dtype=q4.dtype,
            shape=q4.shape,
            dim=q4.dim,
            contiguous=lambda: q4,
        )
        orig_to = torch.Tensor.to

        def fake_to(self, *args, **kwargs):
            if "device" in kwargs:
                kwargs = dict(kwargs)
                kwargs["device"] = self.device
            elif args and isinstance(args[0], (torch.device, str)):
                args = (self.device, *args[1:])
            return orig_to(self, *args, **kwargs)

        with (
            patch.object(dsv41_score, "_deep_gemm", fake_dg_282),
            patch.object(dsv4_score, "_deep_gemm", fake_dg_282),
            patch.object(dsv4_score, "_HAS_DEEP_GEMM_MQA", True),
            patch.object(dsv4_score, "_DEEP_GEMM_SM100_LEGACY_OK", True),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            patch("torch.cuda.device", lambda *_: nullcontext()),
            patch.object(torch.Tensor, "to", fake_to),
        ):
            out_mqa = dsv41_score.fp8_mqa_indexer_score(
                fake_q4, wf, kq, ks_f, ks, ke, clean_logits=True, max_seqlen_k=4
            )
        self.assertEqual(seen_clean, [(False, 4)])
        self.assertEqual(tuple(out_mqa.shape), (4, 4))
        self.assertTrue(torch.isfinite(out_mqa[0, :2]).all())
        self.assertTrue(torch.isneginf(out_mqa[0, 2:]).all())
        self.assertTrue(torch.isfinite(out_mqa[1, :2]).all())
        self.assertTrue(torch.isneginf(out_mqa[1, 2:]).all())
        self.assertTrue(torch.isneginf(out_mqa[2]).all())
        self.assertTrue(torch.isfinite(out_mqa[3, :2]).all())
        self.assertTrue(torch.isneginf(out_mqa[3, 2:]).all())

    def test_paged_context_lens_contract_and_sm100_relative_dsv4(self):
        from contextlib import nullcontext

        from rtp_llm.models_py.modules.dsv4.fp8 import _indexer_score as dsv4_score

        cuda_dev = torch.device("cuda", 0)
        orig_to = torch.Tensor.to

        def fake_to(self, *args, **kwargs):
            if "device" in kwargs:
                kwargs = dict(kwargs)
                kwargs["device"] = self.device
            elif args and isinstance(args[0], (torch.device, str)):
                args = (self.device, *args[1:])
            return orig_to(self, *args, **kwargs)

        q_paged = torch.zeros((2, 3, 64, 128), dtype=torch.float8_e4m3fn)
        w_paged = torch.ones((6, 64), dtype=torch.float32)
        kv_pool = torch.zeros((64, 132), dtype=torch.uint8)
        bt = torch.tensor([[0], [0]], dtype=torch.int32)
        fake_q_paged = types.SimpleNamespace(
            is_cuda=True,
            device=cuda_dev,
            dtype=q_paged.dtype,
            shape=q_paged.shape,
            dim=q_paged.dim,
            contiguous=lambda: q_paged,
        )
        fake_paged_dg = types.SimpleNamespace(
            __version__="2.1.1+ee8d9c7.cu129",
            fp8_paged_mqa_logits=lambda *a, **kw: torch.zeros(
                (a[0].shape[0] * a[0].shape[1], a[6])
            ),
            get_paged_mqa_logits_metadata=lambda *a: torch.empty(0, dtype=torch.int32),
            get_num_sms=lambda: 132,
        )
        with (
            patch.object(dsv4_score, "_deep_gemm", fake_paged_dg),
            patch.object(dsv4_score, "_HAS_DEEP_GEMM", True),
            patch.object(dsv4_score, "_DEEP_GEMM_SM100_LEGACY_OK", True),
            patch.object(dsv4_score, "is_sm120", return_value=False),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            patch("torch.cuda.device", lambda *_: nullcontext()),
            patch.object(torch.Tensor, "to", fake_to),
        ):
            for valid_len in (
                torch.tensor([12, 20], dtype=torch.int32),
                torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int32),
            ):
                out_p = dsv4_score.fp8_paged_indexer_score(
                    fake_q_paged,
                    w_paged,
                    kv_pool,
                    bt,
                    valid_len,
                    block_size=64,
                    max_ctx_len=32,
                )
                self.assertEqual(tuple(out_p.shape), (6, 32))
            with self.assertRaisesRegex(ValueError, "expected 2 or 6"):
                dsv4_score.fp8_paged_indexer_score(
                    fake_q_paged,
                    w_paged,
                    kv_pool,
                    bt,
                    torch.tensor([10, 20, 30], dtype=torch.int32),
                    block_size=64,
                    max_ctx_len=32,
                )

        seen_clean = []

        def fake_282_mqa(q_in, k_pair, w_in, ks_t, ke_t, clean, max_k):
            seen_clean.append((clean, max_k))
            self.assertFalse(clean)
            return torch.arange(16, dtype=torch.float32).view(4, 4)

        fake_dg_282 = types.SimpleNamespace(
            __version__="2.8.2+a5165ad.cu132",
            fp8_mqa_logits=fake_282_mqa,
            get_legacy_paged_mqa_logits_metadata=lambda *a: None,
        )
        q4 = torch.zeros((4, 64, 128), dtype=torch.float8_e4m3fn)
        kq = torch.zeros((8, 128), dtype=torch.float8_e4m3fn)
        ks_f = torch.ones((8,), dtype=torch.float32)
        wf = torch.ones((4, 64), dtype=torch.float32)
        ks = torch.tensor([2, 1, 3, 6], dtype=torch.int32)
        ke = torch.tensor([4, 3, 3, 10], dtype=torch.int32)
        fake_q4 = types.SimpleNamespace(
            is_cuda=True,
            device=cuda_dev,
            dtype=q4.dtype,
            shape=q4.shape,
            dim=q4.dim,
            contiguous=lambda: q4,
        )
        with (
            patch.object(dsv4_score, "_deep_gemm", fake_dg_282),
            patch.object(dsv4_score, "_HAS_DEEP_GEMM_MQA", True),
            patch.object(dsv4_score, "_DEEP_GEMM_SM100_LEGACY_OK", True),
            patch.object(dsv4_score, "is_sm120", return_value=False),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            patch("torch.cuda.device", lambda *_: nullcontext()),
            patch.object(torch.Tensor, "to", fake_to),
        ):
            out_mqa = dsv4_score.fp8_mqa_indexer_score(
                fake_q4, wf, kq, ks_f, ks, ke, clean_logits=True, max_seqlen_k=4
            )
        self.assertEqual(seen_clean, [(False, 4)])
        self.assertEqual(tuple(out_mqa.shape), (4, 4))
        self.assertTrue(torch.isfinite(out_mqa[0, :2]).all())
        self.assertTrue(torch.isneginf(out_mqa[0, 2:]).all())
        self.assertTrue(torch.isfinite(out_mqa[1, :2]).all())
        self.assertTrue(torch.isneginf(out_mqa[1, 2:]).all())
        self.assertTrue(torch.isneginf(out_mqa[2]).all())
        self.assertTrue(torch.isfinite(out_mqa[3, :2]).all())
        self.assertTrue(torch.isneginf(out_mqa[3, 2:]).all())

    def test_forward_fast_empty_kv_fills_out_buffer_with_minus_one(self):
        from rtp_llm.models_py.modules.dsv4.decode import indexer_decode_op as dsv4_mod
        from rtp_llm.models_py.modules.dsv41.decode import (
            indexer_decode_op as dsv41_mod,
        )

        B, q_len, H, D, K = 2, 2, 4, 128, 8
        for mod in (dsv4_mod, dsv41_mod):
            op = mod.IndexerDecodeV4Op(
                index_n_heads=H,
                index_head_dim=D,
                index_topk=K,
                softmax_scale=0.5,
                scale_fmt="ue8m0",
                page_block_size=64,
            )
            q = torch.zeros((B, q_len, H, D), dtype=torch.bfloat16)
            kv_empty = torch.zeros((B, 0, D), dtype=torch.bfloat16)
            weights = torch.ones((B, q_len, H), dtype=torch.bfloat16)
            cmp_lens = torch.zeros((B,), dtype=torch.int32)
            out_buf = torch.full((B, q_len, K), 99, dtype=torch.int32)

            ret = op._forward_fast(q, kv_empty, weights, cmp_lens, out_buf)
            self.assertIs(ret, out_buf)
            self.assertTrue(torch.all(ret == -1))

            # Empty batch/query preserves out_buffer cleanly without error.
            empty_out = torch.empty((0, q_len, K), dtype=torch.int32)
            ret_empty = op._forward_fast(
                q[:0], kv_empty[:0], weights[:0], cmp_lens[:0], empty_out
            )
            self.assertIs(ret_empty, empty_out)


if __name__ == "__main__":
    unittest.main()
