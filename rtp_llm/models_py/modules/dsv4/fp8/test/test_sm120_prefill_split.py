"""Exact production split parity and opt-in/fallback regressions on SM120."""

import importlib.util
import os
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _sm120_prefill_indices as indices
from rtp_llm.models_py.modules.dsv4.fp8 import _sm120_prefill_split as fused
from rtp_llm.models_py.modules.dsv4.fp8._swa_ops_triton import (
    combine_topk_swa_indices,
    combine_topk_swa_indices_cp,
)
from rtp_llm.models_py.utils.arch import is_sm120


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Sm120PrefillSplitTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0")
        self.assertTrue(is_sm120(self.device))
        torch.manual_seed(2781)

    def assertParity(self, ci, cl, **kw):
        originals = (ci.clone(), cl.clone())
        kw["device"] = self.device
        with mock.patch.object(indices, "_SPLIT_FUSION_ENABLED", False):
            expected = indices.split_sm120_combined_tables(ci, cl, **kw)
        self.assertTrue(fused.can_fuse_split(ci, cl, **kw))
        with mock.patch.object(indices, "_SPLIT_FUSION_ENABLED", True):
            actual = indices.split_sm120_combined_tables(ci, cl, **kw)
        for got, want in zip(actual[:4], expected[:4]):
            self.assertEqual(got.dtype, want.dtype)
            self.assertEqual(got.device, want.device)
            self.assertEqual(got.shape, want.shape)
            self.assertTrue(got.is_contiguous())
            self.assertTrue(torch.equal(got, want))
        self.assertEqual(actual[4], expected[4])
        self.assertTrue(torch.equal(ci, originals[0]))
        self.assertTrue(torch.equal(cl, originals[1]))

    def test_cp2_cp4_real_producers(self):
        for cp in (2, 4):
            for rank in range(cp):
                for topk, ratio, window in (
                    (0, 1, 8),
                    (8, 4, 16),
                    (128, 4, 128),
                    (256, 128, 128),
                    (2048, 4, 128),
                ):
                    with self.subTest(cp=cp, rank=rank, topk=topk):
                        pair = 32768 // (2 * cp)
                        gp = torch.cat(
                            (
                                torch.arange(
                                    rank * pair, rank * pair + 31, device=self.device
                                ),
                                torch.arange(
                                    32768 - (rank + 1) * pair,
                                    32768 - (rank + 1) * pair + 36,
                                    device=self.device,
                                ),
                            )
                        ).long()
                        n = 32768 // ratio if topk else 0
                        m = n + 32768 + window
                        top = torch.randint(
                            0,
                            max(n, 1),
                            (67, topk),
                            device=self.device,
                            dtype=torch.int32,
                        )
                        ci, cl = combine_topk_swa_indices_cp(
                            top,
                            gp,
                            0,
                            window,
                            ratio,
                            topk,
                            m,
                            n,
                            flash_mla_indices=rank % 2 == 0,
                        )
                        self.assertParity(
                            ci,
                            cl,
                            M=m,
                            N=n,
                            window_size=window,
                            extra_width=max(topk, 1),
                            ratio=ratio,
                        )

    def test_populated_csa_hca_full_rows(self):
        for ratio, topk in ((4, 2048), (128, 256)):
            for rows in (1024, 8192):
                with self.subTest(ratio=ratio, rows=rows):
                    n = 32768 // ratio
                    m = n + 32768
                    gp = torch.arange(
                        32768 - rows, 32768, device=self.device, dtype=torch.int64
                    )
                    top = (
                        torch.arange(topk, device=self.device, dtype=torch.int32)
                        .unsqueeze(0)
                        .expand(rows, -1)
                    )
                    ci, cl = combine_topk_swa_indices_cp(
                        top, gp, 0, 128, ratio, topk, m, n
                    )
                    self.assertParity(
                        ci, cl, M=m, N=n, window_size=128, extra_width=topk, ratio=ratio
                    )

    def test_cp1_varlen_and_empty_tail(self):
        for topk, ratio in ((0, 1), (7, 4), (128, 4), (256, 128), (2048, 4)):
            n = 32768 // ratio if topk else 0
            m = n + 32768 + 128
            top = torch.randint(
                0, max(n, 1), (80, topk), device=self.device, dtype=torch.int32
            )
            ci, cl = combine_topk_swa_indices(
                top,
                torch.tensor([0, 13, 80], device=self.device, dtype=torch.int32),
                torch.tensor([32768, 24576], device=self.device, dtype=torch.int32),
                torch.tensor([32768, 24576], device=self.device, dtype=torch.int32),
                128,
                ratio,
                topk,
                m,
                n,
            )
            self.assertParity(
                ci, cl, M=m, N=n, window_size=128, extra_width=max(topk, 1), ratio=ratio
            )
            self.assertParity(
                ci[:0],
                cl[:0],
                M=m,
                N=n,
                window_size=128,
                extra_width=max(topk, 1),
                ratio=ratio,
            )

    def test_metadata_rejection_and_legacy_fallback(self):
        ci = torch.full((8, 128), -1, device=self.device, dtype=torch.int32)
        cl = torch.zeros(8, device=self.device, dtype=torch.int32)
        kw = dict(
            M=1024, N=256, window_size=128, extra_width=1, ratio=4, device=self.device
        )
        for a, b in ((ci[::2], cl[::2]), (ci.long(), cl), (ci, cl.long())):
            self.assertFalse(fused.can_fuse_split(a, b, **kw))
            with mock.patch.object(indices, "_SPLIT_FUSION_ENABLED", False):
                expected = indices.split_sm120_combined_tables(a, b, **kw)
            with mock.patch.object(
                indices, "_SPLIT_FUSION_ENABLED", True
            ), mock.patch.object(
                fused,
                "fused_split",
                side_effect=AssertionError("ineligible kernel called"),
            ):
                actual = indices.split_sm120_combined_tables(a, b, **kw)
            for x, y in zip(actual[:4], expected[:4]):
                self.assertTrue(torch.equal(x, y))
        self.assertFalse(fused.can_fuse_split(ci, cl.cpu(), **kw))
        self.assertFalse(fused.can_fuse_split(ci.cpu(), cl.cpu(), **kw))
        self.assertFalse(
            fused.can_fuse_split(ci, cl, **dict(kw, device=torch.device("cuda:1")))
        )
        for name, value in (
            ("M", 256),
            ("N", -1),
            ("window_size", 129),
            ("extra_width", 129),
            ("ratio", 3),
        ):
            self.assertFalse(fused.can_fuse_split(ci, cl, **dict(kw, **{name: value})))

    def test_default_off_is_frozen(self):
        self.assertFalse(indices._SPLIT_FUSION_ENABLED)
        ci = torch.full((1, 128), -1, device=self.device, dtype=torch.int32)
        cl = torch.zeros(1, device=self.device, dtype=torch.int32)
        with mock.patch.dict(
            os.environ, {"DSV4_SM120_PREFILL_SPLIT_FUSION": "1"}
        ), mock.patch.object(
            fused,
            "fused_split",
            side_effect=AssertionError("default path called kernel"),
        ):
            result = indices.split_sm120_combined_tables(
                ci,
                cl,
                M=1024,
                N=0,
                window_size=128,
                extra_width=1,
                ratio=1,
                device=self.device,
            )
            self.assertEqual(result[4], 2)
            self.assertFalse(indices._SPLIT_FUSION_ENABLED)

    def test_unknown_mode_fails_closed(self):
        spec = importlib.util.spec_from_file_location(
            "invalid_split_mode", indices.__file__
        )
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(os.environ, {"DSV4_SM120_PREFILL_SPLIT_FUSION": "typo"}):
            with self.assertRaisesRegex(RuntimeError, "PREFILL_SPLIT_FUSION"):
                spec.loader.exec_module(module)


if __name__ == "__main__":
    unittest.main()
