from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4.quant_layouts import (
    _deep_gemm_rank_nvcc_tmpdir,
    prepare_fp4_weight_scale_for_deepgemm,
)


class _FakeScale:
    dtype = torch.float8_e8m0fnu

    def float(self):
        return "scale-fp32"


class QuantLayoutsTmpdirTest(unittest.TestCase):
    def test_transform_uses_rank_local_tmpdir_and_restores_environment(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            expected_tmpdir = os.path.join(
                cache_dir,
                "rtp_llm_dsv4_mega_moe_nvcc",
                "rank_3",
            )

            def transform(*args):
                self.assertEqual(os.environ["TMPDIR"], expected_tmpdir)
                self.assertTrue(os.path.isdir(expected_tmpdir))
                self.assertEqual(args, ("scale-fp32", 64, 32, (1, 32), 8))
                return "transformed"

            fake_deep_gemm = types.SimpleNamespace(
                transform_sf_into_required_layout=transform
            )
            with mock.patch.dict(
                os.environ,
                {"DG_JIT_CACHE_DIR": cache_dir, "TMPDIR": "/old/tmp"},
                clear=True,
            ), mock.patch.dict(sys.modules, {"deep_gemm": fake_deep_gemm}), mock.patch(
                "torch.distributed.is_available", return_value=True
            ), mock.patch(
                "torch.distributed.is_initialized", return_value=True
            ), mock.patch(
                "torch.distributed.get_rank", return_value=3
            ):
                self.assertEqual(
                    prepare_fp4_weight_scale_for_deepgemm(
                        _FakeScale(), 64, 32, num_groups=8
                    ),
                    "transformed",
                )
                self.assertEqual(os.environ["TMPDIR"], "/old/tmp")

    def test_transform_failure_restores_unset_tmpdir(self):
        fake_deep_gemm = types.SimpleNamespace(
            transform_sf_into_required_layout=mock.Mock(
                side_effect=RuntimeError("compile failed")
            )
        )
        with tempfile.TemporaryDirectory() as cache_dir:
            with mock.patch.dict(
                os.environ,
                {"DG_JIT_CACHE_DIR": cache_dir, "RANK": "7"},
                clear=True,
            ), mock.patch.dict(sys.modules, {"deep_gemm": fake_deep_gemm}), mock.patch(
                "torch.distributed.is_available", return_value=True
            ), mock.patch(
                "torch.distributed.is_initialized", return_value=False
            ):
                self.assertTrue(_deep_gemm_rank_nvcc_tmpdir().endswith("rank_7"))
                with self.assertRaisesRegex(RuntimeError, "compile failed"):
                    prepare_fp4_weight_scale_for_deepgemm(_FakeScale(), 64, 32)
                self.assertNotIn("TMPDIR", os.environ)


if __name__ == "__main__":
    unittest.main()
