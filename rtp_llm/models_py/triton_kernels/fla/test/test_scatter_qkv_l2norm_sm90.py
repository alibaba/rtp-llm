"""SM90 fused QKV preparation: exact BF16 boundary and FLA cache parity."""

import os
import unittest
from unittest.mock import patch

import torch

import rtp_llm.models_py.triton_kernels.common.scatter_qkv as scatter_ops
from rtp_llm.models_py.triton_kernels.common.scatter_qkv import (
    scatter_qkv,
    try_scatter_qkv_l2norm_sm90,
)
from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd


FLAG = "QWEN4_GDN_SM90_FUSED_SCATTER_NORM"
GEOMETRY = (2, 6, 128, 128)


class LaunchRecorder:
    def __init__(self, kernel):
        self.kernel = kernel
        self.launches = []

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def record(*args, **kwargs):
            result = launch(*args, **kwargs)
            self.launches.append((grid, kwargs["num_warps"], kwargs["num_stages"]))
            return result

        return record


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ScatterQKVL2NormSM90Test(unittest.TestCase):
    def setUp(self):
        if torch.version.hip or torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("Only qualified on CUDA SM90")

    def packed(self, length, stress=False, geometry=GEOMETRY):
        torch.manual_seed(4200 + length)
        h, hv, k, v = geometry
        x = torch.randn(length, 2 * h * k + hv * v, dtype=torch.bfloat16, device="cuda")
        if stress:
            qk = x[:, : 2 * h * k]
            powers = torch.randint(-14, 14, qk.shape, device=x.device)
            qk.mul_(torch.exp2(powers.float()))
            qk[0].zero_()
            qk[1].mul_(1e-5)
        return x

    def assert_bytes_equal(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertTrue(
            torch.equal(actual.view(torch.uint8), expected.contiguous().view(torch.uint8))
        )

    def test_guard_keeps_unqualified_geometries_on_original_path(self):
        x = self.packed(2048)
        with patch.dict(os.environ):
            os.environ.pop(FLAG, None)
            self.assertIsNone(try_scatter_qkv_l2norm_sm90(x, *GEOMETRY))
        with patch.dict(os.environ, {FLAG: "1"}):
            self.assertIsNotNone(try_scatter_qkv_l2norm_sm90(x, *GEOMETRY))
            for other in (x[:2047], x[:, :-1], x.float(), x.cpu()):
                self.assertIsNone(try_scatter_qkv_l2norm_sm90(other, *GEOMETRY))
            padded = torch.empty(2048, 1281, device=x.device, dtype=x.dtype)
            self.assertIsNone(try_scatter_qkv_l2norm_sm90(padded[:, :1280], *GEOMETRY))
            unaligned = padded.reshape(-1)[1 : 1 + x.numel()].view_as(x)
            self.assertTrue(unaligned.is_contiguous())
            self.assertIsNone(try_scatter_qkv_l2norm_sm90(unaligned, *GEOMETRY))
            self.assertIsNone(try_scatter_qkv_l2norm_sm90(x, 2, 4, 128, 128))
            with patch.object(scatter_ops, "gluon", None):
                self.assertIsNone(try_scatter_qkv_l2norm_sm90(x, *GEOMETRY))
            with patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)):
                self.assertIsNone(try_scatter_qkv_l2norm_sm90(x, *GEOMETRY))

    def test_opt_in_launches_one_multi_head_kernel(self):
        recorder = LaunchRecorder(scatter_ops._scatter_qkv_l2norm_sm90_multi_head)
        with patch.dict(os.environ, {FLAG: "1"}), patch.object(
            scatter_ops, "_scatter_qkv_l2norm_sm90_multi_head", recorder
        ):
            for length in (2048, 8191, 8192, 32768):
                with self.subTest(length=length):
                    before = len(recorder.launches)
                    self.assertIsNotNone(
                        try_scatter_qkv_l2norm_sm90(self.packed(length), *GEOMETRY)
                    )
                    self.assertEqual(
                        recorder.launches[before:], [(((length + 1) // 2,), 8, 1)]
                    )
        torch.cuda.synchronize()

    def test_exact_qkv_boundary(self):
        with patch.dict(os.environ, {FLAG: "1"}):
            for length in (2048, 8191, 8192, 32768):
                for stress in (False, True):
                    with self.subTest(length=length, stress=stress):
                        x = self.packed(length, stress)
                        q, k, v = scatter_qkv(x, *GEOMETRY)
                        reference = l2norm_fwd(q), l2norm_fwd(k), v
                        fused = try_scatter_qkv_l2norm_sm90(x, *GEOMETRY)
                        self.assertIsNotNone(fused)
                        for actual, expected in zip(fused, reference):
                            self.assert_bytes_equal(actual, expected)

    def test_exact_chunk_outputs_and_cache_states(self):
        with patch.dict(os.environ, {FLAG: "1", "QWEN4_GDN_SM90_BV16": "1"}):
            for lengths in ((2048, 31), (8192,), (32768,)):
                for stress in (False, True):
                    with self.subTest(lengths=lengths, stress=stress):
                        self.check_chunk(lengths, stress)

    def test_tp2_exact_qkv_and_chunk_boundaries(self):
        geometry = (8, 24, 128, 128)
        with patch.dict(os.environ, {FLAG: "1", "QWEN4_GDN_SM90_BV16": "1"}):
            for length in (8192, 32768):
                with self.subTest(length=length):
                    x = self.packed(length, True, geometry)
                    q, k, v = scatter_qkv(x, *geometry)
                    fused = try_scatter_qkv_l2norm_sm90(x, *geometry)
                    self.assertIsNotNone(fused)
                    for actual, expected in zip(fused, (l2norm_fwd(q), l2norm_fwd(k), v)):
                        self.assert_bytes_equal(actual, expected)
                    self.check_chunk((length,), False, geometry)

    def check_chunk(self, lengths, stress, geometry=GEOMETRY):
        m = sum(lengths)
        x = self.packed(m, stress, geometry)
        _, hv, k, v = geometry
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        cu = torch.tensor(offsets, dtype=torch.int32, device=x.device)
        gate = -torch.nn.functional.softplus(torch.randn(1, m, hv, device=x.device)) * 0.03
        beta = torch.randn(1, m, hv, device=x.device).sigmoid()
        initial = torch.randn(len(lengths), hv, v, k, device=x.device) * 0.02
        kwargs = dict(initial_state=initial, output_final_state=True, cu_seqlens=cu)
        baseline = chunk_gated_delta_rule(
            *scatter_qkv(x, *geometry), gate, beta,
            use_qk_l2norm_in_kernel=True, **kwargs,
        )
        fused = try_scatter_qkv_l2norm_sm90(x, *geometry)
        self.assertIsNotNone(fused)
        candidate = chunk_gated_delta_rule(
            *fused, gate, beta, use_qk_l2norm_in_kernel=False, **kwargs,
        )
        for name, actual, expected in zip(("output", "h", "final_state"), candidate, baseline):
            with self.subTest(tensor=name):
                self.assert_bytes_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
