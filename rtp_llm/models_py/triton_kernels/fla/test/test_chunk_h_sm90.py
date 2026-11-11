"""Explicit SM90 BV16 dispatch and exact wrapper parity with the BV32 path."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.triton_kernels.fla import chunk, chunk_delta_h


class KernelRecorder:
    def __init__(self, kernel):
        self.kernel = kernel
        self.tiles = []

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def recorded(*args, **kwargs):
            self.tiles.append(kwargs["BV"])
            return launch(*args, **kwargs)

        return recorded


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ChunkHSM90Test(unittest.TestCase):
    def setUp(self):
        if torch.version.hip or torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("This tile is qualified only on SM90")

    def inputs(self, length, seed, heads=(2, 6)):
        torch.manual_seed(seed)
        h, hv = heads
        cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
        k = torch.nn.functional.normalize(
            torch.randn(1, length, h, 128, device="cuda"), dim=-1
        ).to(torch.bfloat16)
        v = torch.randn(1, length, hv, 128, device="cuda", dtype=torch.bfloat16)
        beta = torch.sigmoid(torch.randn(1, length, hv, device="cuda"))
        raw_g = -torch.nn.functional.softplus(
            torch.randn(1, length, hv, device="cuda")
        ) * 0.03
        g = chunk.chunk_local_cumsum(raw_g, chunk_size=64, cu_seqlens=cu)
        a = chunk.chunk_scaled_dot_kkt_fwd(
            k=k, beta=beta, g_cumsum=g, cu_seqlens=cu
        )
        a = chunk.solve_tril(A=a, cu_seqlens=cu, output_dtype=k.dtype)
        w, u = chunk.recompute_w_u_fwd(
            k=k, v=v, beta=beta, A=a, g_cumsum=g, cu_seqlens=cu
        )
        initial = torch.randn(1, hv, 128, 128, device="cuda") * 0.02
        return dict(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=initial,
            output_final_state=True,
            cu_seqlens=cu,
            state_dtype=torch.float32,
        )

    def assert_exact(self, actual, expected):
        for name, value, reference in zip(
            ("h", "v_new", "final_state"), actual, expected
        ):
            self.assertTrue(torch.isfinite(value).all().item(), name)
            self.assertTrue(
                torch.equal(
                    value.contiguous().view(torch.uint8),
                    reference.contiguous().view(torch.uint8),
                ),
                name,
            )

    def test_default_dispatch_and_exact_long_sequence_outputs(self):
        self.check_exact_long_sequences((2, 6), (65, 8192, 32768))

    def test_tp2_default_dispatch_and_exact_long_sequence_outputs(self):
        self.check_exact_long_sequences((8, 24), (8192, 32768))

    def test_tp1_default_dispatch_and_exact_long_sequence_outputs(self):
        self.check_exact_long_sequences((16, 48), (65, 8192, 32768))

    def test_tp4_default_dispatch_and_exact_long_sequence_outputs(self):
        self.check_exact_long_sequences((4, 12), (65, 8192, 32768))

    def check_exact_long_sequences(self, heads, lengths):
        recorder = KernelRecorder(
            chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
        )
        with patch.object(
            chunk_delta_h,
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
            recorder,
        ):
            for length in lengths:
                with self.subTest(length=length):
                    inputs = self.inputs(length, 7000 + length, heads)
                    with patch.object(
                        chunk_delta_h, "_use_sm90_bv16_chunk_h", return_value=False
                    ):
                        expected = chunk_delta_h.chunk_gated_delta_rule_fwd_h(
                            **inputs
                        )
                    self.assertEqual(recorder.tiles[-1], 32)
                    actual = chunk_delta_h.chunk_gated_delta_rule_fwd_h(**inputs)
                    self.assertEqual(recorder.tiles[-1], 16)
                    self.assert_exact(actual, expected)

    def test_support_gate_preserves_unmeasured_paths(self):
        inputs = self.inputs(64, 34)
        k, w, u = (inputs[name] for name in ("k", "w", "u"))
        supported = chunk_delta_h._use_sm90_bv16_chunk_h
        self.assertTrue(supported(k, w, u, None, torch.float32, 1))
        self.assertFalse(supported(k, w, u, None, torch.float32, 2))
        self.assertFalse(supported(k, w, u, None, torch.bfloat16, 1))
        self.assertFalse(supported(k, w, u, inputs["g"], torch.float32, 1))
        self.assertFalse(supported(k.cpu(), w, u, None, torch.float32, 1))
        self.assertFalse(supported(k.float(), w, u, None, torch.float32, 1))
        self.assertFalse(supported(k, w.float(), u, None, torch.float32, 1))
        self.assertFalse(supported(k, w, u.float(), None, torch.float32, 1))
        self.assertFalse(supported(k[:, :, :1], w, u, None, torch.float32, 1))
        self.assertFalse(supported(k, w, u[:, :, :3], None, torch.float32, 1))
        for capability in ((8, 0), (10, 0), (12, 0)):
            with patch.object(
                chunk_delta_h.torch.cuda,
                "get_device_capability",
                return_value=capability,
            ):
                self.assertFalse(supported(k, w, u, None, torch.float32, 1))

    def test_multiple_sequences_keep_bv32(self):
        inputs = self.inputs(64, 35)
        inputs["cu_seqlens"] = torch.tensor(
            [0, 32, 64], device="cuda", dtype=torch.int32
        )
        inputs["initial_state"] = inputs["initial_state"].repeat(2, 1, 1, 1)
        recorder = KernelRecorder(
            chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
        )
        with patch.object(
            chunk_delta_h,
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
            recorder,
        ):
            chunk_delta_h.chunk_gated_delta_rule_fwd_h(**inputs)
        self.assertEqual(recorder.tiles, [32])


if __name__ == "__main__":
    unittest.main()
