import unittest

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.glm5_mega_moe.mega_nvfp4_input_packer_triton import (
    _cast_ue4m3_nearest,
    fused_pack_mega_nvfp4_inputs,
)


@triton.jit
def _test_cast_ue4m3_kernel(
    values, codes, scales, N: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.arange(0, BLOCK)
    value = tl.load(values + offsets, offsets < N, other=0.0)
    code, scale = _cast_ue4m3_nearest(value)
    tl.store(codes + offsets, code, offsets < N)
    tl.store(scales + offsets, scale, offsets < N)


class NVFP4InputPackerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This is a GPU numerical target: a missing device must fail, not
        # produce a successful test receipt containing only skipped cases.
        if not torch.cuda.is_available():
            raise RuntimeError("NVFP4InputPackerTest requires an SM100 CUDA device")

    def test_scale_cast_subnormal_ties_and_clamps(self):
        """Exercise FP32 rounding directly, without BF16 masking scale ties."""
        from deep_gemm.utils.math import cast_to_ue4m3_nearest

        def decode(code):
            exponent, mantissa = code >> 3, code & 7
            if exponent == 0:
                return mantissa / 512.0
            return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)

        # Every positive finite code, every adjacent midpoint, both FP32
        # neighbors, and the clamp boundaries (including zero and infinity).
        representable = [decode(code) for code in range(1, 127)]
        midpoint = torch.tensor(
            [(a + b) / 2 for a, b in zip(representable, representable[1:])],
            dtype=torch.float32,
            device="cuda",
        )
        values = torch.cat(
            [
                torch.tensor(
                    [
                        -1.0,
                        -0.0,
                        0.0,
                        1.0 / 1024.0,
                        1.0 / 512.0,
                        448.0,
                        449.0,
                        float("inf"),
                    ]
                    + representable,
                    dtype=torch.float32,
                    device="cuda",
                ),
                midpoint,
                torch.nextafter(midpoint, torch.full_like(midpoint, -float("inf"))),
                torch.nextafter(midpoint, torch.full_like(midpoint, float("inf"))),
            ]
        )
        codes = torch.empty_like(values, dtype=torch.uint8)
        scales = torch.empty_like(values)
        _test_cast_ue4m3_kernel[(1,)](
            values,
            codes,
            scales,
            values.numel(),
            triton.next_power_of_2(values.numel()),
        )
        expected_scales, expected_codes = cast_to_ue4m3_nearest(values)
        self.assertTrue(torch.equal(codes, expected_codes))
        self.assertTrue(torch.equal(scales, expected_scales))

    def test_tiny_groups_match_reference_and_graph_updates(self):
        """A tiny group beside a large row maximum must keep subnormal scales."""
        hidden, topk = 6144, 4
        # Include the measured small vector decode tiles, partial tiles, and
        # the separate large Prefill path under changed-input Graph replay.
        for tokens in (17, 25, 40, 80, 128, 4097):
            with self.subTest(tokens=tokens):
                x = torch.zeros(tokens, hidden, dtype=torch.bfloat16, device="cuda")
                x[1].fill_(-0.0)
                # Row amax=2688 gives exactly GSF=1. These BF16 values then
                # quantize to FP4 codes 7 and 1, formerly 2 and 0 respectively.
                x[2, :16] = 3.0 / 256.0
                x[3, :16] = 1.0 / 1024.0
                x[2:8, -1] = 2688.0
                # BF16-representable group amax=6*scale at every subnormal
                # midpoint, including the subnormal-to-normal midpoint.
                amplitudes = torch.tensor(
                    [6.0 / 512.0]
                    + [6.0 * (code + 0.5) / 512.0 for code in range(1, 8)]
                    + [6.0 / 64.0, 6.375, 7.125],
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                below = torch.nextafter(
                    amplitudes, torch.full_like(amplitudes, -float("inf"))
                )
                above = torch.nextafter(
                    amplitudes, torch.full_like(amplitudes, float("inf"))
                )
                for row, values in ((4, amplitudes), (5, below), (6, above)):
                    x[row, : values.numel() * 16] = (
                        values[:, None].expand(-1, 16).reshape(-1)
                    )
                x[7, :16] = torch.tensor(
                    [
                        0.0,
                        -0.0,
                        0.25,
                        0.75,
                        1.25,
                        1.75,
                        2.5,
                        3.5,
                        5.0,
                        6.0,
                        -0.25,
                        -0.75,
                        -1.25,
                        -1.75,
                        -2.5,
                        -5.0,
                    ],
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                weights = torch.rand(tokens, topk, dtype=torch.float32, device="cuda")
                indices = torch.randint(
                    0, 128, (tokens, topk), dtype=torch.int64, device="cuda"
                )
                outputs = self._allocate(tokens, hidden, topk)

                def run():
                    fused_pack_mega_nvfp4_inputs(x, weights, indices, *outputs)

                run()
                self._assert_matches_reference(x, weights, indices, outputs)
                scale_bytes = outputs[1].view(torch.uint8).reshape(tokens, -1)
                self.assertTrue(bool(torch.all(scale_bytes[0] == 1)))
                self.assertEqual(int(scale_bytes[2, 0]), 1)
                self.assertEqual(int(scale_bytes[3, 0]), 1)
                self.assertEqual(int(outputs[0][2, 0]) & 15, 7)
                self.assertEqual(int(outputs[0][3, 0]) & 15, 1)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                pointers = [
                    tensor.data_ptr() for tensor in (x, weights, indices, *outputs)
                ]
                for step in range(4):
                    x.neg_()
                    weights.mul_(0.5)
                    indices.add_(1)
                    if step == 3:
                        x.zero_()
                    graph.replay()
                    self._assert_matches_reference(x, weights, indices, outputs)
                    self.assertEqual(
                        pointers,
                        [
                            tensor.data_ptr()
                            for tensor in (x, weights, indices, *outputs)
                        ],
                    )

    @staticmethod
    def _allocate(tokens, hidden, topk):
        device = torch.device("cuda")
        return (
            torch.empty(tokens, hidden // 2, dtype=torch.int8, device=device),
            torch.empty(tokens, hidden // 64, dtype=torch.int32, device=device),
            torch.empty(tokens, dtype=torch.float32, device=device),
            torch.empty(tokens, topk, dtype=torch.int64, device=device),
            torch.empty(tokens, topk, dtype=torch.float32, device=device),
        )

    @staticmethod
    def _reference(x):
        from deep_gemm.utils import per_token_cast_to_nvfp4

        return per_token_cast_to_nvfp4(x, gran_k=16, use_packed_ue4m3=True)

    def _assert_matches_reference(self, x, weights, indices, outputs):
        fp4, sf, gsf, out_indices, out_weights = outputs
        expected_fp4, expected_sf, expected_gsf = self._reference(x)
        self.assertTrue(torch.equal(fp4, expected_fp4))
        self.assertTrue(torch.equal(sf, expected_sf))
        self.assertTrue(torch.equal(gsf, expected_gsf))
        self.assertTrue(torch.equal(out_indices, indices))
        self.assertTrue(torch.equal(out_weights, weights))

    def test_real_m31_decode_shapes_are_bitwise_exact(self):
        torch.manual_seed(20260923)
        hidden, topk = 6144, 4
        for tokens in (1, 4, 8, 16):
            with self.subTest(tokens=tokens):
                x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
                # Exercise positive zero, signed zero and saturation-scale rows.
                x[0, :4] = torch.tensor(
                    [0.0, -0.0, 1.0e-7, -1.0e-7],
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                weights = torch.rand(tokens, topk, dtype=torch.float32, device="cuda")
                indices = torch.randint(
                    0, 128, (tokens, topk), dtype=torch.int64, device="cuda"
                )
                outputs = self._allocate(tokens, hidden, topk)
                fused_pack_mega_nvfp4_inputs(x, weights, indices, *outputs)
                torch.cuda.synchronize()
                self._assert_matches_reference(x, weights, indices, outputs)

    def test_prefill_default_tile_is_bitwise_exact(self):
        """Exercise the adaptive BLOCK_M=16 path with a non-divisible row count."""
        torch.manual_seed(20260924)
        tokens, hidden, topk = 2049, 6144, 4
        x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
        weights = torch.rand(tokens, topk, dtype=torch.float32, device="cuda")
        indices = torch.randint(
            0, 128, (tokens, topk), dtype=torch.int64, device="cuda"
        )
        outputs = self._allocate(tokens, hidden, topk)
        fused_pack_mega_nvfp4_inputs(x, weights, indices, *outputs)
        torch.cuda.synchronize()
        self._assert_matches_reference(x, weights, indices, outputs)

    def test_preallocated_cuda_graph_tracks_updated_inputs(self):
        torch.manual_seed(31)
        tokens, hidden, topk = 16, 6144, 4
        x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
        weights = torch.rand(tokens, topk, dtype=torch.float32, device="cuda")
        indices = torch.randint(
            0, 128, (tokens, topk), dtype=torch.int64, device="cuda"
        )
        outputs = self._allocate(tokens, hidden, topk)

        def run():
            fused_pack_mega_nvfp4_inputs(x, weights, indices, *outputs)

        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        x.copy_(torch.randn_like(x))
        weights.copy_(torch.rand_like(weights))
        indices.copy_(torch.randint_like(indices, 0, 128))
        graph.replay()
        torch.cuda.synchronize()
        self._assert_matches_reference(x, weights, indices, outputs)

        first = tuple(output.clone() for output in outputs)
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
        for actual, expected in zip(outputs, first):
            self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
