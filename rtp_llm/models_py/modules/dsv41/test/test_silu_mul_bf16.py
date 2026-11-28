"""UT for the fused BF16 SiLU + (optional clamp) + multiply Triton kernel
used by the V4.1 MXFP8 ``W13SharedExpert`` fused mid-chain.

The kernel must be bit-identical to the reference eager chain it replaces::

    gate_up = w13(x).float()               # bf16 -> fp32 (exact)
    gate, up = gate_up.chunk(2, dim=-1)
    hidden = silu_mul_split(               # fp32 math
        gate.contiguous(), up.contiguous(), clamp_limit=L)
    hidden.to(torch.bfloat16)              # single fp32 -> bf16 rounding

Tests cover:
  - V4.1 shared-expert shape (D = 2304, clamp via swiglu_limit = 10)
  - no-clamp path (swiglu_limit = 0)
  - small-D / non-pow2 D edge cases (BLOCK_N padding)
  - single-row and large-batch shapes
  - empty input
  - clamp arm actually firing
  - bit-exact equality (torch.equal) against the reference chain

Bypasses rtp_llm package init via importlib.
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from unittest.mock import patch

import torch


def _load_kernel(name="silu_mul_split_bf16"):
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.abspath(os.path.join(here, "..", "moe", "_silu_mul_bf16_triton.py"))
    spec = importlib.util.spec_from_file_location("_v4_silu_mul_bf16", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def _load_split_kernel():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.abspath(os.path.join(here, "..", "_silu_mul_split_triton.py"))
    spec = importlib.util.spec_from_file_location("_v4_silu_mul_split", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.silu_mul_split


def _ref_chain(gate_up_bf16: torch.Tensor, clamp_limit: float):
    """Reference chain matching the pre-fusion W13SharedExpert.forward."""
    silu_mul_split = _load_split_kernel()
    gate_up = gate_up_bf16.float()
    gate, up = gate_up.chunk(2, dim=-1)
    hidden = silu_mul_split(gate.contiguous(), up.contiguous(), clamp_limit=clamp_limit)
    return hidden.to(torch.bfloat16)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SiluMulSplitBf16EquivTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            _load_kernel()
        except Exception as e:
            raise unittest.SkipTest(f"silu_mul_split_bf16 not importable: {e}")

    def _check(self, *, M, D, clamp_limit):
        torch.manual_seed(0)
        device = "cuda:0"
        gate_up = (torch.randn(M, 2 * D, device=device, dtype=torch.float32) * 3.0).to(
            torch.bfloat16
        )
        # Make sure some values exceed clamp_limit so the clamp branch fires.
        if clamp_limit > 0 and M > 0:
            gate_up[0, :10] = clamp_limit + 1.0
            gate_up[0, D : D + 10] = clamp_limit + 1.0
            if M > 1:
                gate_up[1, D : D + 10] = -(clamp_limit + 1.0)

        ref = _ref_chain(gate_up, clamp_limit)
        out = _load_kernel()(gate_up, clamp_limit=clamp_limit)

        self.assertEqual(out.shape, ref.shape)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(
            torch.equal(out, ref),
            f"bit mismatch (M={M},D={D},L={clamp_limit}): "
            f"max abs diff {(out.float() - ref.float()).abs().max().item():.3e}, "
            f"mismatches {(out != ref).sum().item()}",
        )

    def test_shared_expert_v41_shape(self):
        # V4.1: moe_intermediate_size=2304, swiglu_limit=10.0.
        self._check(M=128, D=2304, clamp_limit=10.0)

    def test_no_clamp(self):
        self._check(M=64, D=2304, clamp_limit=0.0)

    def test_small_D(self):
        # D < default BLOCK_N=1024.
        self._check(M=16, D=384, clamp_limit=0.0)
        self._check(M=16, D=384, clamp_limit=4.0)

    def test_D_not_pow2(self):
        # D=1500 — BLOCK_N rounds up; mask discards padding.
        self._check(M=8, D=1500, clamp_limit=3.0)

    def test_single_row(self):
        self._check(M=1, D=2304, clamp_limit=10.0)
        self._check(M=1, D=2304, clamp_limit=0.0)

    def test_prefill_batch(self):
        # 16K CP4: 4096 rank-local tokens.
        self._check(M=4096, D=2304, clamp_limit=10.0)

    def test_empty_M(self):
        device = "cuda:0"
        gate_up = torch.empty(0, 4608, device=device, dtype=torch.bfloat16)
        out = _load_kernel()(gate_up, clamp_limit=10.0)
        self.assertEqual(out.shape, (0, 2304))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_out_buffer(self):
        torch.manual_seed(1)
        device = "cuda:0"
        gate_up = (torch.randn(32, 512, device=device, dtype=torch.float32)).to(
            torch.bfloat16
        )
        ref = _ref_chain(gate_up, 0.0)
        out = torch.empty(32, 256, device=device, dtype=torch.bfloat16)
        ret = _load_kernel()(gate_up, clamp_limit=0.0, out=out)
        self.assertIs(ret, out)
        self.assertTrue(torch.equal(out, ref))

    def test_clamp_actually_applied(self):
        torch.manual_seed(7)
        device = "cuda:0"
        gate_up = torch.full((4, 128), 10.0, device=device, dtype=torch.bfloat16)
        kernel = _load_kernel()
        no_clamp = kernel(gate_up, clamp_limit=0.0)
        clamped = kernel(gate_up, clamp_limit=2.0)
        self.assertFalse(torch.equal(no_clamp, clamped))
        self.assertTrue(torch.equal(clamped, _ref_chain(gate_up, 2.0)))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SiluMulGroup32QuantEquivTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        cls.quantize = staticmethod(sgl_per_token_group_quant_fp8)
        cls.bf16 = staticmethod(_load_kernel())
        cls.fused = staticmethod(_load_kernel("silu_mul_fp8_g32_quant"))

    def reference(self, gate_up, clamp, mode):
        hidden = self.bf16(gate_up, clamp_limit=clamp)
        with patch.dict(os.environ, DSV4_FP8_QUANT_KERNEL=mode):
            return self.quantize(
                hidden,
                group_size=32,
                eps=torch.finfo(torch.float32).tiny,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )

    def assert_exact(self, actual, expected, width):
        for a, b in zip(actual, expected):
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.dtype, b.dtype)
            self.assertEqual(a.device, b.device)
            self.assertEqual(a.stride(), b.stride())
        self.assertTrue(
            torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8))
        )
        # Legacy leaves the final pack's unused bytes unspecified.
        a = actual[1].clone(memory_format=torch.contiguous_format).view(torch.uint8)
        b = expected[1].clone(memory_format=torch.contiguous_format).view(torch.uint8)
        a, b = a[:, : width // 32], b[:, : width // 32]
        self.assertTrue(torch.equal(a, b), "packed UE8M0 scales differ")

    def check(self, gate_up, clamp, mode):
        expected = self.reference(gate_up, clamp, mode)
        with patch.dict(os.environ, DSV4_FP8_QUANT_KERNEL=" " + mode.upper() + " "):
            actual = self.fused(gate_up, clamp_limit=clamp)
        self.assert_exact(actual, expected, gate_up.shape[1] // 2)
        return actual

    def test_shapes_clamps_and_quant_modes(self):
        torch.manual_seed(51)
        # Include both sides of the former 4Mi-element auto boundary.
        for rows, width in (
            (0, 2304),
            (1, 32),
            (33, 160),
            (17, 1056),
            (1820, 2304),
            (1821, 2304),
            (2048, 2048),
        ):
            gate_up = (torch.randn(rows, 2 * width, device="cuda") * 16).bfloat16()
            for clamp in (0.0, 10.0):
                for mode in ("legacy", "v2", "auto"):
                    with self.subTest(rows=rows, width=width, clamp=clamp, mode=mode):
                        self.check(gate_up, clamp, mode)

    def test_auto_keeps_legacy_scales_across_former_size_boundary(self):
        # Zero Q alone cannot distinguish the policies: check packed scales too.
        for rows, width in ((1820, 2304), (1821, 2304), (2048, 2048), (3072, 2304)):
            with self.subTest(rows=rows, width=width):
                gate_up = torch.zeros(
                    rows, 2 * width, device="cuda", dtype=torch.bfloat16
                )
                actual = self.check(gate_up, 10.0, "auto")
                self.assert_exact(
                    actual, self.reference(gate_up, 10.0, "legacy"), width
                )
                scales = actual[1].clone(memory_format=torch.contiguous_format)
                codes = scales.view(torch.uint8)[:, : width // 32]
                # legacy floor: ceil(log2(1e-10)) + UE8M0 bias = -33 + 127.
                self.assertTrue(bool((codes == 94).all()))
                # Explicit v2 retains its distinct tiny-value scale semantics.
                self.check(gate_up, 10.0, "v2")

    def test_scale_rounding_signed_zero_and_dynamic_range(self):
        groups = []
        for exponent in (-130, -126, -120, -34, -33, -32, -10, -1, 0, 1, 10, 60):
            for maximum in (446.0, 448.0, 450.0):
                values = (
                    maximum,
                    -maximum,
                    0.0,
                    -0.0,
                    1.0625,
                    1.1875,
                    0.0009765625,
                    -1.0625,
                )
                groups.extend([value * 2.0**exponent for value in values] * 4)
        width = len(groups)
        # sigmoid(32) rounds to one; the multiply retains scale boundaries.
        gate_up = torch.full((5, 2 * width), 32.0, device="cuda", dtype=torch.bfloat16)
        gate_up[:, width:] = torch.tensor(groups, device="cuda").div(32).bfloat16()
        gate_up[1, :width] = -32.0
        gate_up[2, :width] = 0.0
        gate_up[3, :width] = -0.0
        gate_up[4, :width] = 1.0
        for mode in ("legacy", "v2", "auto"):
            for clamp in (0.0, 10.0):
                with self.subTest(mode=mode, clamp=clamp):
                    self.check(gate_up, clamp, mode)

    def test_native_nonfinite_quantization(self):
        values = torch.tensor(
            [
                float("nan"),
                float("inf"),
                -float("inf"),
                0.0,
                -0.0,
                1.0,
                -1.0,
                torch.finfo(torch.bfloat16).max,
            ],
            device="cuda",
            dtype=torch.bfloat16,
        )
        gate_up = values.repeat_interleave(32).repeat(3, 2)
        for mode in ("legacy", "v2"):
            for clamp in (0.0, 10.0):
                with self.subTest(mode=mode, clamp=clamp):
                    self.check(gate_up, clamp, mode)

    def test_bf16_only_buffers_empty_input_and_contiguous_slices(self):
        gate_up = torch.ones(3, 4608, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(3, 2304, device="cuda", dtype=torch.bfloat16)
        self.bf16(gate_up, out=out)
        # BF16-only callers must not read the quantization setting or allocate q/s.
        with patch.dict(os.environ, DSV4_FP8_QUANT_KERNEL="invalid"), patch.object(
            torch, "empty", side_effect=AssertionError("unexpected allocation")
        ):
            self.assertIs(self.bf16(gate_up, out=out), out)
        self.assertTrue(torch.equal(out, _ref_chain(gate_up, 0.0)))
        with patch.dict(os.environ, DSV4_FP8_QUANT_KERNEL="invalid"):
            self.assertEqual(self.fused(gate_up[:0])[0].shape, (0, 2304))
        # Forward checks were removed in 77275c0e6f: callers supply contiguous
        # BF16 input and a valid mode; rejection of invalid input is not an API.
        # A contiguous slice with a nonzero storage offset is still supported.
        for mode in ("legacy", "v2", "auto"):
            with self.subTest(mode=mode):
                self.check(gate_up[1:], 10.0, mode)

    def test_graph_replay_on_side_stream(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            gate_up = torch.ones(33, 4608, device="cuda", dtype=torch.bfloat16)
            for mode in ("legacy", "v2"):
                with patch.dict(os.environ, DSV4_FP8_QUANT_KERNEL=mode):
                    for _ in range(3):
                        self.fused(gate_up, clamp_limit=10.0)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        actual = self.fused(gate_up, clamp_limit=10.0)
                    for value in (0.0, 1e-20, 3.0, -20.0):
                        gate_up.fill_(value)
                        graph.replay()
                        self.assert_exact(
                            actual, self.reference(gate_up, 10.0, mode), 2304
                        )
        torch.cuda.current_stream().wait_stream(stream)

    @staticmethod
    def fp32_quant_reference(gate_up, clamp):
        """Independent FP32 SwiGLU and mathematical E4M3/UE8M0 reference."""
        gate, up = gate_up.float().chunk(2, dim=-1)
        if clamp > 0:
            gate = gate.clamp(max=clamp)
            up = up.clamp(-clamp, clamp)
        hidden = torch.nn.functional.silu(gate) * up
        rows, width = hidden.shape
        groups = hidden.reshape(rows, width // 32, 32)
        maximum = groups.abs().amax(dim=-1)
        # All fixture groups contain a nonzero, normal maximum, so the legacy
        # and v2 tiny-value floors do not affect this independent reference.
        raw_scale = maximum * (1.0 / 448.0)
        bits = raw_scale.view(torch.int32)
        encoded = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
        scale = (encoded << 23).view(torch.float32)
        quant = (groups / scale.unsqueeze(-1)).clamp(-448, 448)
        quant = quant.reshape(rows, width).to(torch.float8_e4m3fn)
        scale_bytes = torch.zeros(
            (rows, (width + 127) // 128 * 4),
            dtype=torch.uint8,
            device=gate_up.device,
        )
        scale_bytes[:, : width // 32] = encoded.to(torch.uint8)
        scales = torch.empty(
            ((width + 127) // 128, (rows + 3) // 4 * 4),
            dtype=torch.int32,
            device=gate_up.device,
        ).T[:rows]
        scales.copy_(scale_bytes.contiguous().view(torch.int32))
        return quant, scales

    @staticmethod
    def quant_threshold_fixture(rows, width):
        # Each group has a maximum 56.0967... which rounds to 56 in BF16.
        # Direct FP32 quantization needs scale 1/4; BF16 intermediate needs
        # scale 1/8. Other values straddle E4M3 midpoints after BF16 rounding.
        up = torch.tensor(
            [
                0.8984375,
                1.0703125,
                1.2421875,
                1.796875,
                2.140625,
                2.484375,
                3.59375,
                4.28125,
            ],
            dtype=torch.bfloat16,
            device="cuda",
        ).repeat(width // 8)
        gate = torch.ones(width, dtype=torch.bfloat16, device="cuda")
        gate[::32] = 8.125
        up[::32] = 6.90625
        return torch.cat((gate, up)).repeat(rows, 1)

    def test_fp32_swiglu_quantization_thresholds_preserve_legacy(self):
        gate_up = self.quant_threshold_fixture(5, 128)
        for mode in ("legacy", "v2", "auto"):
            with self.subTest(mode=mode), patch.dict(
                os.environ, DSV4_FP8_QUANT_KERNEL=mode
            ):
                actual = self.fused(gate_up, clamp_limit=10, round_to_bf16=False)
                expected = self.fp32_quant_reference(gate_up, 10)
                self.assert_exact(actual, expected, 128)
                legacy = self.fused(gate_up, clamp_limit=10)
                explicit_legacy = self.fused(
                    gate_up, clamp_limit=10, round_to_bf16=True
                )
                self.assert_exact(legacy, explicit_legacy, 128)
                self.assert_exact(legacy, self.reference(gate_up, 10, mode), 128)
                # A singleton packed dimension may retain its TMA stride even
                # when is_contiguous() is true; unpack values without a dtype view.
                codes = torch.stack(
                    [(actual[1] >> shift) & 255 for shift in (0, 8, 16, 24)], dim=-1
                )
                old_codes = torch.stack(
                    [(legacy[1] >> shift) & 255 for shift in (0, 8, 16, 24)], dim=-1
                )
                self.assertTrue(bool((codes == 125).all()))
                self.assertTrue(bool((old_codes == 124).all()))
                self.assertFalse(
                    torch.equal(
                        actual[0].view(torch.uint8), legacy[0].view(torch.uint8)
                    )
                )

    def test_fp32_swiglu_quantization_shapes_and_clamp(self):
        for rows, width in ((1, 32), (33, 160), (5, 2304)):
            gate_up = self.quant_threshold_fixture(rows, width)
            for clamp in (0.0, 10.0):
                if clamp:
                    gate_up[:, :width:32] = 12
                    gate_up[:, width::32] = -12
                for mode in ("legacy", "v2"):
                    with self.subTest(rows=rows, width=width, clamp=clamp, mode=mode):
                        with patch.dict(os.environ, DSV4_FP8_QUANT_KERNEL=mode):
                            actual = self.fused(
                                gate_up, clamp_limit=clamp, round_to_bf16=False
                            )
                        self.assert_exact(
                            actual, self.fp32_quant_reference(gate_up, clamp), width
                        )

    def test_fp32_swiglu_quantization_graph_replay(self):
        gate_up = self.quant_threshold_fixture(5, 2304)
        with patch.dict(os.environ, DSV4_FP8_QUANT_KERNEL="legacy"):
            for _ in range(3):
                self.fused(gate_up, clamp_limit=10, round_to_bf16=False)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = self.fused(gate_up, clamp_limit=10, round_to_bf16=False)
            for sign in (1, -1):
                gate_up[:, 2304:].mul_(sign)
                graph.replay()
                self.assert_exact(actual, self.fp32_quant_reference(gate_up, 10), 2304)


if __name__ == "__main__":
    unittest.main()
