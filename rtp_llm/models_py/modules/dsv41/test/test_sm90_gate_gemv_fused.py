# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the SM90 fused Gate-GEMV/WGMMA + sqrtsoftplus + TopK + FP8 Pack path."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41 import _sm90_gate_gemv_fused as fused_mod
from rtp_llm.models_py.modules.dsv41.moe.strategies.mega_sm90 import (
    SM90MegaMoEStrategy,
)


def _is_sm90_cuda() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


class _FakeBuf:
    def __init__(self, capacity: int, hidden: int, topk: int, device: torch.device):
        self.x = torch.empty((capacity, hidden), dtype=torch.float8_e4m3fn, device=device)
        self.x_sf = torch.empty((capacity, hidden // 128), dtype=torch.float32, device=device)
        self.topk_weights = torch.empty((capacity, topk), dtype=torch.float32, device=device)
        self.topk_idx = torch.empty((capacity, topk), dtype=torch.int64, device=device)


class Sm90GateGemvFusedCpuContractTest(unittest.TestCase):
    """Host-side eligibility and kill-switch tests (run on any machine)."""

    def test_rejects_cpu_and_env_killswitch(self):
        x = torch.zeros((4, 5120), dtype=torch.bfloat16)
        w = torch.zeros((384, 5120), dtype=torch.bfloat16)
        b = torch.zeros((384,), dtype=torch.float32)
        buf = _FakeBuf(4, 5120, 6, torch.device("cpu"))
        self.assertFalse(
            fused_mod.can_use_sm90_gate_gemv_pack(
                x, w, b, buf.x, buf.x_sf, buf.topk_weights, buf.topk_idx, topk=6
            )
        )
        with mock.patch.dict(os.environ, {"DSV4_SM90_GATE_GEMV_FUSED": "0"}):
            self.assertFalse(fused_mod.warmup_sm90_gate_gemv_pack())


@unittest.skipUnless(_is_sm90_cuda(), "Requires SM90 CUDA GPU")
class Sm90GateGemvFusedCudaTest(unittest.TestCase):
    """SM90 hardware precision, edge-case, determinism, strategy-wiring, and CUDA Graph tests."""

    def setUp(self):
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.assertTrue(fused_mod.warmup_sm90_gate_gemv_pack(self.device))

    def _run_reference(self, x, w, bias, topk, route_scale=2.5, norm_eps=1e-12, exact_fp32_dot=False):
        from rtp_llm.models_py.modules.dsv41._sm90_gate_fused_triton import (
            fused_sqrtsoftplus_gate_pack,
        )
        from rtp_llm.models_py.modules.dsv41.moe.mega_sm90_input_packer import (
            MegaMoeSM90InputPacker,
        )

        m, h = x.shape
        buf = _FakeBuf(m, h, topk, self.device)
        scores = (
            F.linear(x.float(), w.float()).to(torch.bfloat16)
            if exact_fp32_dot
            else F.linear(x, w)
        )
        fused_sqrtsoftplus_gate_pack(
            scores.contiguous(),
            bias.contiguous(),
            x,
            buf.x,
            buf.x_sf,
            buf.topk_weights,
            buf.topk_idx,
            topk=topk,
            route_scale=route_scale,
            norm_eps=norm_eps,
        )
        # Canonical bit-exact FP8 K128 packing oracle (exact IEEE 1.0 / scale)
        MegaMoeSM90InputPacker().pack(x, buf.topk_weights, buf.topk_idx, buf, m)
        return buf

    def _run_fused(self, x, w, bias, topk, route_scale=2.5, norm_eps=1e-12):
        m, h = x.shape
        buf = _FakeBuf(m, h, topk, self.device)
        self.assertTrue(
            fused_mod.can_use_sm90_gate_gemv_pack(
                x, w, bias, buf.x, buf.x_sf, buf.topk_weights, buf.topk_idx, topk=topk
            )
        )
        fused_mod.fused_sm90_gate_gemv_pack(
            x,
            w,
            bias,
            buf.x,
            buf.x_sf,
            buf.topk_weights,
            buf.topk_idx,
            topk=topk,
            route_scale=route_scale,
            norm_eps=norm_eps,
        )
        return buf

    def _assert_pack_bitwise_equal(self, actual: _FakeBuf, expected: _FakeBuf):
        self.assertTrue(
            torch.equal(actual.x.view(torch.uint8), expected.x.view(torch.uint8)),
            "FP8 packed activations differ from reference",
        )
        self.assertTrue(
            torch.equal(actual.x_sf.view(torch.int32), expected.x_sf.view(torch.int32)),
            "FP32 K128 scales differ from reference",
        )

    @torch.no_grad()
    def test_matches_baseline_across_all_decode_buckets(self):
        """Verify precision & bit-exact FP8 packing across M=1..128 for E=384,K=6 and E=128,K=3."""
        h = 5120
        for e, k in ((384, 6), (128, 3)):
            for m in (1, 4, 5, 8, 16, 20, 32, 40, 64, 80, 128):
                with self.subTest(E=e, K=k, M=m):
                    g = torch.Generator(device=self.device).manual_seed(20261125 + e + m)
                    # Well-separated normal logits (realistic scale)
                    x = torch.randn((m, h), dtype=torch.bfloat16, device=self.device, generator=g)
                    w = (
                        torch.randn((e, h), dtype=torch.bfloat16, device=self.device, generator=g)
                        * (h ** -0.5)
                    )
                    bias = (
                        torch.randn((e,), dtype=torch.float32, device=self.device, generator=g)
                        * 0.1
                    )

                    ref = self._run_reference(x, w, bias, k)
                    act = self._run_fused(x, w, bias, k)
                    torch.cuda.synchronize()

                    self._assert_pack_bitwise_equal(act, ref)
                    idx_match = (act.topk_idx == ref.topk_idx).all().item()
                    w_diff = (act.topk_weights - ref.topk_weights).abs().max().item()
                    if not idx_match or w_diff >= 1e-4:
                        # When a dot product lands within ~1e-6 of a BF16 half-ULP
                        # midpoint, different FP32 accumulation orders (cuBLAS vs
                        # our kernel) can round to adjacent BF16 values. Verify our
                        # output matches the exact FP32->BF16 reference on those rows.
                        exact_ref = self._run_reference(x, w, bias, k, exact_fp32_dot=True)
                        torch.cuda.synchronize()
                        self.assertGreaterEqual(
                            (act.topk_idx == ref.topk_idx).float().mean().item(), 0.995
                        )
                        both_or_exact_idx = (act.topk_idx == ref.topk_idx) | (
                            act.topk_idx == exact_ref.topk_idx
                        )
                        self.assertTrue(both_or_exact_idx.all().item())
                        min_w_diff = torch.minimum(
                            (act.topk_weights - ref.topk_weights).abs(),
                            (act.topk_weights - exact_ref.topk_weights).abs(),
                        )
                        self.assertLess(min_w_diff.max().item(), 1e-4)
                    else:
                        self.assertTrue(idx_match)
                        self.assertLess(w_diff, 1e-4)

    @torch.no_grad()
    def test_strided_padded_input_view(self):
        """Non-contiguous padded [M, 5120] view (stride(0) = 5248, stride(1) = 1) works cleanly."""
        h = 5120
        m = 4
        g = torch.Generator(device=self.device).manual_seed(777)
        backing = torch.randn((m, h + 128), dtype=torch.bfloat16, device=self.device, generator=g)
        x = backing[:, :h]
        self.assertFalse(x.is_contiguous())
        w = torch.randn((384, h), dtype=torch.bfloat16, device=self.device, generator=g) * 0.02
        bias = torch.randn((384,), dtype=torch.float32, device=self.device, generator=g) * 0.1

        ref = self._run_reference(x, w, bias, 6)
        act = self._run_fused(x, w, bias, 6)
        torch.cuda.synchronize()

        self._assert_pack_bitwise_equal(act, ref)
        self.assertTrue(torch.equal(act.topk_idx, ref.topk_idx))
        self.assertLess((act.topk_weights - ref.topk_weights).abs().max().item(), 1e-4)

    @torch.no_grad()
    def test_nonfinite_inputs_bias_and_zero_ties(self):
        """NaN/Inf in x or bias triggers exact fallback; all-zero ties pick [0..K-1]."""
        h = 5120
        for e, k, m in ((384, 6, 4), (384, 6, 16), (384, 6, 64), (384, 6, 80), (128, 3, 5), (128, 3, 64)):
            with self.subTest(E=e, K=k, M=m):
                g = torch.Generator(device=self.device).manual_seed(9000 + e + m)
                x = torch.randn((m, h), dtype=torch.bfloat16, device=self.device, generator=g)
                w = torch.randn((e, h), dtype=torch.bfloat16, device=self.device, generator=g) * 0.02
                bias = torch.randn((e,), dtype=torch.float32, device=self.device, generator=g) * 0.1

                # Inject NaN, +Inf, -Inf, and an all-zero row
                x[0, 17] = float("nan")
                x[1, 256] = float("inf")
                x[2, 511] = -float("inf")
                x[3].zero_()

                ref = self._run_reference(x, w, bias, k, route_scale=2.5)
                act = self._run_fused(x, w, bias, k, route_scale=2.5)
                torch.cuda.synchronize()

                self._assert_pack_bitwise_equal(act, ref)
                expected_fallback_idx = torch.arange(k, dtype=torch.int64, device=self.device)
                expected_fallback_w = torch.full((k,), 2.5 / k, dtype=torch.float32, device=self.device)
                for bad_row in (0, 1, 2):
                    self.assertTrue(torch.equal(act.topk_idx[bad_row], expected_fallback_idx))
                    torch.testing.assert_close(
                        act.topk_weights[bad_row], expected_fallback_w, rtol=0, atol=0
                    )
                self.assertTrue(torch.isfinite(act.topk_weights).all().item())

                # Poison bias -> every row must fall back identically
                poison_bias = bias.clone()
                poison_bias[7] = float("nan")
                act_pb = self._run_fused(x, w, poison_bias, k, route_scale=2.5)
                torch.cuda.synchronize()
                for row in range(m):
                    self.assertTrue(torch.equal(act_pb.topk_idx[row], expected_fallback_idx))
                    torch.testing.assert_close(
                        act_pb.topk_weights[row], expected_fallback_w, rtol=0, atol=0
                    )

                # All-zero x and zero bias -> exact tie-breaking [0..K-1]
                zero_x = torch.zeros_like(x)
                zero_bias = torch.zeros_like(bias)
                act_z = self._run_fused(zero_x, w, zero_bias, k, route_scale=2.5)
                ref_z = self._run_reference(zero_x, w, zero_bias, k, route_scale=2.5)
                torch.cuda.synchronize()
                self._assert_pack_bitwise_equal(act_z, ref_z)
                self.assertTrue(torch.equal(act_z.topk_idx, ref_z.topk_idx))
                for row in range(m):
                    self.assertTrue(torch.equal(act_z.topk_idx[row], expected_fallback_idx))

    @torch.no_grad()
    def test_determinism_across_repeated_runs(self):
        """Repeated invocations produce bitwise-identical outputs."""
        h = 5120
        for e, k, m in ((384, 6, 4), (384, 6, 16), (384, 6, 64), (384, 6, 80), (128, 3, 5)):
            with self.subTest(E=e, K=k, M=m):
                g = torch.Generator(device=self.device).manual_seed(42 + e + m)
                x = torch.randn((m, h), dtype=torch.bfloat16, device=self.device, generator=g)
                w = torch.randn((e, h), dtype=torch.bfloat16, device=self.device, generator=g) * 0.02
                bias = torch.randn((e,), dtype=torch.float32, device=self.device, generator=g) * 0.1
                first = self._run_fused(x, w, bias, k)
                torch.cuda.synchronize()
                for _ in range(5):
                    nxt = self._run_fused(x, w, bias, k)
                    torch.cuda.synchronize()
                    self._assert_pack_bitwise_equal(nxt, first)
                    self.assertTrue(torch.equal(nxt.topk_idx, first.topk_idx))
                    self.assertTrue(
                        torch.equal(
                            nxt.topk_weights.view(torch.int32),
                            first.topk_weights.view(torch.int32),
                        )
                    )

    @torch.no_grad()
    def test_cuda_graph_capture_and_replay_with_mutated_inputs(self):
        """Verify CUDA Graph capture & multi-step replay across 1-kernel, mid-M, and WGMMA buckets."""
        h = 5120
        for e, k, m in ((384, 6, 4), (384, 6, 16), (384, 6, 64), (384, 6, 80), (128, 3, 5), (128, 3, 80)):
            with self.subTest(E=e, K=k, M=m):
                x = torch.randn((m, h), dtype=torch.bfloat16, device=self.device)
                w = torch.randn((e, h), dtype=torch.bfloat16, device=self.device) * 0.02
                bias = torch.randn((e,), dtype=torch.float32, device=self.device) * 0.1
                buf = _FakeBuf(m, h, k, self.device)

                # Warmup before capture
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        fused_mod.fused_sm90_gate_gemv_pack(
                            x, w, bias, buf.x, buf.x_sf, buf.topk_weights, buf.topk_idx,
                            topk=k, route_scale=2.5, norm_eps=1e-12,
                        )
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fused_mod.fused_sm90_gate_gemv_pack(
                        x, w, bias, buf.x, buf.x_sf, buf.topk_weights, buf.topk_idx,
                        topk=k, route_scale=2.5, norm_eps=1e-12,
                    )

                # Mutate inputs in-place across multiple replays (including NaN step!)
                for step in range(4):
                    g = torch.Generator(device=self.device).manual_seed(1234 + step * 17 + m)
                    x.copy_(torch.randn((m, h), dtype=torch.bfloat16, device=self.device, generator=g))
                    w.copy_(torch.randn((e, h), dtype=torch.bfloat16, device=self.device, generator=g) * 0.02)
                    bias.copy_(torch.randn((e,), dtype=torch.float32, device=self.device, generator=g) * 0.1)
                    if step == 2:
                        x[0, 5] = float("nan")
                    graph.replay()
                    ref = self._run_reference(x, w, bias, k, route_scale=2.5)
                    torch.cuda.synchronize()
                    self._assert_pack_bitwise_equal(buf, ref)
                    if step == 2:
                        self.assertTrue(
                            torch.equal(
                                buf.topk_idx[0],
                                torch.arange(k, dtype=torch.int64, device=self.device),
                            )
                        )
                    else:
                        self.assertGreaterEqual(
                            (buf.topk_idx == ref.topk_idx).float().mean().item(), 0.995
                        )

    @torch.no_grad()
    def test_strategy_forward_with_gate_pack_uses_fused_gemv_path(self):
        """End-to-end check that SM90MegaMoEStrategy.forward_with_gate_pack dispatches to fused GEMV."""
        h, e, k, m = 5120, 384, 6, 4
        cfg = mock.Mock(
            dim=h,
            is_decode_role=True,
            fake_balance_expert=None,
            swiglu_limit=0.0,
        )
        strategy = SM90MegaMoEStrategy(cfg)
        buf = _FakeBuf(m, h, k, self.device)
        strategy._mega_buf = buf
        strategy._mega_y = torch.zeros((m, h), dtype=torch.bfloat16, device=self.device)
        strategy._input_packer = mock.Mock()

        g = torch.Generator(device=self.device).manual_seed(555)
        x = torch.randn((m, h), dtype=torch.bfloat16, device=self.device, generator=g)
        w = torch.randn((e, h), dtype=torch.bfloat16, device=self.device, generator=g) * 0.02
        bias = torch.randn((e,), dtype=torch.float32, device=self.device, generator=g) * 0.1

        gate = mock.Mock(
            hash=False,
            bias=bias,
            score_func="sqrtsoftplus",
            topk=k,
            route_scale=2.5,
            weight=w,
            _ced_row_projection=None,
            _prefill_gate_chunk_rows=0,
            _has_visual_tokens=False,
            _dbg_prefix=None,
        )
        gate._weight_bf16.return_value = w
        gate._project_scores.side_effect = AssertionError(
            "Fallback F.linear must not be called when fused GEMV path is eligible"
        )

        with mock.patch.object(strategy, "_launch") as launch:
            strategy.forward_with_gate_pack(x, gate, None)
            launch.assert_called_once()
        torch.cuda.synchronize()

        ref = self._run_reference(x, w, bias, k, route_scale=2.5)
        self._assert_pack_bitwise_equal(buf, ref)
        self.assertTrue(torch.equal(buf.topk_idx, ref.topk_idx))
        self.assertLess((buf.topk_weights - ref.topk_weights).abs().max().item(), 1e-4)

    @torch.no_grad()
    def test_multi_stream_concurrent_execution(self):
        """Concurrent launches across multiple non-default streams must not race on partial/flag."""
        h = 5120
        configs = [(384, 6, 4), (128, 3, 5), (384, 6, 16), (384, 6, 64)]
        streams = [torch.cuda.Stream() for _ in configs]
        cases = []
        for idx, (e, k, m) in enumerate(configs):
            g = torch.Generator(device=self.device).manual_seed(8100 + idx)
            x = torch.randn((m, h), dtype=torch.bfloat16, device=self.device, generator=g)
            w = torch.randn((e, h), dtype=torch.bfloat16, device=self.device, generator=g) * 0.02
            bias = torch.randn((e,), dtype=torch.float32, device=self.device, generator=g) * 0.1
            expected = self._run_fused(x, w, bias, k)
            cases.append((e, k, m, x, w, bias, expected))
        torch.cuda.synchronize()

        bufs = [
            [_FakeBuf(cases[s][2], h, cases[s][1], self.device) for _ in range(10)]
            for s in range(len(configs))
        ]
        for step in range(10):
            for s_idx, st in enumerate(streams):
                _, k, _, x, w, bias, _ = cases[s_idx]
                b = bufs[s_idx][step]
                with torch.cuda.stream(st):
                    fused_mod.fused_sm90_gate_gemv_pack(
                        x, w, bias, b.x, b.x_sf, b.topk_weights, b.topk_idx,
                        topk=k, route_scale=2.5, norm_eps=1e-12,
                    )
        torch.cuda.synchronize()
        for s_idx in range(len(configs)):
            expected = cases[s_idx][6]
            for step in range(10):
                b = bufs[s_idx][step]
                self._assert_pack_bitwise_equal(b, expected)
                self.assertTrue(torch.equal(b.topk_idx, expected.topk_idx))
                self.assertTrue(
                    torch.equal(
                        b.topk_weights.view(torch.int32),
                        expected.topk_weights.view(torch.int32),
                    )
                )


if __name__ == "__main__":
    unittest.main()
