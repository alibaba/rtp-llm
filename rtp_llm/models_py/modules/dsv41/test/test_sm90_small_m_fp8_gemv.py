"""Unit tests for SM90 small-M G32 FP8 dense GEMM optimization."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8._sm90_small_m_fp8_gemm import (
    attach_sm90_small_m_fp8_gemm,
    is_enabled,
    select_config,
    sm90_small_m_fp8_gemm_nt,
    warmup_sm90_small_m_fp8_gemm,
)


def _is_sm90() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _mn_major_stride(rows: int) -> int:
    return max(4, ((rows + 3) // 4) * 4)


def _make_g32_inputs(m: int, n: int, k: int, *, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x_q = (torch.randn((m, k), device="cuda", generator=gen) * 0.5).to(torch.float8_e4m3fn)
    stride = _mn_major_stride(m)
    x_s = torch.empty_strided((m, k // 32), (1, stride), device="cuda", dtype=torch.float32)
    exp_a = torch.randint(-4, 4, (m, k // 32), device="cuda", generator=gen)
    x_s.copy_(torch.exp2(exp_a.float()))

    w = (torch.randn((n, k), device="cuda", generator=gen) * 0.5).to(torch.float8_e4m3fn)
    exp_b = torch.randint(-4, 4, (n // 32, k // 32), device="cuda", generator=gen)
    w_s = torch.exp2(exp_b.float()).contiguous()
    return x_q, x_s, w, w_s


class Sm90SmallMFp8GemmPolicyTest(unittest.TestCase):
    def test_select_config_v41_shapes(self):
        # qkv_a (1792x5120): Split-K across M=1..80
        self.assertEqual(select_config(4, 1792, 5120), (0, 3, 1, 8))
        self.assertEqual(select_config(5, 1792, 5120), (0, 4, 2, 8))
        self.assertEqual(select_config(16, 1792, 5120), (0, 4, 2, 16))
        self.assertEqual(select_config(20, 1792, 5120), (0, 4, 2, 24))
        self.assertEqual(select_config(32, 1792, 5120), (1, 2, 1, 64))
        self.assertEqual(select_config(40, 1792, 5120), (1, 2, 1, 64))
        self.assertEqual(select_config(64, 1792, 5120), (1, 2, 1, 64))
        self.assertEqual(select_config(80, 1792, 5120), (2, 4, 1, 128))

        # o_proj (5120x8192): Swap64 <=32, NS32 <=64, Dual-WG NS64 SK=3 at 80
        self.assertEqual(select_config(4, 5120, 8192), (0, 1, 1, 8))
        self.assertEqual(select_config(32, 5120, 8192), (0, 1, 1, 32))
        self.assertEqual(select_config(40, 5120, 8192), (1, 1, 2, 64))
        self.assertEqual(select_config(64, 5120, 8192), (1, 1, 2, 64))
        self.assertEqual(select_config(80, 5120, 8192), (2, 3, 1, 128))

        # kv_b/q_b (32768x1280): Swap64 2-CTA <=32, NS64 2-CTA <=64, fallback >64
        self.assertEqual(select_config(4, 32768, 1280), (0, 1, 2, 8))
        self.assertEqual(select_config(32, 32768, 1280), (0, 1, 2, 32))
        self.assertEqual(select_config(40, 32768, 1280), (2, 1, 2, 64))
        self.assertEqual(select_config(64, 32768, 1280), (2, 1, 2, 64))
        self.assertIsNone(select_config(80, 32768, 1280))

        # Out-of-range / unaligned fallback
        self.assertIsNone(select_config(0, 1792, 5120))
        self.assertIsNone(select_config(129, 1792, 5120))
        self.assertIsNone(select_config(4, 1760, 5120))  # N % 64 != 0
        self.assertIsNone(select_config(4, 1792, 5152))  # K % 128 != 0

    def test_env_disable_and_non_sm90_noop(self):
        with patch.dict(os.environ, {"DSV41_SM90_SMALL_M_FP8_GEMM": "0"}):
            self.assertFalse(is_enabled())
            self.assertFalse(attach_sm90_small_m_fp8_gemm(SimpleNamespace(_sm90_linear=object())))
        self.assertFalse(attach_sm90_small_m_fp8_gemm(None))
        self.assertFalse(attach_sm90_small_m_fp8_gemm(SimpleNamespace(_sm90_linear=None)))


@unittest.skipUnless(_is_sm90(), "Requires NVIDIA SM90 (Hopper) GPU")
class Sm90SmallMFp8GemmGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert warmup_sm90_small_m_fp8_gemm(), "Failed to compile/warm SM90 small-M FP8 GEMM"

    def test_precision_and_determinism_all_v41_shapes(self):
        import deep_gemm

        shapes = [
            (1792, 5120),   # qkv_a
            (5120, 8192),   # o_proj
            (32768, 1280),  # kv_b / q_b
            (25600, 6144),  # engram
            (5120, 15360),  # fc_5120x15360
            (4608, 5120),   # shared_w13
            (5120, 2304),   # shared_w2
            (4096, 1280),   # index_wq
        ]
        ms = [1, 4, 5, 8, 16, 20, 32, 40, 64, 80, 128]
        for idx, (n, k) in enumerate(shapes):
            for m in ms:
                if select_config(m, n, k) is None:
                    continue
                x_q, x_s, w, w_s = _make_g32_inputs(m, n, k, seed=idx * 1000 + m)
                ref = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
                deep_gemm.fp8_gemm_nt((x_q, x_s), (w, w_s), ref, recipe=(1, 32, 32), compiled_dims="")

                out1 = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
                out2 = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
                h1 = sm90_small_m_fp8_gemm_nt((x_q, x_s), (w, w_s), out1)
                h2 = sm90_small_m_fp8_gemm_nt((x_q, x_s), (w, w_s), out2)
                torch.cuda.synchronize()

                self.assertTrue(h1 and h2, f"Expected fast path for {(m, n, k)}")
                self.assertTrue(torch.equal(out1, out2), f"Non-deterministic for {(m, n, k)}")
                diff = out1.float() - ref.float()
                rel_l2 = (diff.norm() / (ref.float().norm() + 1e-12)).item()
                cos = torch.nn.functional.cosine_similarity(
                    out1.float().flatten(), ref.float().flatten(), dim=0
                ).item()
                self.assertLess(rel_l2, 1e-3, f"rel_l2={rel_l2} on {(m, n, k)}")
                self.assertGreater(cos, 0.999999, f"cos={cos} on {(m, n, k)}")

    def test_ue8m0_edge_scales_and_zeros(self):
        import deep_gemm

        for m, n, k in [(4, 1792, 5120), (40, 5120, 8192), (80, 1792, 5120)]:
            x_q, x_s, w, w_s = _make_g32_inputs(m, n, k, seed=77)
            # Include subnormal UE8M0 byte 0 (2^-127) and zeros
            x_s[:, 0] = 2.0 ** -127
            w_s[0, 0] = 2.0 ** -127
            x_q[0, :32] = 0.0
            ref = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            deep_gemm.fp8_gemm_nt((x_q, x_s), (w, w_s), ref, recipe=(1, 32, 32), compiled_dims="")
            out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            self.assertTrue(sm90_small_m_fp8_gemm_nt((x_q, x_s), (w, w_s), out))
            torch.cuda.synchronize()
            self.assertFalse(torch.isnan(out).any().item())
            diff = out.float() - ref.float()
            rel_l2 = (diff.norm() / (ref.float().norm() + 1e-12)).item()
            self.assertLess(rel_l2, 1e-3)

    def test_fallback_for_large_m_and_non_mn_major_scales(self):
        import deep_gemm

        # M=160 (>128) must fall back to DeepGEMM
        x_q, x_s, w, w_s = _make_g32_inputs(160, 1792, 5120, seed=11)
        out = torch.empty((160, 1792), device="cuda", dtype=torch.bfloat16)
        handled = sm90_small_m_fp8_gemm_nt((x_q, x_s), (w, w_s), out)
        self.assertFalse(handled)

        # Row-major x_s must fall back cleanly
        x_q, x_s, w, w_s = _make_g32_inputs(4, 1792, 5120, seed=12)
        calls = []
        handled = sm90_small_m_fp8_gemm_nt(
            (x_q, x_s.contiguous()),
            (w, w_s),
            torch.empty((4, 1792), device="cuda", dtype=torch.bfloat16),
            fallback=lambda *a, **kw: calls.append(True),
        )
        self.assertFalse(handled)
        self.assertEqual(len(calls), 1)

    def test_cuda_graph_capture_and_replay_exact(self):
        import deep_gemm

        for m, n, k in [(4, 1792, 5120), (16, 1792, 5120), (40, 5120, 8192), (64, 32768, 1280), (80, 5120, 8192)]:
            x_q, x_s, w, w_s = _make_g32_inputs(m, n, k, seed=m * 13 + n)
            out_eager = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            out_graph = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            ref = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            deep_gemm.fp8_gemm_nt((x_q, x_s), (w, w_s), ref, recipe=(1, 32, 32), compiled_dims="")
            self.assertTrue(sm90_small_m_fp8_gemm_nt((x_q, x_s), (w, w_s), out_eager))
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                sm90_small_m_fp8_gemm_nt((x_q, x_s), (w, w_s), out_graph)
            for _ in range(5):
                out_graph.zero_()
                g.replay()
            torch.cuda.synchronize()

            self.assertTrue(torch.equal(out_graph, out_eager), f"Graph replay mismatch on {(m, n, k)}")
            diff = out_graph.float() - ref.float()
            rel_l2 = (diff.norm() / (ref.float().norm() + 1e-12)).item()
            self.assertLess(rel_l2, 1e-3)

    def test_v41_mxfp8_linear_attachment_integration(self):
        from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

        n, k = 1792, 5120
        gen = torch.Generator(device="cuda").manual_seed(99)
        w = (torch.randn((n, k), device="cuda", generator=gen) * 0.5).to(torch.float8_e4m3fn)
        s = torch.randint(123, 131, (n // 32, k // 32), device="cuda", dtype=torch.uint8, generator=gen).view(
            torch.float8_e8m0fnu
        )
        lin_ref = V41MXFP8Linear(w, s)
        lin_opt = V41MXFP8Linear(w, s)
        self.assertTrue(attach_sm90_small_m_fp8_gemm(lin_opt))
        self.assertTrue(getattr(lin_opt._sm90_linear, "_sm90_small_m_attached", False))

        for m in (1, 4, 5, 16, 32, 40, 64, 80):
            x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=gen)
            y_ref = lin_ref(x)
            y_opt = lin_opt(x)
            torch.cuda.synchronize()
            diff = y_opt.float() - y_ref.float()
            rel_l2 = (diff.norm() / (y_ref.float().norm() + 1e-12)).item()
            cos = torch.nn.functional.cosine_similarity(
                y_opt.float().flatten(), y_ref.float().flatten(), dim=0
            ).item()
            self.assertLess(rel_l2, 1e-3, f"V41MXFP8Linear M={m} rel_l2={rel_l2}")
            self.assertGreater(cos, 0.999999, f"V41MXFP8Linear M={m} cos={cos}")

    def test_row_sliced_mn_major_scales_and_unaligned_slice_fallback(self):
        import deep_gemm

        # 1) Tail slice [32:64] from a 64-row MN-major buffer (previously OOB if boxDim=stride)
        #    and [64:69] (M=5) from a 128-row MN-major buffer with NaN-poisoned padding rows [69:72].
        full_q, full_s, w, w_s = _make_g32_inputs(64, 1792, 5120, seed=301)
        sub_q, sub_s = full_q[32:64], full_s[32:64]
        ref = torch.empty((32, 1792), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((32, 1792), device="cuda", dtype=torch.bfloat16)
        deep_gemm.fp8_gemm_nt((sub_q, sub_s), (w, w_s), ref, recipe=(1, 32, 32), compiled_dims="")
        self.assertTrue(sm90_small_m_fp8_gemm_nt((sub_q, sub_s), (w, w_s), out))
        torch.cuda.synchronize()
        self.assertLess(((out.float() - ref.float()).norm() / (ref.float().norm() + 1e-12)).item(), 1e-3)

        big_q, big_s, w2, w2_s = _make_g32_inputs(128, 5120, 8192, seed=302)
        big_s[69:72, :] = float("nan")
        s5_q, s5_s = big_q[64:69], big_s[64:69]
        ref5 = torch.empty((5, 5120), device="cuda", dtype=torch.bfloat16)
        out5 = torch.empty((5, 5120), device="cuda", dtype=torch.bfloat16)
        # Copy valid 5 rows for reference so DeepGEMM doesn't see NaN in unused rows if it loads wider
        clean_s5 = torch.empty_strided((5, 8192 // 32), (1, 8), device="cuda", dtype=torch.float32)
        clean_s5.copy_(s5_s)
        deep_gemm.fp8_gemm_nt((s5_q, clean_s5), (w2, w2_s), ref5, recipe=(1, 32, 32), compiled_dims="")
        self.assertTrue(sm90_small_m_fp8_gemm_nt((s5_q, s5_s), (w2, w2_s), out5))
        torch.cuda.synchronize()
        self.assertFalse(torch.isnan(out5).any().item())
        self.assertLess(((out5.float() - ref5.float()).norm() / (ref5.float().norm() + 1e-12)).item(), 1e-3)

        # 2) Unaligned row offset [1:5] (16-byte misaligned SFA pointer) falls back cleanly
        fb_calls = []
        handled = sm90_small_m_fp8_gemm_nt(
            (big_q[1:5], big_s[1:5]),
            (w2, w2_s),
            torch.empty((4, 5120), device="cuda", dtype=torch.bfloat16),
            fallback=lambda *a, **kw: fb_calls.append(True),
        )
        self.assertFalse(handled)
        self.assertEqual(len(fb_calls), 1)

    def test_multi_stream_concurrent_splitk_safety(self):
        streams = [torch.cuda.Stream() for _ in range(4)]
        cases = []
        for idx, m in enumerate((4, 8, 16, 80)):
            x_q, x_s, w, w_s = _make_g32_inputs(m, 1792, 5120, seed=500 + idx)
            expected = torch.empty((m, 1792), device="cuda", dtype=torch.bfloat16)
            self.assertTrue(sm90_small_m_fp8_gemm_nt((x_q, x_s), (w, w_s), expected))
            cases.append((m, x_q, x_s, w, w_s, expected))
        torch.cuda.synchronize()

        outputs = [
            [torch.empty((cases[s][0], 1792), device="cuda", dtype=torch.bfloat16) for _ in range(12)]
            for s in range(4)
        ]
        for step in range(12):
            for s_idx, st in enumerate(streams):
                _, x_q, x_s, w, w_s, _ = cases[s_idx]
                with torch.cuda.stream(st):
                    self.assertTrue(sm90_small_m_fp8_gemm_nt((x_q, x_s), (w, w_s), outputs[s_idx][step]))
        torch.cuda.synchronize()
        for s_idx in range(4):
            expected = cases[s_idx][5]
            for step in range(12):
                self.assertTrue(torch.equal(outputs[s_idx][step], expected))


if __name__ == "__main__":
    unittest.main()
