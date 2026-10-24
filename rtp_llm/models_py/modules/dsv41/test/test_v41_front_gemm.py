"""Numerics and capture-stable reload contract for the DeepGEMM front GEMM."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41._v41_query_quant import try_project_quantized_qkv
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_front_gemm as front
from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear


class FrontGemmTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        props = torch.cuda.get_device_properties(0)
        if (props.major, props.minor, props.multi_processor_count) != (10, 3, 148):
            self.skipTest("SM103 / 148 SMs required")
        import deep_gemm

        self.assertTrue(
            hasattr(deep_gemm, "dsv41_front_gemm"), "DeepGEMM front API required"
        )
        self.op = deep_gemm.dsv41_front_gemm

    def make_attention(self):
        def linear(n, k):
            w = torch.randn(n, k, device="cuda").clamp(-3, 3).to(torch.float8_e4m3fn)
            s = torch.full((n // 32, k // 32), 0.0625, device="cuda").to(
                torch.float8_e8m0fnu
            )
            return V41MXFP8Linear(w, s)

        attn = SimpleNamespace(
            wq_a_wkv=linear(1792, 5120),
            wq_b=linear(128, 1280),
            index_weights=torch.randn(32, 5120, device="cuda", dtype=torch.bfloat16)
            * 0.02,
            q_norm=torch.ones(1280, device="cuda", dtype=torch.bfloat16),
            q_lora_rank=1280,
            eps=1e-6,
            is_index_source=True,
            is_kv_source=False,
        )
        front.prepare(attn)
        self.assertIsNotNone(getattr(attn, "_front_weights", None))
        return attn

    def reference(self, attn, x):
        q, kv, quant = try_project_quantized_qkv(attn, x)
        return (
            q,
            kv,
            quant,
            torch.nn.functional.linear(x.flatten(0, 1), attn.index_weights),
        )

    @patch.dict("os.environ", {"DSV41_FUSED_FRONT_PROJECTION": "1"})
    def test_older_deepgemm_keeps_fallback(self):
        import deep_gemm

        with torch.inference_mode():
            attn = self.make_attention()
            del attn._front_weights
            with patch.object(deep_gemm, "dsv41_front_gemm", None):
                front.prepare(attn)
            self.assertFalse(hasattr(attn, "_front_weights"))
            x = torch.randn(64, 2, 5120, device="cuda", dtype=torch.bfloat16)
            self.assertIsNone(front.project(attn, x))

    def check_result(self, got, ref):
        for a, b in (
            (got[0], ref[0]),
            (got[1], ref[1]),
            (got[2][0], ref[2][0]),
            (got[2][1], ref[2][1]),
        ):
            if a.dtype == torch.float8_e4m3fn:
                a, b = a.view(torch.uint8), b.view(torch.uint8)
            self.assertTrue(torch.equal(a, b))
        error = (got[3].float() - ref[3].float()).norm() / ref[
            3
        ].float().norm().clamp_min(1e-10)
        self.assertTrue(torch.isfinite(got[3]).all())
        self.assertLess(error.item(), 0.006)

    @patch.dict(
        "os.environ",
        {"DSV41_FUSED_QUERY_QUANT": "1", "DSV4_FP8_QUANT_KERNEL": "legacy"},
    )
    def test_default_enabled_and_explicit_disable(self):
        with patch.dict(os.environ):
            os.environ.pop("DSV41_FUSED_FRONT_PROJECTION", None)
            with torch.inference_mode():
                attn = self.make_attention()
                x = torch.randn(64, 2, 5120, device="cuda", dtype=torch.bfloat16)
                self.check_result(front.project(attn, x), self.reference(attn, x))
                os.environ["DSV41_FUSED_FRONT_PROJECTION"] = "0"
                self.assertIsNone(front.project(attn, x))
                del attn._front_weights
                front.prepare(attn)
                self.assertFalse(hasattr(attn, "_front_weights"))

    @patch.dict(
        "os.environ",
        {
            "DSV41_FUSED_FRONT_PROJECTION": "1",
            "DSV41_FUSED_QUERY_QUANT": "1",
            "DSV4_FP8_QUANT_KERNEL": "legacy",
        },
    )
    def test_eager_graph_reload(self):
        torch.manual_seed(20261019)
        with torch.inference_mode():
            for m in (128, 256, 384, 512):
                with self.subTest(m=m):
                    attn = self.make_attention()
                    x = torch.randn(
                        m // 2, 2, 5120, device="cuda", dtype=torch.bfloat16
                    )
                    self.check_result(front.project(attn, x), self.reference(attn, x))
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            front.project(attn, x)
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        got = front.project(attn, x)
                    graph.replay()
                    self.check_result(got, self.reference(attn, x))
                    pointers = [t.data_ptr() for t in attn._front_weights.tensors]
                    # Independently replace each input to expose stale-pointer bugs.
                    for target in ("weight", "weight_scales", "head"):
                        torch.cuda.synchronize()
                        if target == "head":
                            attn.index_weights = -attn.index_weights
                        elif target == "weight":
                            attn.wq_a_wkv.weight = (-attn.wq_a_wkv.weight.float()).to(
                                torch.float8_e4m3fn
                            )
                        else:
                            attn.wq_a_wkv.weight_scales = (
                                attn.wq_a_wkv.weight_scales + 0x01010101
                            )
                        with self.assertRaisesRegex(
                            RuntimeError, "without init_rope_cache"
                        ):
                            front.project(attn, x)
                        front.prepare(attn)
                        self.assertEqual(
                            pointers,
                            [t.data_ptr() for t in attn._front_weights.tensors],
                        )
                        x.mul_(0.5)
                        graph.replay()
                        self.check_result(got, self.reference(attn, x))
                    self.assertIsNone(front.project(attn, x[:1]))
                    torch.cuda.synchronize()
                    before = attn._front_weights.tensors[0].clone()
                    attn.index_weights = attn.index_weights[:, :-1].contiguous()
                    with self.assertRaisesRegex(ValueError, "layout changed"):
                        front.prepare(attn)
                    self.assertTrue(
                        torch.equal(
                            before.view(torch.uint8),
                            attn._front_weights.tensors[0].view(torch.uint8),
                        )
                    )


if __name__ == "__main__":
    unittest.main()
