"""GLM53 CUDA norm routing, mutation and graph contracts."""

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.glm53.norm import (
    Glm53RMSNorm,
    Glm53RMSResNorm,
    _rows_per_launch,
)


def _inputs(rows, seed, dtype=torch.bfloat16, hidden_size=4096):
    rng = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((rows, hidden_size), device="cuda", generator=rng, dtype=dtype)
    r = torch.randn((rows, hidden_size), device="cuda", generator=rng, dtype=dtype)
    w = torch.randn((hidden_size,), device="cuda", generator=rng, dtype=dtype)
    return x, r, w


def _reference(x, residual, weight):
    # Preserve the unrounded FP32 sum for the fused normalization. The separate
    # residual output rounds that same sum to the input dtype.
    summed = x.float() if residual is None else x.float() + residual.float()
    normed = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-5)
    return (normed * weight.float()).to(x.dtype), summed.to(x.dtype)


class Glm53NormTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise the model initialization path before importing the generic
        # API, including the packaged CUTLASS import path and CUDA JIT build.
        Glm53RMSNorm(torch.ones(4096, device="cuda", dtype=torch.bfloat16))

    def test_eager_math_alias_and_global_selector(self):
        # The test environment pins the generic selector OFF. GLM must use its
        # own CUDA entry without changing generic FlashInfer or another model.
        import flashinfer.norm as generic

        self.assertFalse(generic._USE_CUDA_NORM)
        for dtype in (torch.bfloat16, torch.float16):
            for rows in (1, 17, 122, 257):
                for pdl in (False, True, None):
                    with self.subTest(dtype=dtype, rows=rows, pdl=pdl):
                        x, r, w = _inputs(rows, 5302, dtype)
                        original = x.clone()
                        norm = Glm53RMSNorm(w, 1e-5, enable_pdl=pdl)
                        self.assertIs(norm.weight, w)
                        expected, _ = _reference(x, None, w)
                        with patch.object(
                            generic,
                            "rmsnorm",
                            side_effect=AssertionError("generic route"),
                        ):
                            out = torch.empty_like(x)
                            self.assertIs(norm(x, out), out)
                        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
                        self.assertTrue(torch.equal(x, original))
                        alias = x.clone()
                        self.assertIs(norm(alias, alias), alias)
                        self.assertTrue(torch.equal(alias, out))
                        fused = Glm53RMSResNorm(w, 1e-5, enable_pdl=pdl)
                        expected_x, expected_r = _reference(x, r, w)
                        with patch.object(
                            generic,
                            "fused_add_rmsnorm",
                            side_effect=AssertionError("generic route"),
                        ):
                            result = fused(x, r)
                        self.assertIs(result[0], x)
                        self.assertIs(result[1], r)
                        torch.testing.assert_close(x, expected_x, atol=3e-2, rtol=3e-2)
                        self.assertTrue(torch.equal(r, expected_r))
        self.assertFalse(generic._USE_CUDA_NORM)

    def test_changed_inputs_current_stream_and_graph(self):
        for fused in (False, True):
            for pdl in (False, True, None):
                with self.subTest(fused=fused, pdl=pdl):
                    x, r, w = _inputs(122, 5301)
                    norm = (Glm53RMSResNorm if fused else Glm53RMSNorm)(
                        w, 1e-5, enable_pdl=pdl
                    )
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        # Each invocation may mutate x/r. Restore fresh values
                        # before capture and before every graph replay.
                        for seed in (5302, 5303):
                            a, b, c = _inputs(122, seed)
                            x.copy_(a)
                            r.copy_(b)
                            w.copy_(c)
                            norm(x, r) if fused else norm(x, x)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            norm(x, r) if fused else norm(x, x)
                    torch.cuda.current_stream().wait_stream(stream)
                    for seed in (5304, 5305):
                        a, b, c = _inputs(122, seed)
                        expected_x, expected_r = _reference(a, b if fused else None, c)
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            x.fill_(float("nan"))
                            r.fill_(float("nan"))
                            w.fill_(float("nan"))
                            x.copy_(a)
                            r.copy_(b)
                            w.copy_(c)
                            graph.replay()
                            captured_x, captured_r = x.clone(), r.clone()
                        torch.cuda.current_stream().wait_stream(stream)
                        eager_a, eager_b = a.clone(), b.clone()
                        norm(eager_a, eager_b) if fused else norm(eager_a, eager_a)
                        self.assertTrue(torch.equal(captured_x, eager_a))
                        torch.testing.assert_close(
                            captured_x, expected_x, atol=3e-2, rtol=3e-2
                        )
                        if fused:
                            self.assertTrue(torch.equal(captured_r, eager_b))
                            self.assertTrue(torch.equal(captured_r, expected_r))
                    torch.cuda.synchronize()

    def test_launch_route_pdl_and_chunking(self):
        from flashinfer.norm import device_support_pdl

        adapter = importlib.import_module("rtp_llm.models_py.modules.glm53.norm")
        real_module = adapter._cuda_norm_module()
        calls = []

        class RecordedModule:
            def rmsnorm(self, out, x, weight, eps, pdl):
                calls.append(("rms", x.shape[0], pdl))
                return real_module.rmsnorm(out, x, weight, eps, pdl)

            def fused_add_rmsnorm(self, x, r, weight, eps, pdl):
                calls.append(("fused", x.shape[0], pdl))
                return real_module.fused_add_rmsnorm(x, r, weight, eps, pdl)

        for pdl in (False, True, None):
            with self.subTest(pdl=pdl):
                x, r, w = _inputs(5, 5302)
                expected_x, _ = _reference(x, None, w)
                expected_fused, expected_r = _reference(x, r, w)
                resolved = False if pdl is False else bool(device_support_pdl(x.device))
                out = torch.full_like(x, float("nan"))
                # Force three real GPU launches with small buffers. The
                # geometry test separately checks the actual uint32 boundary.
                with patch.object(
                    adapter, "_rows_per_launch", return_value=2
                ), patch.object(
                    adapter, "_cuda_norm_module", return_value=RecordedModule()
                ):
                    calls.clear()
                    self.assertIs(Glm53RMSNorm(w, 1e-5, enable_pdl=pdl)(x, out), out)
                    self.assertEqual(calls, [("rms", n, resolved) for n in (2, 2, 1)])
                    torch.testing.assert_close(out, expected_x, atol=1e-2, rtol=1e-2)
                    calls.clear()
                    result = Glm53RMSResNorm(w, 1e-5, enable_pdl=pdl)(x, r)
                    self.assertIs(result[0], x)
                    self.assertIs(result[1], r)
                    self.assertEqual(calls, [("fused", n, resolved) for n in (2, 2, 1)])
                    torch.testing.assert_close(x, expected_fused, atol=3e-2, rtol=3e-2)
                    self.assertTrue(torch.equal(r, expected_r))

    def test_geometry_empty_and_invalid_aliases(self):
        for hidden, strides in ((4096, (4096,)), (4096, (8192, 12288))):
            rows = _rows_per_launch(hidden, *strides)
            for stride in strides:
                self.assertLessEqual((rows - 1) * stride + hidden, 1 << 32)
            self.assertTrue(any(rows * stride + hidden > 1 << 32 for stride in strides))
        with self.assertRaises(ValueError):
            _rows_per_launch(4096, 1 << 32)
        x, r, w = _inputs(3, 5302)
        norm, fused = Glm53RMSNorm(w), Glm53RMSResNorm(w)
        empty = x[:0]
        self.assertEqual(norm(empty).shape, empty.shape)
        self.assertIs(norm(empty, empty), empty)
        self.assertIs(fused(empty, r[:0])[0], empty)
        original = x.clone()
        with self.assertRaises(ValueError):
            norm(x[:-1], x[1:])
        with self.assertRaises(ValueError):
            fused(x, x)
        with self.assertRaises(ValueError):
            norm(x[:, ::2])
        with self.assertRaises(ValueError):
            norm(x, torch.empty_like(x, dtype=torch.float32))
        with self.assertRaises(ValueError):
            norm(x.float())
        self.assertTrue(torch.equal(x, original))
        # Nonoverlapping row-strided input/output retain their real row pitch.
        storage = torch.empty((3, 8192), device="cuda", dtype=x.dtype)
        strided = storage[:, :4096]
        strided.copy_(x)
        expected = norm(x)
        self.assertIs(norm(strided, strided), strided)
        self.assertTrue(torch.equal(strided, expected))

    def test_target_mtp_and_draft_sites(self):
        for name in ("glm53_flash", "glm53_flash_mtp", "glm53_moe"):
            module = importlib.import_module("rtp_llm.models_py.model_desc." + name)
            self.assertIs(module.RMSResNorm, Glm53RMSResNorm)
            if name != "glm53_moe":
                self.assertIs(module.RMSNorm, Glm53RMSNorm)

    def test_mla_q_kv_constructor_sites_and_actual_widths(self):
        import flashinfer.norm as generic

        from rtp_llm.models_py.modules.hybrid import glm53_mla_attention as mla
        from rtp_llm.models_py.modules.hybrid import mla_attention as other_mla

        self.assertFalse(generic._USE_CUDA_NORM)
        self.assertIs(mla.RMSNorm, Glm53RMSNorm)
        self.assertIsNot(other_mla.RMSNorm, Glm53RMSNorm)
        config = SimpleNamespace(
            head_num=8,
            nope_head_dim=128,
            rope_head_dim=64,
            kv_lora_rank=512,
            v_head_dim=128,
            q_lora_rank=1536,
            kernel_tokens_per_block=128,
            is_sparse=False,
        )
        parallelism = SimpleNamespace(get_attn_tp_size=lambda: 1)
        for dtype in (torch.bfloat16, torch.float16):
            q_weight = torch.randn(1536, device="cuda", dtype=dtype)
            kv_weight = torch.randn(512, device="cuda", dtype=dtype)
            # Only projection allocation is stubbed. The real MLA constructor
            # must install and execute its two actual CUDA norm modules.
            with patch.object(
                mla.LinearFactory,
                "create_linear_from_weights",
                return_value=torch.nn.Identity(),
            ):
                attention = mla.MlaAttention(
                    config,
                    parallelism,
                    {
                        mla.W.mla_q_a_ln_gamma: q_weight,
                        mla.W.mla_kv_a_ln_gamma: kv_weight,
                    },
                    layer_idx=0,
                    layernorm_eps=1e-5,
                    quant_config=None,
                    hw_kernel_config=SimpleNamespace(enable_fuse_kernels=False),
                )
            for hidden, norm, weight in (
                (1536, attention.q_a_layernorm, q_weight),
                (512, attention.kv_a_layernorm, kv_weight),
            ):
                self.assertIsInstance(norm, Glm53RMSNorm)
                self.assertIs(norm.weight, weight)
                for rows in (1, 17, 122):
                    with self.subTest(dtype=dtype, hidden=hidden, rows=rows):
                        x, _, _ = _inputs(rows, 5308, dtype, hidden)
                        expected, _ = _reference(x, None, weight)
                        with patch.object(
                            generic,
                            "rmsnorm",
                            side_effect=AssertionError("generic MLA norm route"),
                        ):
                            actual = norm(x)
                        torch.testing.assert_close(
                            actual, expected, atol=1e-2, rtol=1e-2
                        )
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            captured = norm(x)
                        x.copy_(_inputs(rows, 5309, dtype, hidden)[0])
                        expected, _ = _reference(x, None, weight)
                        graph.replay()
                        torch.testing.assert_close(
                            captured, expected, atol=1e-2, rtol=1e-2
                        )
        self.assertFalse(generic._USE_CUDA_NORM)


if __name__ == "__main__":
    unittest.main()
