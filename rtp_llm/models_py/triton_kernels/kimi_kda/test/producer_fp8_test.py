"""Compare fused producers against the existing CUDA producers and quantizer."""

import os
import unittest

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_fp8_kernels import (
    gather_fp8_prefix,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_prefix_fp8_producer import (
    Fp8MlaPrefixGather,
)
from rtp_llm.models_py.modules.kimi_k3.fp8_producers import KdaOutputNorm
from rtp_llm.models_py.triton_kernels.kimi_kda.attn_res import kimi_k3_attn_res
from rtp_llm.models_py.triton_kernels.kimi_kda.attn_res_fp8 import kimi_k3_attn_res_fp8
from rtp_llm.models_py.triton_kernels.kimi_kda.fp8_producers import (
    kda_output_fp8,
    rmsnorm_fp8,
    sigmoid_gate_fp8,
)


def quant(x):
    return sgl_per_token_group_quant_fp8(
        x.contiguous(),
        128,
        eps=1.0e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )


class ProducerTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        torch.manual_seed(901)

    def check(self, actual, original):
        a, s = quant(original)
        torch.testing.assert_close(actual.values.float(), a.float(), atol=0, rtol=0)
        torch.testing.assert_close(actual.scales, s, atol=0, rtol=0)
        if actual.bf16 is not None:
            torch.testing.assert_close(actual.bf16, original, atol=0, rtol=0)

    def test_rms(self):
        for m in (1, 3, 4, 17, 257):
            for k in (512, 1536, 7168):
                x = torch.randn(m, k + 64, device="cuda").bfloat16()[:, :k]
                w = torch.randn(k, device="cuda").bfloat16()
                for eps in (1.0e-6, 1.0e-5):
                    with self.subTest(m=m, k=k, eps=eps):
                        # Only current KV-B inputs retain BF16 cache data (K=512).
                        self.check(
                            rmsnorm_fp8(x, w, eps, retain_bf16=(k == 512)),
                            RMSNorm(w, eps)(x.contiguous()),
                        )

    def test_activation_lifetime_and_row_padding(self):
        x = torch.randn(7, 512, device="cuda", dtype=torch.bfloat16)
        y = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
        w = torch.ones(512, device="cuda", dtype=torch.bfloat16)
        a = rmsnorm_fp8(x, w, 1.0e-5, retain_bf16=True)
        b = rmsnorm_fp8(y, w, 1.0e-5, retain_bf16=True)
        reference = RMSNorm(w, 1.0e-5)(x)
        self.check(a, reference)
        self.check(b, RMSNorm(w, 1.0e-5)(y))
        self.check(a.narrow_rows(1, 3), reference[1:4])
        padded = a.pad_rows(8)
        torch.testing.assert_close(
            padded.values[:7].float(), a.values.float(), atol=0, rtol=0
        )
        torch.testing.assert_close(padded.scales[:7], a.scales, atol=0, rtol=0)
        self.assertEqual(padded.values[7].float().count_nonzero().item(), 0)
        self.assertTrue(torch.all(padded.scales[7] == 0x7F7F7F7F).item())

    def test_gate(self):
        for m in (1, 3, 17):
            for k in (1536, 3072, 6144, 12288):
                x, g = [torch.randn(m, k, device="cuda").bfloat16() for _ in range(2)]
                with self.subTest(m=m, k=k):
                    self.check(sigmoid_gate_fp8(x, g), x * torch.sigmoid(g))

    def test_kda(self):
        for m in (1, 4, 257):
            for heads in (12, 24, 48, 96):
                x = torch.randn(1, m, heads, 128, device="cuda").bfloat16()
                g = torch.randn(1, m, heads, 256, device="cuda").bfloat16()[..., :128]
                w = torch.randn(128, device="cuda").bfloat16()
                for mode in ("prefill", "decode"):
                    with self.subTest(m=m, heads=heads, mode=mode):
                        actual = kda_output_fp8(x, g, w, 1.0e-5, mode=mode)
                        reference = KdaOutputNorm(w, 1.0e-5)(x, g, mode).reshape(
                            m, heads * 128
                        )
                        try:
                            self.check(actual, reference)
                        except AssertionError:
                            artifact = os.environ.get("K3_PRODUCER_FAILURE_ARTIFACT")
                            if artifact:
                                torch.save(
                                    dict(
                                        x=x,
                                        gate=g,
                                        weight=w,
                                        mode=mode,
                                        reference=reference,
                                        values=actual.values,
                                        scale=actual.scales,
                                    ),
                                    artifact,
                                )
                            raise

    def test_attnres(self):
        for m in (1, 4, 257):
            x = torch.randn(m, 7168, device="cuda").bfloat16()
            bank = torch.randn(m, 3, 7168, device="cuda").bfloat16()
            w, p, ow = [torch.randn(7168, device="cuda").bfloat16() for _ in range(3)]
            for n in (0, 2):
                for delta in (None, torch.randn_like(x)):
                    xa, xb, ba, bb = x.clone(), x.clone(), bank.clone(), bank.clone()
                    kw = dict(
                        output_norm_weight=ow,
                        output_norm_eps=1.0e-5,
                        delta=delta,
                        num_blocks=n,
                        block_write_idx=n,
                    )
                    actual = kimi_k3_attn_res_fp8(xa, ba, w, p, 1.0e-5, **kw)
                    original = kimi_k3_attn_res(xb, bb, w, p, 1.0e-5, **kw)
                    self.check(actual, original)
                    torch.testing.assert_close(xa, xb, atol=0, rtol=0)
                    torch.testing.assert_close(ba, bb, atol=0, rtol=0)

    def test_prefix(self):
        cache = (
            torch.randn(8, 129, 576, device="cuda")
            .bfloat16()
            .to(torch.float8_e4m3fn)[:, :128]
        )
        pages = torch.tensor([5, 1, 7], device="cuda", dtype=torch.int32)
        info = torch.tensor(
            [[0, 133, 0, 2], [1, 17, 2, 1]], device="cuda", dtype=torch.int32
        )
        qi = torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32)
        new = torch.randn(5, 576, device="cuda").bfloat16()
        c, rope = new[:, :512], new[:, 512:]
        out = torch.empty(155, 512, device="cuda", dtype=torch.bfloat16)
        r = torch.empty(155, 64, device="cuda", dtype=torch.bfloat16)
        ref_r = torch.empty_like(r)
        for scale in (0.5, 1.0, 1.7):
            gather_fp8_prefix(
                out, ref_r, c, rope, cache, pages, info, qi, 128, scale=scale
            )
            self.check(
                Fp8MlaPrefixGather(scale)(out, r, c, rope, cache, pages, info, qi, 128),
                out,
            )
            torch.testing.assert_close(r, ref_r, atol=0, rtol=0)

    def test_prefix_bf16_cache(self):
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        cache = torch.randn(8, 128, 576, device="cuda", dtype=torch.bfloat16)
        pages = torch.tensor([5, 1, 7], device="cuda", dtype=torch.int32)
        info = torch.tensor(
            [[0, 133, 0, 2], [1, 17, 2, 1]], device="cuda", dtype=torch.int32
        )
        qi = torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32)
        new = torch.randn(5, 576, device="cuda", dtype=torch.bfloat16)
        c, rope = new[:, :512], new[:, 512:]
        out = torch.empty(155, 512, device="cuda", dtype=torch.bfloat16)
        ref_r = torch.empty(155, 64, device="cuda", dtype=torch.bfloat16)
        rtp_llm_ops.reuse_kv_cache_indexed_batched(
            out, ref_r, c.contiguous(), rope.contiguous(), cache, pages, info, qi, 128
        )
        actual_r = torch.empty(155, 128, device="cuda", dtype=torch.bfloat16)[:, ::2]
        actual = Fp8MlaPrefixGather()(
            out, actual_r, c, rope, cache, pages, info, qi, 128
        )
        self.check(actual, out)
        torch.testing.assert_close(actual_r, ref_r, atol=0, rtol=0)

    def test_independent_graphs_on_two_streams(self):
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        inputs = []
        for factor in (1.0, 8.0):
            x = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
            w = torch.full((512,), factor, device="cuda", dtype=torch.bfloat16)
            for _ in range(10):
                rmsnorm_fp8(x, w, 1.0e-5, retain_bf16=True)
            inputs.append((x, w))
        torch.cuda.synchronize()
        graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
        results = []
        for graph, stream, (x, w) in zip(graphs, streams, inputs):
            with torch.cuda.graph(graph, stream=stream):
                result = rmsnorm_fp8(x, w, 1.0e-5, retain_bf16=True)
            results.append(result)
        for step in range(4):
            for x, w in inputs:
                x.normal_().mul_(step + 1)
            current = torch.cuda.current_stream()
            for graph, stream in zip(graphs, streams):
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    graph.replay()
            for stream in streams:
                current.wait_stream(stream)
            for actual, (x, w) in zip(results, inputs):
                self.check(actual, RMSNorm(w, 1.0e-5)(x))

    def test_graph(self):
        x = torch.randn(4, 512, device="cuda").bfloat16()
        w = torch.ones(512, device="cuda", dtype=torch.bfloat16)
        for _ in range(10):
            rmsnorm_fp8(x, w, 1.0e-5)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = rmsnorm_fp8(x, w, 1.0e-5)
        for v in (0.0, 1.0e-8, 3.14, -400.0):
            x.fill_(v)
            graph.replay()
            self.check(result, RMSNorm(w, 1.0e-5)(x))


if __name__ == "__main__":
    unittest.main()
