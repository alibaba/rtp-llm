"""Native FP4 scoring, page addressing and replay with DeepGEMM."""

import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp4.native import (
    paged_score,
    prefill_score,
    provider,
)
from rtp_llm.models_py.modules.dsv4.fp4.quantization import (
    dequantize_rows,
    quantize_q_rope,
    quantize_q_rope_packed,
    quantize_rows,
)
from rtp_llm.models_py.modules.dsv4.fp4.score import paged_score as reference_score
from rtp_llm.models_py.modules.dsv4.test.test_fp4_indexer_contract import (
    write_reference_pages,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class Fp4NativeIndexerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (12, 0):
            raise unittest.SkipTest("requires SM120")
        cache = tempfile.TemporaryDirectory(prefix="fp4-native-jit-")
        cls.addClassCleanup(cache.cleanup)
        cache_env = patch.dict(os.environ, DG_JIT_CACHE_DIR=cache.name)
        cache_env.start()
        cls.addClassCleanup(cache_env.stop)
        provider()

    def test_paged_scores_and_updated_graph_inputs(self):
        torch.manual_seed(20261014)
        shapes = [(2, n, h, 129) for n in (1, 2, 3, 4) for h in (16, 32, 64)]
        shapes += [
            (b, n, 64, k) for b in (1, 2, 7) for n in (1, 3, 4) for k in (7999, 8000)
        ]
        for batch, queries, heads, keys in shapes:
            with self.subTest(batch=batch, queries=queries, heads=heads, keys=keys):
                self.check_shape(batch, queries, heads, keys)

    def test_unsupported_heads(self):
        q = torch.zeros((1, 8, 64), device="cuda", dtype=torch.uint8)
        scales = torch.zeros((1, 8, 4), device="cuda", dtype=torch.uint8)
        weights = torch.zeros((1, 8), device="cuda", dtype=torch.float32)
        k = torch.zeros((1, 64), device="cuda", dtype=torch.uint8)
        ks = torch.zeros((1, 4), device="cuda", dtype=torch.uint8)
        lengths = torch.ones((1,), device="cuda", dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "16, 32 or 64 heads"):
            prefill_score((q, scales), weights, k, ks, lengths - 1, lengths)
        with self.assertRaisesRegex(ValueError, "16, 32 or 64 heads"):
            paged_score(
                (q.unsqueeze(0), scales.unsqueeze(0)),
                weights,
                torch.zeros((1, 64, 68), device="cuda", dtype=torch.uint8),
                lengths.view(1, 1),
                lengths.view(1, 1),
                block_size=64,
                max_ctx_len=1,
            )

    def test_prefill_scores(self):
        torch.manual_seed(20261015)
        for heads in (16, 32, 64):
            with self.subTest(heads=heads):
                q, qs = quantize_rows(torch.randn(7, heads, 128, device="cuda"))
                k, ks = quantize_rows(torch.randn(129, 128, device="cuda"), rotate=True)
                weights = torch.randn(7, heads, device="cuda")
                starts = torch.arange(7, device="cuda", dtype=torch.int32)
                ends = torch.full_like(starts, 129)
                ends[0] = 0
                actual = prefill_score((q, qs), weights, k, ks, starts, ends)
                qd = dequantize_rows(q, qs).cpu().double()
                kd = dequantize_rows(k, ks).cpu().double()
                expected = torch.einsum("mhd,kd->mhk", qd, kd).relu()
                expected = (expected * weights.cpu().double()[:, :, None]).sum(1)
                mask = (torch.arange(129)[None] >= starts.cpu()[:, None]) & (
                    torch.arange(129)[None] < ends.cpu()[:, None]
                )
                torch.testing.assert_close(
                    actual.cpu()[mask].double(), expected[mask], rtol=1e-4, atol=3e-3
                )

    def check_shape(self, batch, queries, heads, keys):
        rows = batch * queries
        q = torch.randn(
            (batch, queries, heads, 128), device="cuda", dtype=torch.bfloat16
        )
        weights = torch.randn(q.shape[:-1], device="cuda", dtype=torch.bfloat16)
        angles = torch.randn((rows, 32), device="cuda")
        freqs = torch.polar(torch.ones_like(angles), angles)
        scale = 128**-0.5 * heads**-0.5
        k, ks = quantize_rows(torch.randn(keys, 128, device="cuda"), rotate=True)
        pool = write_reference_pages(k, ks)
        blocks = pool.shape[0] - 1
        pool = torch.cat((pool[:1], pool[1:].flip(0)), dim=0)
        table = torch.arange(blocks, 0, -1, dtype=torch.int32, device="cuda").repeat(
            batch, 1
        )
        lengths = torch.full((batch, queries), keys, dtype=torch.int32, device="cuda")
        # The scheduler uses the last query length; RTP positions are consecutive.
        boundaries = [0, 1, 63, 64, 65, 128, keys]
        for request in range(batch):
            base = keys if request == batch - 1 else boundaries[request]
            for step in range(queries):
                lengths[request, step] = max(0, min(keys, (base * 4 - 1 + step) // 4))

        def reference():
            restored, folded = quantize_q_rope(
                q, weights, freqs, 64, weight_scale=scale
            )
            return reference_score(
                restored,
                folded.view(rows, heads),
                pool,
                table,
                lengths,
                block_size=64,
                max_ctx_len=keys,
            )

        def native():
            payload, folded, scales = quantize_q_rope_packed(
                q, weights, freqs, 64, weight_scale=scale
            )
            return paged_score(
                (payload, scales),
                folded.view(rows, heads),
                pool,
                table,
                lengths,
                block_size=64,
                max_ctx_len=keys,
            )

        actual = native()
        self.check_scores(actual, reference(), lengths)
        if keys == 129:
            payload, folded, scales = quantize_q_rope_packed(
                q, weights, freqs, 64, weight_scale=scale
            )
            q_cpu = dequantize_rows(payload, scales).cpu().double()
            k_cpu = dequantize_rows(k, ks).cpu().double()
            oracle = torch.einsum("bnhd,kd->bnhk", q_cpu, k_cpu).relu()
            oracle = (oracle * folded.cpu().double().unsqueeze(-1)).sum(-2)
            mask = torch.arange(keys)[None] < lengths.cpu().reshape(rows, 1)
            torch.testing.assert_close(
                actual.cpu()[mask].double(),
                oracle.reshape(rows, keys)[mask],
                rtol=1e-4,
                atol=3e-3,
            )
        if keys == 8000:
            lengths.fill_(keys)
            native()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = native()
            q.neg_()
            weights.mul_(0.5)
            lengths.fill_(129)
            if rows > 1:
                lengths.flatten()[0] = 0
            table.copy_(
                torch.arange(1, blocks + 1, dtype=torch.int32, device="cuda").expand_as(
                    table
                )
            )
            graph.replay()
            torch.cuda.synchronize()
            fresh = native()
            mask = torch.arange(keys, device="cuda")[None] < lengths.reshape(rows, 1)
            torch.testing.assert_close(captured[mask], fresh[mask], rtol=0, atol=0)
            self.check_scores(captured, reference(), lengths)

    def check_scores(self, actual, expected, lengths):
        from rtp_llm.models_py.modules.dsv4.fp8.indexer import _run_topk_v3

        rows, width = actual.shape
        mask = torch.arange(width, device=actual.device)[None] < lengths.reshape(-1, 1)
        torch.testing.assert_close(actual[mask], expected[mask], rtol=1e-4, atol=3e-3)
        outputs = []
        for scores in (actual, expected):
            selected = torch.empty((rows, 512), device=scores.device, dtype=torch.int32)
            self.assertTrue(
                _run_topk_v3(scores, lengths.flatten(), selected, 512, width)
            )
            outputs.append(selected.sort(-1).values)
        torch.testing.assert_close(*outputs, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
