"""Exercise the installed native MQA API against independent FP32 arithmetic."""

import unittest

import torch

from rtp_llm.models_py.modules.base.cuda.fp8_indexer_score import NativeFp8IndexerScore
from rtp_llm.ops import compute_ops


class NativeFp8IndexerScoreTest(unittest.TestCase):
    def test_sparse_cache_ue8m0_is_opt_in(self):
        latent = torch.arange(3 * 512, dtype=torch.float32, device="cuda").reshape(
            3, 512
        )
        latent = (latent.remainder(127) - 63).to(torch.bfloat16)
        suffix = torch.arange(3 * 64, dtype=torch.bfloat16, device="cuda").reshape(
            3, 64
        )
        slots = torch.tensor([128, 129, -1], dtype=torch.int64, device="cuda")
        scale = torch.ones(1, dtype=torch.float32, device="cuda")
        maximum = latent[:2].float().reshape(2, 4, 128).abs().amax(-1)
        base_scale = maximum.div(448).clamp_min(torch.finfo(torch.float32).tiny)
        for power_of_two in (False, True):
            with self.subTest(power_of_two=power_of_two):
                cache = torch.zeros((2, 128, 656), dtype=torch.uint8, device="cuda")
                kwargs = {"use_ue8m0": True} if power_of_two else {}
                compute_ops.concat_and_cache_mla(
                    latent, suffix, cache, slots, "fp8_ds_mla", scale, **kwargs
                )
                expected_scale = (
                    base_scale.log2().ceil().exp2() if power_of_two else base_scale
                )
                actual_scale = cache[1, :2, 512:528].contiguous().view(torch.float32)
                torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
                actual_suffix = cache[1, :2, 528:].contiguous().view(torch.bfloat16)
                torch.testing.assert_close(actual_suffix, suffix[:2], rtol=0, atol=0)
                self.assertEqual(torch.count_nonzero(cache[0]).item(), 0)
                self.assertEqual(torch.count_nonzero(cache[1, 2:]).item(), 0)

    def test_dense_score_and_changed_input_graph(self):
        score = NativeFp8IndexerScore()
        query = torch.ones((4, 32, 128), device="cuda").to(torch.float8_e4m3fn)
        key = torch.ones((64, 128), device="cuda").to(torch.float8_e4m3fn)
        key[1::2].copy_(-key[1::2].float())
        scales = torch.ones(64, dtype=torch.float32, device="cuda")
        weights = torch.full((4, 32), 1 / 32, dtype=torch.float32, device="cuda")
        starts = torch.tensor([0, 16, 0, 64], dtype=torch.int32, device="cuda")
        ends = torch.tensor([64, 32, 64, 64], dtype=torch.int32, device="cuda")
        positions = torch.arange(64, dtype=torch.int32, device="cuda")

        def reference():
            products = torch.einsum("mhd,nd->mhn", query.float(), key.float())
            expected = (products.relu() * weights[:, :, None]).sum(1) * scales[None, :]
            columns = torch.arange(64, device="cuda")[None, :]
            return expected.masked_fill(
                (columns < starts[:, None]) | (columns >= ends[:, None]), -float("inf")
            )

        def run():
            return score.dense(query, key, scales, weights, starts, ends, positions)

        run()  # Materialize native JIT specializations before capture.
        actual = run()
        torch.testing.assert_close(actual, reference(), rtol=0, atol=0)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        query.copy_(-query.float())
        starts.copy_(torch.tensor([8, 0, 32, 0], dtype=torch.int32, device="cuda"))
        ends.copy_(torch.tensor([24, 64, 48, 0], dtype=torch.int32, device="cuda"))
        graph.replay()
        torch.testing.assert_close(captured, reference(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
