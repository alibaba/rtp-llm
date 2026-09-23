import unittest

import torch

from rtp_llm.models_py.modules.glm5_mega_moe.mega_nvfp4_input_packer_triton import (
    fused_pack_mega_nvfp4_inputs,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class NVFP4InputPackerTest(unittest.TestCase):
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
