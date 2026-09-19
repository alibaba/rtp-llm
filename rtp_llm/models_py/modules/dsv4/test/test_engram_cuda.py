"""Run on an explicitly reserved GPU; covers pinned host reads in graph replay."""

import unittest
import uuid

import torch

from rtp_llm.models_py.modules.dsv4.engram import (
    Engram,
    EngramLayout,
    HostEngramEmbedding,
    NgramHashState,
    _PinnedSharedTable,
    gated_engram_residual,
)
from rtp_llm.models_py.modules.dsv4.utils import V41MXFP8Linear


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class EngramCudaTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        config = dict(
            engram_layer_ids=[1, 14],
            engram_num_embeddings=[4000, 4000],
            engram_max_ngram_size=4,
            engram_n_heads=4,
            engram_head_dim=32,
            engram_compressed_vocab_size=32,
            engram_pad_token_id=2,
            engram_vocab_size=13,
        )
        self.layout = EngramLayout(config)
        self.hash = NgramHashState(
            self.layout, token_map=list(range(32)), device="cuda:0"
        )
        self.weight = torch.randn(4000, 32).to(torch.float8_e4m3fn)
        self.scales = torch.randint(125, 130, (4000, 1), dtype=torch.uint8)
        self.pinned = _PinnedSharedTable(
            self.weight, self.scales, "test-" + uuid.uuid4().hex, "cuda:0"
        )
        self.embedding = HostEngramEmbedding(
            self.pinned.weight, self.pinned.scales, pinned=self.pinned
        )

    def test_hash_and_host_lookup_capture_replay(self):
        self.assertTrue(self.pinned.storage.is_pinned())
        windows = torch.tensor(
            [[7, 6, 5, 4], [9, 8, -1, -1]], dtype=torch.int32, device="cuda"
        )
        dead = windows == 6

        def forward():
            return self.embedding(self.hash(windows, dead)[:, 0], "cuda")

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = forward()
        windows.copy_(torch.tensor([[1, 0, -1, -1], [17, 16, 15, 14]], device="cuda"))
        dead.copy_(windows == 16)
        graph.replay()
        torch.cuda.synchronize()
        expected_hashes = self.hash(windows.cpu(), dead.cpu())
        torch.testing.assert_close(self.hash(windows, dead).cpu(), expected_hashes)
        expected_rows = HostEngramEmbedding(self.weight, self.scales)(
            expected_hashes[:, 0], "cpu"
        )
        torch.testing.assert_close(output.cpu(), expected_rows, rtol=0, atol=0)

    def test_gated_engram_residual_native_matches_reference(self):
        # Production shape (hc=4, dim=5120): the fused Triton kernel (inference
        # mode) must match the torch reference (grad mode keeps the reference
        # path) bitwise for >= 99.9% of elements. The few differing elements
        # are either 1-ulp bf16 rounding flips (fp32 summation order) or
        # catastrophic-cancellation noise (h + gate*value at bf16 input
        # quantization level); both are bounded: no element may exceed 1 ulp
        # AND 1e-5 absolute difference at once.
        for index, tokens in enumerate((1, 33, 257)):
            torch.manual_seed(57 + index)
            hidden = torch.randn(tokens, 4, 5120, device="cuda").bfloat16()
            kv = torch.randn(tokens, 5 * 5120, device="cuda").bfloat16()
            q_weight = torch.randn(4, 5120, device="cuda").bfloat16()
            k_weight = torch.randn(4, 5120, device="cuda").bfloat16()
            mask = torch.rand(tokens, device="cuda") < 0.8
            for token_mask in (None, mask):
                expected = gated_engram_residual(
                    hidden, kv, q_weight, k_weight, 1e-20, token_mask
                )
                with torch.inference_mode():
                    actual = gated_engram_residual(
                        hidden, kv, q_weight, k_weight, 1e-20, token_mask
                    )
                ai = actual.view(torch.int16).to(torch.int32)
                bi = expected.view(torch.int16).to(torch.int32)
                ulp = (
                    torch.where(ai >= 0, ai, -32768 - ai)
                    - torch.where(bi >= 0, bi, -32768 - bi)
                ).abs()
                diff = (actual.float() - expected.float()).abs()
                exact = (actual == expected).float().mean().item()
                bad = (ulp > 1) & (diff > 1e-5)
                self.assertGreaterEqual(exact, 0.999, (tokens, exact))
                self.assertEqual(int(bad.sum()), 0, (tokens, int(bad.sum())))

    def test_gated_engram_residual_native_mask_suppression(self):
        torch.manual_seed(63)
        hidden = torch.ones(2, 4, 5120, device="cuda").bfloat16()
        kv = torch.ones(2, 5 * 5120, device="cuda").bfloat16()
        qk = torch.ones(4, 5120, device="cuda").bfloat16()
        mask = torch.tensor([False, True], device="cuda")
        with torch.inference_mode():
            result = gated_engram_residual(hidden, kv, qk, qk, 1e-20, mask)
        self.assertTrue(torch.equal(result[0], hidden[0]))
        self.assertTrue(torch.all(result[1] > hidden[1]).item())

    def test_gated_engram_residual_native_capture_replay(self):
        # The fused kernel is captured by the decode CUDA graph: replay must
        # track input changes exactly like the eager kernel path.
        torch.manual_seed(71)
        hidden = torch.randn(2, 4, 5120, device="cuda").bfloat16()
        kv = torch.randn(2, 5 * 5120, device="cuda").bfloat16()
        q_weight = torch.randn(4, 5120, device="cuda").bfloat16()
        k_weight = torch.randn(4, 5120, device="cuda").bfloat16()

        def call():
            return gated_engram_residual(hidden, kv, q_weight, k_weight, 1e-20)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                call()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream), torch.inference_mode():
            output = call()
        hidden.copy_(torch.randn_like(hidden))
        kv.copy_(torch.randn_like(kv))
        with torch.inference_mode():
            expected = call()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_complete_engram_capture_changes_input_and_preserves_mask(self):
        projection = V41MXFP8Linear(
            (torch.randn(640, 384, device="cuda") * 0.02).to(torch.float8_e4m3fn),
            torch.ones(20, 12, device="cuda").to(torch.float8_e8m0fnu),
        )
        model = Engram(
            self.layout,
            0,
            self.embedding,
            projection,
            torch.randn(4, 128, dtype=torch.bfloat16, device="cuda"),
            torch.randn(4, 128, dtype=torch.bfloat16, device="cuda"),
            1e-20,
        )
        hidden = torch.randn(2, 4, 128, dtype=torch.bfloat16, device="cuda")
        windows = torch.tensor(
            [[7, 6, 5, 4], [-1, -1, -1, -1]], dtype=torch.int32, device="cuda"
        )

        def forward():
            return model(hidden, self.hash(windows)[:, 0], windows[:, 0] >= 0)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = forward()
        hidden.copy_(torch.randn_like(hidden))
        windows[0].copy_(torch.tensor([17, 16, 15, 14], device="cuda"))
        expected = forward()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(output[1], hidden[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
