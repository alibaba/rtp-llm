"""Current DSV4 KV-cache lifecycle tests.

The production DSV4 path is pool-only: compressor/indexer KV tensors are
bound for a single call, optionally scattered back to framework pools, and
then cleared.  These tests cover that standalone fallback lifecycle without
depending on the external official DeepSeek source tree.
"""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.compressor import Compressor


def _make_compressor(ratio: int = 4, dim: int = 16, head_dim: int = 8) -> Compressor:
    coff = 2 if ratio == 4 else 1
    weights = {
        "ape": torch.zeros(ratio, coff * head_dim, dtype=torch.float32),
        "wkv": torch.randn(coff * head_dim, dim, dtype=torch.float32) * 0.05,
        "wgate": torch.randn(coff * head_dim, dim, dtype=torch.float32) * 0.05,
        "norm": torch.ones(head_dim, dtype=torch.bfloat16),
    }
    comp = Compressor(
        dim=dim,
        head_dim=head_dim,
        rope_head_dim=0,
        compress_ratio=ratio,
        max_batch_size=2,
        compressor_weights=weights,
    )
    comp.configure_kv_cache_shape(16)
    comp.freqs_cis = torch.empty(128, 0, dtype=torch.complex64)
    return comp


def _move_compressor(comp: Compressor, device: torch.device) -> Compressor:
    comp.ape = comp.ape.to(device)
    comp.wkv = comp.wkv.to(device)
    comp.wgate = comp.wgate.to(device)
    comp.norm.weight = comp.norm.weight.to(device)
    comp.freqs_cis = comp.freqs_cis.to(device)
    return comp


class TestDsv4KvCacheLifecycle(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("DSV4 compressor lifecycle uses CUDA-only rmsnorm op")
        self.device = torch.device("cuda:0")

    def test_decode_boundary_returns_tensor_and_clears_ephemeral_state(self):
        comp = _move_compressor(_make_compressor(ratio=4), self.device)
        x = torch.randn(2, 1, comp.dim, device=self.device, dtype=torch.bfloat16)
        start_pos = torch.tensor([3, 7], device=self.device, dtype=torch.int32)

        out = comp.forward_decode(x, start_pos)

        self.assertEqual(tuple(out.shape), (2, 1, comp.head_dim))
        self.assertIsNone(comp.kv_state)
        self.assertIsNone(comp.score_state)
        self.assertIsNone(comp.kv_cache)

    def test_decode_non_boundary_returns_none_and_clears_ephemeral_state(self):
        comp = _move_compressor(_make_compressor(ratio=4), self.device)
        x = torch.randn(2, 1, comp.dim, device=self.device, dtype=torch.bfloat16)
        start_pos = torch.tensor([0, 1], device=self.device, dtype=torch.int32)

        out = comp.forward_decode(x, start_pos)

        self.assertIsNone(out)
        self.assertIsNone(comp.kv_state)
        self.assertIsNone(comp.score_state)
        self.assertIsNone(comp.kv_cache)


if __name__ == "__main__":
    unittest.main()
