"""Numerical gate for the runtime-selected PPU sparse attention wheel."""

import unittest

import torch
from rtp_llm.device.device_type import is_ppu

@unittest.skipUnless(
    is_ppu(),
    "requires a PPU",
)
class FlashMlaRuntimeTest(unittest.TestCase):
    def test_sparse_prefill_with_sink_and_masked_indices(self):
        from flash_mla import flash_mla_sparse_fwd

        torch.manual_seed(1434)
        rows, heads, dim, tokens, topk = 4, 64, 512, 256, 128
        q = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16) * 0.1
        kv = torch.randn(tokens, 1, dim, device="cuda", dtype=torch.bfloat16) * 0.1
        sink = torch.linspace(-1, 1, heads, device="cuda")
        indices = torch.randint(tokens, (rows, topk), device="cuda", dtype=torch.int32)
        indices[:, -32:] = -1
        selected = kv[indices.clamp_min(0).long(), 0].float()
        logits = torch.einsum("shd,skd->shk", q.float(), selected) * dim**-0.5
        logits.masked_fill_(indices[:, None, :] < 0, -torch.inf)
        normalizer = torch.logsumexp(
            torch.cat((logits, sink[None, :, None].expand(rows, -1, -1)), dim=-1),
            dim=-1,
            keepdim=True,
        )
        expected = torch.einsum("shk,skd->shd", (logits - normalizer).exp(), selected)
        output, _, _ = flash_mla_sparse_fwd(
            q, kv, indices[:, None, :], dim**-0.5, attn_sink=sink
        )
        torch.cuda.synchronize()
        self.assertEqual(output.shape, q.shape)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(bool(output.isfinite().all()))
        relative_mean = (output.float() - expected).abs().mean() / expected.abs().mean()
        self.assertLess(float(relative_mean), 5e-3)


if __name__ == "__main__":
    unittest.main()
