"""Natural-log paged-KV LSE kernel UT (sink-fallback correction input).

Pins the Triton kernel against a direct torch reference over the RTP-LLM
paged layout [blocks, 2, kv_heads, page_size, head_dim] with multi-page
rows, partial last pages, and GQA head grouping.
"""

import math
import unittest

import torch

from rtp_llm.models_py.triton_kernels.attention.paged_kv_lse import paged_kv_lse


class PagedKVLseTest(unittest.TestCase):
    def _reference(self, q, kv, indptr, indices, last_len, kv_heads, ps, scale):
        rows, H, D = q.shape
        out = torch.zeros(rows, H, dtype=torch.float32)
        for r in range(rows):
            pages = indices[indptr[r]:indptr[r + 1]]
            n_tok = (len(pages) - 1) * ps + int(last_len[r])
            # gather keys: [n_tok, H_kv, D]
            keys = []
            for pi, page in enumerate(pages):
                t = ps if pi < len(pages) - 1 else int(last_len[r])
                keys.append(kv[page, 0, :, :t, :])
            keys = torch.cat(keys, dim=1)  # [H_kv, n_tok, D]
            for h in range(H):
                kv_h = h * kv_heads // H
                logits = torch.einsum(
                    "d,nd->n", q[r, h].float(), keys[kv_h].float()
                ) * scale
                out[r, h] = torch.logsumexp(logits, dim=-1)
        return out

    def test_lse_matches_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        torch.manual_seed(0)
        dev = "cuda"
        for H, H_kv, D, PS in ((4, 2, 64, 16), (8, 8, 32, 8)):
            rows, blocks = 6, 12
            q = torch.randn(rows, H, D, dtype=torch.bfloat16, device=dev)
            kv = torch.randn(blocks, 2, H_kv, PS, D, dtype=torch.bfloat16, device=dev)
            # rows use 1-3 pages, last page partially filled
            spans = torch.tensor([0, 1, 3, 4, 7, 9, 12], dtype=torch.int32, device=dev)
            indices = torch.arange(blocks, dtype=torch.int32, device=dev)
            last_len = torch.tensor(
                [PS, PS, 5, PS, 7, PS], dtype=torch.int32, device=dev
            )
            scale = 1.0 / math.sqrt(D)
            got = paged_kv_lse(
                q, kv, spans, indices, last_len, H, H_kv, D, PS, scale
            )
            ref = self._reference(
                q.cpu(), kv.cpu(), spans.cpu(), indices.cpu(), last_len.cpu(),
                H_kv, PS, scale,
            )
            diff = (got.cpu() - ref).abs().max().item()
            print(f"H={H} kv={H_kv} D={D} PS={PS}: maxdiff={diff:.5f}")
            self.assertLess(diff, 5e-2)

    def test_sink_correction_end_to_end(self):
        """y' = y * sigmoid(lse - sink) vs the direct sink-softmax reference."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        torch.manual_seed(1)
        dev = "cuda"
        H, D, PS, N = 2, 64, 16, 37  # 3 pages (16+16+5)
        q = torch.randn(3, H, D, dtype=torch.bfloat16, device=dev)
        kv = torch.randn(3, 2, H, PS, D, dtype=torch.bfloat16, device=dev)
        spans = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=dev)
        indices = torch.arange(3, dtype=torch.int32, device=dev)
        last_len = torch.tensor([16, 16, 5], dtype=torch.int32, device=dev)
        scale = 1.0 / math.sqrt(D)
        sinks = torch.randn(H, dtype=torch.float32, device=dev) * 4

        lse = paged_kv_lse(q, kv, spans, indices, last_len, H, H, D, PS, scale)
        # plain attention output
        keys = torch.cat([kv[i, 0, :, : (16 if i < 2 else 5), :] for i in range(3)], dim=1)
        vals = torch.cat([kv[i, 1, :, : (16 if i < 2 else 5), :] for i in range(3)], dim=1)
        for r in range(3):
            for h in range(H):
                logits = torch.einsum(
                    "d,nd->n", q[r, h].float(), keys[h].float()
                ) * scale
                y0 = torch.einsum(
                    "n,nd->d", torch.softmax(logits, -1), vals[h].float()
                )
                corrected = y0 * torch.sigmoid(lse[r, h] - sinks[h])
                full = torch.cat([logits, sinks[h:h + 1]])
                probs = torch.softmax(full, -1)[:-1]
                ref = torch.einsum("n,nd->d", probs, vals[h].float())
                self.assertLess(
                    (corrected - ref).abs().max().item(), 1e-2,
                    f"row {r} head {h}",
                )


if __name__ == "__main__":
    unittest.main()
