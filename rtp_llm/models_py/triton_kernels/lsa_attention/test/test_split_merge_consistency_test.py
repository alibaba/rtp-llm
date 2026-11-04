"""Cross-implementation consistency: single-pass vs split+merge sparse MLA.

Both Triton paths are independent implementations of the same absorbed-MQA MLA
math over selected slots.  The split path (flash-decoding partials + fused
merge) must agree with the single-pass kernel within mixed-precision tolerance.
This test locks in the B300 cross-check recorded in
build_logs/longcat-lsa-b300-20261017/baseline-log.md (#5.45).
"""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.lsa_attention.sparse_mla_triton import (
    sparse_mla_attention_auto,
    sparse_mla_fwd,
    sparse_mla_fwd_split,
)
from rtp_llm.models_py.triton_kernels.lsa_attention.merge_triton import merge_attention_triton


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SplitMergeConsistencyTest(unittest.TestCase):
    def test_split_merge_matches_single_pass(self):
        torch.manual_seed(0)
        t, n, s, h, dq = 4, 4096, 256, 32, 576
        dev = torch.device("cuda")
        q = (torch.randn(t, h, dq, device=dev) * 0.2).bfloat16()
        kv = (torch.randn(n, dq, device=dev) * 0.2).bfloat16()
        idx = torch.stack([torch.randperm(n, device=dev)[:s] for _ in range(t)])

        o1, l1 = sparse_mla_fwd(
            q, kv, idx, kv_lora_rank=512, out_dtype=torch.float32,
            block_n=16, num_warps=4, num_stages=1,
        )
        o2, l2 = sparse_mla_fwd_split(
            q, kv, idx, kv_lora_rank=512, out_dtype=torch.float32,
            splits=2, block_n=32, num_warps=4, num_stages=2,
        )
        o2 = merge_attention_triton([(o2, l2)], out_dtype=torch.float32)[0]

        self.assertEqual(o1.shape, o2.shape)
        diff = (o1.float() - o2.float()).abs().max().item()
        self.assertLess(diff, 1e-3, f"single-pass vs split+merge maxdiff={diff:.3e}")
        self.assertTrue(torch.isfinite(o1.float()).all())
        self.assertTrue(torch.isfinite(o2.float()).all())

    def test_auto_entry_matches_both_paths(self):
        """sparse_mla_attention_auto must reproduce the path it dispatches to."""
        torch.manual_seed(1)
        t, n, s, h, dq = 4, 4096, 256, 32, 576
        dev = torch.device("cuda")
        q = (torch.randn(t, h, dq, device=dev) * 0.2).bfloat16()
        kv = (torch.randn(n, dq, device=dev) * 0.2).bfloat16()
        idx = torch.stack([torch.randperm(n, device=dev)[:s] for _ in range(t)])

        # forced single pass (T >= single_min_t)
        o_s, l_s = sparse_mla_fwd(
            q, kv, idx, kv_lora_rank=512, out_dtype=torch.float32,
            block_n=16, num_warps=4, num_stages=1,
        )
        a_s_o, a_s_l = sparse_mla_attention_auto(
            q, kv, idx, kv_lora_rank=512, single_min_t=1,
            block_n=16, num_warps=4, num_stages=1,
        )
        # The auto entry may route through sparse_fast_fwd (bit-identical) or the
        # optional FlashMLA path (numerically equivalent: out ~1e-4 bf16 rounding,
        # lse ~1e-6, see baseline-log #5.134), so compare within tolerance.
        self.assertLess((o_s - a_s_o).abs().max().item(), 2e-2)
        self.assertLess((l_s - a_s_l).abs().max().item(), 1e-4)

        # forced split+merge (T < single_min_t)
        o_p, l_p = sparse_mla_fwd_split(
            q, kv, idx, kv_lora_rank=512, out_dtype=torch.float32,
            splits=2, block_n=32, num_warps=4, num_stages=2,
        )
        p_o, p_l = merge_attention_triton([(o_p, l_p)], out_dtype=torch.float32)
        a_p_o, a_p_l = sparse_mla_attention_auto(
            q, kv, idx, kv_lora_rank=512, single_min_t=1024, splits=2,
            block_n=32, num_warps=4, num_stages=2,
        )
        # The auto entry prefers the optional FlashMLA path when runnable (SM90+),
        # which is numerically equivalent but not bit-identical to the split+merge
        # path, so accept a match with EITHER candidate implementation.
        tolerance = 2e-2
        lse_tolerance = 1e-4
        matches = [
            (o_s, l_s),   # single-pass Triton
            (p_o, p_l),   # split + fused merge
        ]
        ok = any(
            (cand_out - a_p_o).abs().max().item() < tolerance
            and (cand_lse - a_p_l).abs().max().item() < lse_tolerance
            for cand_out, cand_lse in matches
        )
        self.assertTrue(ok, "auto entry output matches neither candidate path")


    def test_union_flashmla_matches_composed(self):
        """Single-call union FlashMLA path must match the SWA+sparse+merge path."""
        from rtp_llm.models_py.triton_kernels.lsa_attention.lsa_attention_flashmla import (
            lsa_attention_flashmla,
            lsa_flashmla_available,
        )

        if not lsa_flashmla_available():
            self.skipTest("FlashMLA requires SM90+; composed path covers this device")

        from rtp_llm.models_py.triton_kernels.lsa_attention.lsa_forward import lsa_attention

        t, n = 8, 4096
        torch.manual_seed(2)
        q = (torch.randn(t, 32, 576, device="cuda") * 0.1).bfloat16()
        kv = (torch.randn(n, 576, device="cuda") * 0.1).bfloat16()
        pos = torch.arange(n - t, n, device="cuda", dtype=torch.int64)
        dyn = torch.full((t, 1008), -1, dtype=torch.int64, device="cuda")
        for r in range(t):
            p = int(pos[r])
            dyn[r] = torch.randint(16, max(17, p - 1023), (1008,), device="cuda")

        o_u, l_u = lsa_attention_flashmla(q, kv, pos, dyn[:, None, :], kv_lora_rank=512,
                                          out_dtype=torch.float32)
        o_c, l_c = lsa_attention(q, kv, pos, dyn[:, None, :], kv_lora_rank=512,
                                 out_dtype=torch.float32, splits=1, overlap=False)
        self.assertLess((o_u - o_c).abs().max().item(), 2e-2)
        self.assertLess((l_u - l_c).abs().max().item(), 1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)