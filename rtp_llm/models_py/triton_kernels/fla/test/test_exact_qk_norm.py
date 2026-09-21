import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.prefill_fusion import prefill_fusion_scope
from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule
from rtp_llm.models_py.triton_kernels.fla.exact_qk_norm import (
    fused_l2norm_qk_exact,
    supports_exact_qk_norm,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "SM10x CUDA required",
)
class ExactQKNormTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(812)

    def test_norm_matches_existing_rounding_across_lengths_and_scales(self):
        for length in (2048, 20000, 24601, 32768, 49202):
            for scale in (0.0, 1e-4, 0.1, 10000.0):
                with self.subTest(length=length, scale=scale):
                    q = (
                        torch.randn(
                            1, length, 16, 128, device="cuda", dtype=torch.bfloat16
                        )
                        * scale
                    )
                    k = torch.randn_like(q) * scale
                    actual = fused_l2norm_qk_exact(q, k)
                    self.assertTrue(torch.equal(l2norm_fwd(q), actual[0]))
                    self.assertTrue(torch.equal(l2norm_fwd(k), actual[1]))
                    self.assertTrue(torch.isfinite(actual[0]).all())
                    self.assertTrue(torch.isfinite(actual[1]).all())

    def test_noncontiguous_token_rows(self):
        packed = torch.randn(1, 24601, 96, 128, device="cuda", dtype=torch.bfloat16)
        q, k, _ = torch.split(packed, (16, 16, 64), dim=2)
        actual = fused_l2norm_qk_exact(q, k)
        self.assertTrue(torch.equal(l2norm_fwd(q.contiguous()), actual[0]))
        self.assertTrue(torch.equal(l2norm_fwd(k.contiguous()), actual[1]))

    def test_unsupported_shapes_keep_fallback_available(self):
        q = torch.empty(1, 24, 16, 128, device="cuda", dtype=torch.bfloat16)
        self.assertFalse(supports_exact_qk_norm(q, q))
        self.assertFalse(supports_exact_qk_norm(q.cpu(), q.cpu()))
        q = torch.empty(1, 2048, 16, 128, device="cuda", dtype=torch.float16)
        self.assertFalse(supports_exact_qk_norm(q, q))

    def test_full_gdn_outputs_final_state_and_all_chunk_states(self):
        for lengths in ([24], [20000], [24601], [32768], [49202], [24, 2024, 22553]):
            for random_state in (False, True):
                with self.subTest(lengths=lengths, random_state=random_state):
                    total = sum(lengths)
                    q = (
                        torch.randn(
                            1, total, 16, 128, device="cuda", dtype=torch.bfloat16
                        )
                        * 0.1
                    )
                    k = torch.randn_like(q)
                    v = (
                        torch.randn(
                            1, total, 64, 128, device="cuda", dtype=torch.bfloat16
                        )
                        * 0.1
                    )
                    a = torch.randn(total, 64, device="cuda", dtype=torch.bfloat16)
                    b = torch.randn_like(a)
                    g, beta = fused_gdn_gating(
                        torch.randn(64, device="cuda"),
                        a,
                        b,
                        torch.randn(64, device="cuda", dtype=torch.bfloat16),
                    )
                    state = torch.zeros(len(lengths), 64, 128, 128, device="cuda")
                    if random_state:
                        state.normal_(std=0.01)
                    cu = torch.tensor(
                        [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                        device="cuda",
                        dtype=torch.int32,
                    )

                    def run(fused):
                        # Explicit reference: preserve the original two norm calls.
                        qn, kn = (q, k) if fused else (l2norm_fwd(q), l2norm_fwd(k))
                        with prefill_fusion_scope(True):
                            return chunk_gated_delta_rule(
                                qn,
                                kn,
                                v,
                                g,
                                beta,
                                initial_state=state,
                                output_final_state=True,
                                cu_seqlens=cu,
                                use_qk_l2norm_in_kernel=fused,
                            )

                    original, actual = run(False), run(True)
                    for index, (left, right) in enumerate(zip(original, actual)):
                        self.assertTrue(torch.isfinite(right).all(), index)
                        self.assertTrue(
                            torch.equal(left, right), (lengths, random_state, index)
                        )


if __name__ == "__main__":
    unittest.main()
