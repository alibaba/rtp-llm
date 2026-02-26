# type: ignore
"""Test for layout_convert.cu: contiguous <-> masked (FP8)."""

from unittest import SkipTest, TestCase, main

import torch

try:
    from rtp_llm.ops.compute_ops import rtp_llm_ops
except ImportError:
    rtp_llm_ops = None


class LayoutConvertTest(TestCase):
    def test_roundtrip_fp8(self) -> None:
        """contiguous -> masked -> contiguous 应恢复原数据 (FP8)."""
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        if rtp_llm_ops is None or not hasattr(
            rtp_llm_ops, "convert_contiguous_to_masked"
        ):
            raise SkipTest("rtp_llm_ops or convert_contiguous_to_masked not available")
        rtp_ops = rtp_llm_ops

        total_tokens, hidden_dim = 64, 128
        num_experts, max_tokens_per_expert = 4, 32

        contiguous = torch.randn(
            total_tokens, hidden_dim, device="cuda", dtype=torch.float32
        )
        contiguous = contiguous.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        grouped_layout = torch.randint(
            0, num_experts, (total_tokens,), device="cuda", dtype=torch.int32
        )

        masked_data, mask = rtp_ops.convert_contiguous_to_masked(
            contiguous, grouped_layout, num_experts, max_tokens_per_expert
        )
        recovered = rtp_ops.convert_masked_to_contiguous(
            masked_data, grouped_layout, mask
        )

        self.assertTrue(torch.equal(recovered, contiguous))


if __name__ == "__main__":
    main()
