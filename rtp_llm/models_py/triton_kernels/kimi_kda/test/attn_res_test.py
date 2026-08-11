import unittest

import torch

from rtp_llm.models_py.triton_kernels.kimi_kda import (
    is_kimi_k3_attn_res_supported,
    kimi_k3_attn_res,
)


class KimiK3AttnResTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        torch.manual_seed(20260810)

    @staticmethod
    def _reference(
        prefix: torch.Tensor,
        blocks: torch.Tensor,
        norm_weight: torch.Tensor,
        projection_weight: torch.Tensor,
        eps: float,
        output_norm_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        candidates = torch.cat((blocks, prefix.unsqueeze(1)), dim=1)
        values = candidates.float()
        normalized = values * torch.rsqrt(
            values.square().mean(dim=-1, keepdim=True) + eps
        )
        score_weight = norm_weight.float() * projection_weight.reshape(-1).float()
        probabilities = torch.softmax(
            (normalized * score_weight).sum(dim=-1), dim=-1
        )
        output = torch.einsum("tb,tbd->td", probabilities, values).to(prefix.dtype)
        if output_norm_weight is None:
            return output
        output_float = output.float()
        normalized_output = output_float * torch.rsqrt(
            output_float.square().mean(dim=-1, keepdim=True) + eps
        )
        return output_norm_weight * normalized_output.to(prefix.dtype)

    def test_block_counts_with_and_without_output_norm(self) -> None:
        eps = 1e-6
        hidden_size = 7168
        for num_blocks in range(1, 9):
            for apply_output_norm in (False, True):
                with self.subTest(
                    num_blocks=num_blocks,
                    apply_output_norm=apply_output_norm,
                ):
                    prefix_storage = torch.randn(
                        2,
                        hidden_size + 7,
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                    block_storage = torch.randn(
                        2,
                        num_blocks,
                        hidden_size + 7,
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                    prefix = prefix_storage[:, :hidden_size]
                    blocks = block_storage[..., :hidden_size]
                    norm_weight = torch.randn(
                        hidden_size, dtype=torch.bfloat16, device="cuda"
                    )
                    projection_weight = (
                        torch.randn(
                            hidden_size,
                            1,
                            dtype=torch.bfloat16,
                            device="cuda",
                        )
                        / hidden_size**0.5
                    )
                    output_norm_weight = (
                        torch.randn(
                            hidden_size,
                            dtype=torch.bfloat16,
                            device="cuda",
                        )
                        if apply_output_norm
                        else None
                    )

                    expected = self._reference(
                        prefix,
                        blocks,
                        norm_weight,
                        projection_weight,
                        eps,
                        output_norm_weight,
                    )
                    actual = kimi_k3_attn_res(
                        prefix,
                        blocks,
                        norm_weight,
                        projection_weight,
                        eps,
                        output_norm_weight,
                        eps if apply_output_norm else None,
                    )

                    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
                    self.assertTrue(actual.is_contiguous())

    def test_delta_and_preallocated_block_write(self) -> None:
        eps = 1e-6
        hidden_size = 7168
        prefix = torch.randn(2, hidden_size, dtype=torch.bfloat16, device="cuda")
        delta = torch.randn_like(prefix)
        blocks = torch.randn(
            2, 8, hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        norm_weight = torch.randn_like(prefix[0])
        projection_weight = (
            torch.randn(hidden_size, 1, dtype=torch.bfloat16, device="cuda")
            / hidden_size**0.5
        )
        output_norm_weight = torch.randn_like(prefix[0])

        expected_prefix = (prefix.float() + delta.float()).to(prefix.dtype)
        expected_blocks = blocks.clone()
        expected_blocks[:, 3].copy_(expected_prefix)
        expected = self._reference(
            expected_prefix,
            expected_blocks[:, :3],
            norm_weight,
            projection_weight,
            eps,
            output_norm_weight,
        )
        actual = kimi_k3_attn_res(
            prefix,
            blocks,
            norm_weight,
            projection_weight,
            eps,
            output_norm_weight,
            eps,
            delta,
            num_blocks=3,
            block_write_idx=3,
        )

        torch.testing.assert_close(prefix, expected_prefix, rtol=0, atol=0)
        torch.testing.assert_close(blocks[:, 3], expected_prefix, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_zero_blocks_writes_first_preallocated_slot(self) -> None:
        eps = 1e-6
        prefix = torch.randn(2, 64, dtype=torch.bfloat16, device="cuda")
        blocks = torch.empty(2, 8, 64, dtype=torch.bfloat16, device="cuda")
        norm_weight = torch.ones(64, dtype=torch.bfloat16, device="cuda")
        projection_weight = torch.ones(64, 1, dtype=torch.bfloat16, device="cuda")
        output_norm_weight = torch.randn_like(norm_weight)
        expected = self._reference(
            prefix,
            blocks[:, :0],
            norm_weight,
            projection_weight,
            eps,
            output_norm_weight,
        )

        actual = kimi_k3_attn_res(
            prefix,
            blocks,
            norm_weight,
            projection_weight,
            eps,
            output_norm_weight,
            eps,
            num_blocks=0,
            block_write_idx=0,
        )

        torch.testing.assert_close(blocks[:, 0], prefix, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_support_gate_rejects_more_than_eight_blocks(self) -> None:
        prefix = torch.randn(1, 64, dtype=torch.bfloat16, device="cuda")
        norm_weight = torch.ones(64, dtype=torch.bfloat16, device="cuda")
        projection_weight = torch.ones(64, 1, dtype=torch.bfloat16, device="cuda")
        blocks = torch.randn(
            1,
            9,
            64,
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.assertFalse(
            is_kimi_k3_attn_res_supported(
                prefix,
                blocks,
                norm_weight,
                projection_weight,
            )
        )


if __name__ == "__main__":
    unittest.main()
