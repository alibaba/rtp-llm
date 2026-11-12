"""SM120 cases split out of test_ep_kernels.py.
These cases live in this sibling file, leaving the original test file
untouched."""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.moe.ep_kernels import ep_gather, ep_scatter


class TestRecomputeTopkIdsSentinelSm120(unittest.TestCase):
    """Sibling of ``TestRecomputeTopkIdsSentinel`` carrying the SM120 cases."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.device = torch.device("cuda", torch.cuda.current_device())

    def test_ep_gather_masks_partial_hidden_tail(self) -> None:
        """Hidden widths below/above 512 must not read or write past the row."""
        for hidden_size in (128, 513):
            with self.subTest(hidden_size=hidden_size):
                source = torch.arange(
                    5 * hidden_size,
                    dtype=torch.float32,
                    device=self.device,
                ).reshape(5, hidden_size)
                source = (source.remainder(31) / 16).to(torch.bfloat16)
                expert_ids = torch.tensor(
                    [[0, 1], [2, -1], [3, 4]],
                    dtype=torch.int32,
                    device=self.device,
                )
                source_rows = torch.tensor(
                    [[0, 1], [2, -1], [4, 3]],
                    dtype=torch.int32,
                    device=self.device,
                )
                weights = torch.tensor(
                    [[0.25, 0.75], [1.0, 9.0], [0.4, 0.6]],
                    dtype=torch.float32,
                    device=self.device,
                )
                output = torch.empty(
                    (3, hidden_size), dtype=torch.float32, device=self.device
                )

                ep_gather(source, expert_ids, weights, source_rows, output)

                expected = torch.stack(
                    (
                        source[0].float() * 0.25 + source[1].float() * 0.75,
                        source[2].float(),
                        source[4].float() * 0.4 + source[3].float() * 0.6,
                    )
                )
                torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    def test_scatter_gather_strided_topk_views(self):
        rows, hidden, experts = 3, 128, 2
        ids = torch.tensor(
            [[0, 1], [1, -1], [0, 1]], device=self.device, dtype=torch.int32
        )
        weights = torch.tensor(
            [[0.25, 0.75], [1.0, 0.0], [0.6, 0.4]], device=self.device
        )
        x = torch.randn(rows, hidden, device=self.device)

        def strided(tensor):
            storage = torch.full((rows, 6), -99, device=self.device, dtype=tensor.dtype)
            storage[:, 1::3] = tensor
            return storage[:, 1::3]

        counts = torch.tensor([2, 3], device=self.device, dtype=torch.int32)
        for use_strides in (False, True):
            topk = strided(ids) if use_strides else ids
            topk_weights = strided(weights) if use_strides else weights
            output_index = torch.full_like(ids, -1)
            if use_strides:
                output_index = strided(output_index)
            scattered = torch.empty((256, hidden), device=self.device)
            scale = torch.ones((rows, 1), device=self.device)
            scattered_scale = torch.empty((256, 1), device=self.device)
            starts = torch.zeros(experts, device=self.device, dtype=torch.int32)
            m_indices = torch.empty(256, device=self.device, dtype=torch.int32)
            ep_scatter(
                x,
                scale,
                topk,
                counts,
                starts,
                scattered,
                scattered_scale,
                m_indices,
                output_index,
            )
            output = torch.empty_like(x)
            ep_gather(scattered, topk, topk_weights, output_index, output)
            torch.testing.assert_close(output, x, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
