"""EP scatter must not read an offset before another warp initializes it."""

from unittest import SkipTest, TestCase, main

import torch
import triton

from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    _fwd_kernel_ep_scatter_1,
    ep_scatter,
)


class EpScatterTest(TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")

    def test_offsets_and_expert_rows_ignore_previous_buffer_contents(self):
        for experts in (1, 3, 32, 64, 129):
            with self.subTest(experts=experts):
                counts_cpu = (torch.arange(experts, device="cpu") * 5 % 14) * 128
                counts_cpu[0] = 128
                rows = int(counts_cpu.sum())
                counts = counts_cpu.to(device="cuda", dtype=torch.int32)
                expected_starts = counts.cumsum(0).to(torch.int32) - counts
                expected_ids = torch.repeat_interleave(
                    torch.arange(experts, device="cuda", dtype=torch.int32), counts
                )
                starts = torch.empty((32, experts), device="cuda", dtype=torch.int32)
                storage = torch.full(
                    (32, rows + 256), -1, device="cuda", dtype=torch.int32
                )
                for repeat in range(32):
                    # Zero is stale but in bounds, so a race fails numerically
                    # without poisoning the CUDA context for subsequent tests.
                    starts[repeat].zero_()
                    _fwd_kernel_ep_scatter_1[(experts,)](
                        counts,
                        starts[repeat],
                        storage[repeat, 128:-128],
                        num_experts=experts,
                        BLOCK_E=128,
                        BLOCK_EXPERT_NUM=triton.next_power_of_2(experts),
                        num_warps=8,
                    )
                torch.testing.assert_close(
                    starts, expected_starts.expand_as(starts), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    storage[:, 128:-128], expected_ids.expand(32, -1), rtol=0, atol=0
                )
                self.assertTrue(bool((storage[:, :128] == -1).all()))
                self.assertTrue(bool((storage[:, -128:] == -1).all()))

    def test_scatter_payload_and_atomic_offsets(self):
        for tokens, experts in ((129, 3), (4004, 64), (8004, 64)):
            with self.subTest(tokens=tokens, experts=experts):
                dim = 256
                ids = (
                    torch.arange(tokens * 2, device="cuda") % (experts + 1) - 1
                ).reshape(tokens, 2)
                counts = torch.bincount(ids[ids >= 0], minlength=experts).to(torch.int32)
                aligned = (counts + 127) // 128 * 128
                rows = int(aligned.sum())
                expected_starts = aligned.cumsum(0).to(torch.int32) - aligned
                x = (
                    torch.arange(tokens * dim, device="cuda").reshape(tokens, dim) % 16
                ).to(torch.float8_e4m3fn)
                scales = torch.arange(
                    tokens * 2, device="cuda", dtype=torch.float32
                ).reshape(tokens, 2)
                out = torch.empty((rows, dim), device="cuda", dtype=x.dtype)
                out_scales = torch.empty((rows, 2), device="cuda", dtype=scales.dtype)
                starts = torch.zeros_like(aligned)
                m_indices = torch.full((rows,), -1, device="cuda", dtype=torch.int32)
                out_indices = torch.full_like(ids, -1)
                ep_scatter(
                    x, scales, ids, aligned, starts, out, out_scales, m_indices, out_indices
                )
                valid = ids >= 0
                source_rows = torch.arange(tokens, device="cuda")[:, None].expand_as(ids)
                source_rows = source_rows[valid]
                destinations = out_indices[valid]
                torch.testing.assert_close(
                    out.float()[destinations], x.float()[source_rows], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    out_scales[destinations], scales[source_rows], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    starts, expected_starts + counts, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    m_indices[destinations].long(), ids[valid], rtol=0, atol=0
                )
                self.assertTrue(bool((out_indices[~valid] == -1).all()))

    def test_empty_experts_initialize_offsets_without_writing_rows(self):
        experts = 64
        counts = torch.zeros(experts, device="cuda", dtype=torch.int32)
        starts = torch.full_like(counts, -1)
        guard = torch.full((256,), -1, device="cuda", dtype=torch.int32)
        _fwd_kernel_ep_scatter_1[(experts,)](
            counts,
            starts,
            guard[128:128],
            num_experts=experts,
            BLOCK_E=128,
            BLOCK_EXPERT_NUM=experts,
            num_warps=8,
        )
        torch.testing.assert_close(starts, counts, rtol=0, atol=0)
        self.assertTrue(bool((guard == -1).all()))


if __name__ == "__main__":
    main()
