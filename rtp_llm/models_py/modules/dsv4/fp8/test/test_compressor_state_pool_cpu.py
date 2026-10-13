"""UT for ``run_save_partial_states`` with CUDA vs pinned CPU state pools.

The compressor writes partial per-token state through a Triton kernel. This
test pins the current contract: CUDA state pools are writable, and pinned CPU
state pools are also valid write targets through CUDA host-pinned memory.
"""

from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8._compressor_vllm_triton import (
    run_save_partial_states,
)


N = 4
HEAD_SIZE = 8
BLOCK_SIZE = 16
COMPRESS_RATIO = 4


def _inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    kv = torch.arange(N * HEAD_SIZE, dtype=torch.float32, device="cuda").reshape(
        N, HEAD_SIZE
    )
    score = torch.arange(N * HEAD_SIZE, dtype=torch.float32, device="cuda").reshape(
        N, HEAD_SIZE
    )
    ape = (
        torch.arange(COMPRESS_RATIO * HEAD_SIZE, dtype=torch.float32, device="cuda")
        .reshape(COMPRESS_RATIO, HEAD_SIZE)
        .mul_(0.01)
    )
    positions = torch.arange(N, dtype=torch.int64, device="cuda")
    state_slots = torch.arange(N, dtype=torch.int64, device="cuda")
    return kv, score, ape, positions, state_slots


def _state_pool(*, device: str, pin_memory: bool = False) -> torch.Tensor:
    return torch.zeros(
        (1, BLOCK_SIZE, 2 * HEAD_SIZE),
        dtype=torch.float32,
        device=device,
        pin_memory=pin_memory,
    )


class CompressorStatePoolCpuTest(unittest.TestCase):

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def test_cuda_state_pool_succeeds_and_writes(self) -> None:
        kv, score, ape, positions, state_slots = _inputs()
        state_pool = _state_pool(device="cuda")

        run_save_partial_states(
            kv,
            score,
            ape,
            positions,
            state_pool,
            state_slots,
            compress_ratio=COMPRESS_RATIO,
        )
        torch.cuda.synchronize()

        expected_score = score + ape[positions % COMPRESS_RATIO]
        self.assertTrue(torch.equal(state_pool[0, :N, :HEAD_SIZE], kv))
        self.assertTrue(
            torch.allclose(state_pool[0, :N, HEAD_SIZE : 2 * HEAD_SIZE], expected_score)
        )

    def test_pinned_cpu_state_pool_succeeds_and_writes(self) -> None:
        kv, score, ape, positions, state_slots = _inputs()
        state_pool = _state_pool(device="cpu", pin_memory=True)
        self.assertTrue(state_pool.is_pinned())

        run_save_partial_states(
            kv,
            score,
            ape,
            positions,
            state_pool,
            state_slots,
            compress_ratio=COMPRESS_RATIO,
        )
        torch.cuda.synchronize()

        expected_score = score + ape[positions % COMPRESS_RATIO]
        self.assertTrue(torch.equal(state_pool[0, :N, :HEAD_SIZE], kv.cpu()))
        self.assertTrue(
            torch.allclose(
                state_pool[0, :N, HEAD_SIZE : 2 * HEAD_SIZE],
                expected_score.cpu(),
            )
        )


if __name__ == "__main__":
    unittest.main()
