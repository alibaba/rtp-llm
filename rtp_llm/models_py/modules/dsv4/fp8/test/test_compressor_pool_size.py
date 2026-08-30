"""Regression test for changing DSV4 state-pool capacity in one process."""

from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8._compressor_vllm_triton import (
    _save_partial_states_kernel,
    run_save_partial_states,
)


def _compiled_kernel_count() -> int:
    return sum(
        len(kernel_cache)
        for kernel_cache, *_ in _save_partial_states_kernel.device_caches.values()
    )


class CompressorPoolSizeTest(unittest.TestCase):
    def test_state_pool_capacity_is_cuda_graph_stable(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

        device = torch.device("cuda")
        token_count = 4
        block_size = 8
        state_width = 256
        compress_ratio = 4
        kv = torch.randn(
            token_count, state_width, dtype=torch.float32, device=device
        )
        score = torch.randn_like(kv)
        ape = torch.randn(
            compress_ratio, state_width, dtype=torch.float32, device=device
        )
        positions = torch.arange(token_count, dtype=torch.int64, device=device)
        slots = block_size + positions

        small_cache = torch.zeros(
            2,
            block_size,
            2 * state_width,
            dtype=torch.float32,
            device=device,
        )
        run_save_partial_states(
            kv, score, ape, positions, small_cache, slots, compress_ratio
        )
        torch.cuda.synchronize()
        compiled_after_warmup = _compiled_kernel_count()

        large_cache = torch.zeros(
            5,
            block_size,
            2 * state_width,
            dtype=torch.float32,
            device=device,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_save_partial_states(
                kv, score, ape, positions, large_cache, slots, compress_ratio
            )
        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(compiled_after_warmup, _compiled_kernel_count())
        torch.testing.assert_close(
            large_cache[1, :token_count, :state_width], kv
        )
        torch.testing.assert_close(
            large_cache[1, :token_count, state_width:], score + ape
        )


if __name__ == "__main__":
    unittest.main()
