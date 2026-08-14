from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.triton_kernels.kimi_kda import (
    kimi_k3_store_linear_cache_state,
    kimi_k3_store_linear_cache_states,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class KimiK3CacheStoreTest(unittest.TestCase):
    def test_batched_store_matches_single_state_wrapper(self) -> None:
        torch.manual_seed(20260814)
        state_count = 4
        heads = 2
        state_dim = 128
        channels = heads * state_dim
        history_size = 3
        cache_blocks = 9
        recurrent = torch.randn(
            state_count,
            heads,
            state_dim,
            state_dim,
            device="cuda",
            dtype=torch.float32,
        )
        q_state = torch.randn(
            state_count,
            channels,
            history_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k_state = torch.randn_like(q_state)
        v_state = torch.randn_like(q_state)
        block_ids = torch.tensor([1, 3, 6, 8], device="cuda", dtype=torch.int32)
        batched_ssm = torch.zeros(
            cache_blocks,
            heads,
            state_dim,
            state_dim,
            device="cuda",
            dtype=torch.float32,
        )
        batched_conv = torch.zeros(
            cache_blocks,
            history_size,
            3 * channels,
            device="cuda",
            dtype=torch.bfloat16,
        )
        reference_ssm = torch.zeros_like(batched_ssm)
        reference_conv = torch.zeros_like(batched_conv)

        kimi_k3_store_linear_cache_states(
            recurrent,
            q_state,
            k_state,
            v_state,
            block_ids,
            batched_ssm,
            batched_conv,
        )
        for state_index, block_id in enumerate(block_ids.tolist()):
            kimi_k3_store_linear_cache_state(
                recurrent[state_index],
                q_state[state_index],
                k_state[state_index],
                v_state[state_index],
                reference_ssm[block_id],
                reference_conv[block_id],
            )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(batched_ssm, reference_ssm))
        self.assertTrue(torch.equal(batched_conv, reference_conv))

    def test_negative_block_id_skips_unmaterialized_checkpoint(self) -> None:
        recurrent = torch.ones((1, 1, 128, 128), device="cuda")
        conv_state = torch.ones((1, 128, 3), device="cuda", dtype=torch.bfloat16)
        ssm_cache = torch.zeros((2, 1, 128, 128), device="cuda")
        conv_cache = torch.zeros((2, 3, 384), device="cuda", dtype=torch.bfloat16)
        kimi_k3_store_linear_cache_states(
            recurrent,
            conv_state,
            conv_state,
            conv_state,
            torch.tensor([-1], device="cuda", dtype=torch.int32),
            ssm_cache,
            conv_cache,
        )
        torch.cuda.synchronize()
        self.assertEqual(int(torch.count_nonzero(ssm_cache)), 0)
        self.assertEqual(int(torch.count_nonzero(conv_cache)), 0)


if __name__ == "__main__":
    unittest.main()
