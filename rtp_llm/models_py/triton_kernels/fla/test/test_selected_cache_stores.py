import unittest

import torch

from rtp_llm.models_py.triton_kernels.causal_conv1d import gather_conv_initial_states
from rtp_llm.models_py.triton_kernels.fla import (
    prepare_selected_state_positions,
    store_selected_conv_states_to_block_map,
    store_selected_ssm_states_to_block_map,
    store_ssm_state_to_block_map,
)


def make_strided_ssm_states(
    blocks: int, heads: int, v: int, k: int, dtype: torch.dtype
):
    """Mimic LinearCacheConverter's padded block stride with contiguous rows."""
    elements = heads * v * k
    padded = torch.randn(blocks, elements + 7, device="cuda", dtype=dtype)
    return padded[:, :elements].view(blocks, heads, v, k)


def make_strided_conv_states(blocks: int, width: int, dim: int, dtype: torch.dtype):
    """Return the [block, dim, width-1] view used by model call sites."""
    elements = width * dim
    padded = torch.randn(blocks, elements + 7, device="cuda", dtype=dtype)
    return padded[:, :elements].view(blocks, width, dim).transpose(1, 2)


class SelectedCacheStoresTest(unittest.TestCase):
    def test_prepare_selected_state_positions_gdn_tail_contract(self):
        prefix = torch.tensor([1300], device="cuda", dtype=torch.int64)
        cu = torch.tensor([0, 700], device="cuda", dtype=torch.int64)
        positions = torch.tensor(
            [[1536, 1792, 2000, -1, 1300, 2500]],
            device="cuda",
            dtype=torch.int64,
        )
        block_ids = torch.tensor(
            [[10, 11, 12, -1, 13, 14]], device="cuda", dtype=torch.int32
        )

        result = prepare_selected_state_positions(
            prefix, cu, positions, block_ids, positions.device
        )

        torch.testing.assert_close(
            result,
            torch.tensor(
                [[236, 492, -1, -1, -1, -1]], device="cuda", dtype=torch.int32
            ),
        )
        self.assertEqual(result.dtype, torch.int32)
        self.assertTrue(result.is_contiguous())

    def test_store_selected_ssm_states_tail_uses_final_and_skips_padding(self):
        blocks, heads, v, k = 8, 2, 3, 4
        ssm_states = make_strided_ssm_states(blocks, heads, v, k, torch.bfloat16)
        before = ssm_states.clone()
        prefix = torch.tensor([1300], device="cuda", dtype=torch.int64)
        cu = torch.tensor([0, 700], device="cuda", dtype=torch.int64)
        positions = torch.tensor(
            [[1536, 1792, 2000, -1, 1300]], device="cuda", dtype=torch.int64
        )
        block_ids = torch.tensor([[3, 4, 5, 6, 7]], device="cuda", dtype=torch.int32)
        selected_states = torch.empty(
            1, positions.shape[1], heads, v, k, device="cuda", dtype=torch.float32
        )
        for slot in range(positions.shape[1]):
            selected_states[:, slot] = slot + 10
        final_states = torch.full(
            (1, heads, v, k), 99.0, device="cuda", dtype=torch.float32
        )

        store_selected_ssm_states_to_block_map(
            selected_states,
            final_states,
            prefix,
            cu,
            positions,
            block_ids,
            ssm_states,
        )

        torch.testing.assert_close(
            ssm_states[3], selected_states[0, 0].to(torch.bfloat16)
        )
        torch.testing.assert_close(
            ssm_states[4], selected_states[0, 1].to(torch.bfloat16)
        )
        torch.testing.assert_close(ssm_states[5], final_states[0].to(torch.bfloat16))
        torch.testing.assert_close(ssm_states[6], before[6])
        torch.testing.assert_close(ssm_states[7], before[7])

    def test_ordinary_ssm_store_suppresses_nonaligned_intermediate_write(self):
        blocks, heads, v, k = 3, 2, 3, 4
        ssm_states = make_strided_ssm_states(blocks, heads, v, k, torch.bfloat16)
        before = ssm_states.clone()
        prefix = torch.tensor([4], device="cuda", dtype=torch.int32)
        cu = torch.tensor([0, 20], device="cuda", dtype=torch.int32)
        block_map = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        h = torch.randn(5, heads, v, k, device="cuda", dtype=torch.float32)
        final_states = torch.randn(1, heads, v, k, device="cuda", dtype=torch.float32)

        store_ssm_state_to_block_map(
            h,
            final_states,
            prefix,
            cu,
            block_map,
            ssm_states,
            seq_size_per_block=8,
            chunk_size=4,
        )

        # Prefix 4 is not block-aligned. The old writer could treat the suffix
        # chunk ending at 8 as a complete global block and overwrite block 1.
        torch.testing.assert_close(ssm_states[0], before[0])
        torch.testing.assert_close(ssm_states[1], before[1])
        torch.testing.assert_close(ssm_states[2], final_states[0].to(torch.bfloat16))

    def test_store_selected_conv_states_copies_endpoints_and_respects_alias(self):
        blocks, width, dim = 110, 4, 5
        conv_states = make_strided_conv_states(blocks, width, dim, torch.bfloat16)
        for block in range(blocks):
            conv_states[block] = block
        before = conv_states.clone()
        prefix = torch.tensor([1300], device="cuda", dtype=torch.int64)
        cu = torch.tensor([0, 700], device="cuda", dtype=torch.int64)
        positions = torch.tensor(
            [[1536, 1792, 2000, 1536, -1]], device="cuda", dtype=torch.int64
        )
        block_ids = torch.tensor(
            [[100, 101, 102, 10, 103]], device="cuda", dtype=torch.int32
        )
        runtime_map = torch.tensor(
            [[0, 1, 2, 3, 4, 10, 11, 12]], device="cuda", dtype=torch.int32
        )

        store_selected_conv_states_to_block_map(
            conv_states,
            prefix,
            cu,
            positions,
            block_ids,
            runtime_map,
            seq_size_per_block=256,
        )

        torch.testing.assert_close(conv_states[100], before[10])
        torch.testing.assert_close(conv_states[101], before[11])
        torch.testing.assert_close(conv_states[102], before[12])
        # Alias destination is the runtime source and must remain untouched.
        torch.testing.assert_close(conv_states[10], before[10])
        torch.testing.assert_close(conv_states[103], before[103])

    def test_gather_conv_initial_states_masks_zero_prefix_rows(self):
        blocks, width, dim = 4, 4, 5
        conv_states = make_strided_conv_states(blocks, width, dim, torch.bfloat16)
        for block in range(blocks):
            conv_states[block] = block
        prefix = torch.tensor([0, 5], device="cuda", dtype=torch.int64)
        block_map = torch.full((2, 2), -1, device="cuda", dtype=torch.int32)
        block_map[1, 0] = 2

        gathered = gather_conv_initial_states(
            conv_states, prefix, block_map, seq_size_per_block=8
        )

        torch.testing.assert_close(gathered[0], torch.zeros_like(gathered[0]))
        torch.testing.assert_close(gathered[1], conv_states[2])
        self.assertEqual(gathered.stride(1), 1)
        self.assertEqual(gathered.shape, (2, dim, width))


if __name__ == "__main__":
    unittest.main()
