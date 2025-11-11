import logging
import math
import random
import unittest
from typing import List

import torch

from rtp_llm.models_py.triton_kernels.fla import (
    load_initial_state_from_block_map,
    store_ssm_state_to_block_map,
)

logging.basicConfig(
    level="INFO",
    format="[process-%(process)d][%(name)s][%(asctime)s.%(msecs)03d][%(filename)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class BlockTest(unittest.TestCase):
    def test_load_initial_state_from_block_map(self):
        device = torch.device("cuda")
        head_nums = [1, 4, 8, 16]
        k_sizes = [128, 256]
        v_size = 128
        batch_size = [1, 4, 16]
        seq_size_per_block = 16

        def test_one_case(k_size: int, head_num: int, bs: int):
            logging.info(
                f"test_load_initial_state_from_block_map: k_size: {k_size} head_num {head_num} bs {bs}"
            )
            if bs > 1:
                prefix_length = [random.randint(10, 1024) for _ in range(bs - 1)] + [0]
            else:
                prefix_length = [random.randint(10, 1024)]
            block_num = [
                math.ceil(prefix_length[i] / seq_size_per_block) for i in range(bs)
            ]
            total_block_num = sum(block_num)
            ssm_states_with_padding = torch.randn(
                total_block_num,
                head_num * v_size * k_size + 1024,
                dtype=torch.bfloat16,
                device=device,
            )
            ssm_states = ssm_states_with_padding[:, : head_num * v_size * k_size].view(
                total_block_num, head_num, v_size, k_size
            )
            initial_states = torch.empty(
                bs, head_num, v_size, k_size, device=device, dtype=torch.bfloat16
            )

            max_block_num = max(block_num)
            block_map = torch.ones([bs, max_block_num], dtype=torch.int32)
            offset = 0
            for i in range(bs):
                block_map[i, : block_num[i]] = torch.arange(
                    offset, offset + block_num[i], dtype=torch.int32
                )
                offset += block_num[i]
            block_map = block_map.to(device)
            prefix_length_t = torch.tensor(
                prefix_length, device=device, dtype=torch.int32
            )
            load_initial_state_from_block_map(
                prefix_length_t,
                block_map,
                ssm_states,
                initial_states,
                seq_size_per_block,
            )

            for i in range(bs):
                if prefix_length[i] > 0:
                    expect_value = ssm_states[
                        block_map[i][(prefix_length[i] - 1) // seq_size_per_block]
                    ]
                    torch.testing.assert_close(initial_states[i], expect_value)
                else:
                    torch.testing.assert_close(
                        initial_states[i], torch.zeros_like(initial_states[i])
                    )

        for k_size in k_sizes:
            for head_num in head_nums:
                for bs in batch_size:
                    test_one_case(k_size, head_num, bs)

    def test_store_ssm_state_to_block_map(self):
        device = torch.device("cuda")
        head_nums = [4, 8]
        k_sizes = [128]
        v_size = 128
        batch_size = [1, 4, 16]
        seq_size_per_block = 128
        chunk_size = 64

        def _test_one_case(
            k_size: int,
            head_num: int,
            bs: int,
            prefix_lengths: List[int],
            input_lengths: List[int],
        ):
            logging.info(
                f"test_store_ssm_state_to_block_map: k_size: {k_size} head_num {head_num} bs {bs}"
            )
            block_num = [
                math.ceil((input_lengths[i] + prefix_lengths[i]) / seq_size_per_block)
                for i in range(bs)
            ]
            chunk_lengths = [
                math.ceil(input_lengths[i] / chunk_size) for i in range(bs)
            ]
            total_block_num = sum(block_num)
            total_chunk_size = sum(chunk_lengths)
            block_map = torch.ones([bs, max(block_num)], dtype=torch.int32)
            offset = 0
            for i in range(bs):
                block_map[i, : block_num[i]] = torch.arange(
                    offset, offset + block_num[i], dtype=torch.int32
                )
                offset += block_num[i]
            ssm_states_with_padding = torch.ones(
                total_block_num,
                head_num * v_size * k_size + 1024,
                dtype=torch.bfloat16,
                device=device,
            )
            ssm_states = ssm_states_with_padding[:, : head_num * v_size * k_size].view(
                total_block_num, head_num, v_size, k_size
            )

            final_states = torch.randn(
                bs, head_num, v_size, k_size, device=device, dtype=torch.bfloat16
            )
            h = torch.randn(
                total_chunk_size,
                head_num,
                v_size,
                k_size,
                device=device,
                dtype=torch.bfloat16,
            )

            prefix_lengths_t = torch.tensor(
                prefix_lengths, device=device, dtype=torch.int32
            )
            cu_seq_len = [0]
            for length in input_lengths:
                cu_seq_len.append(cu_seq_len[-1] + length)
            cu_seq_len = torch.tensor(cu_seq_len, device=device, dtype=torch.int32)
            block_map_gpu = block_map.to(device)
            store_ssm_state_to_block_map(
                h,
                final_states,
                prefix_lengths_t,
                cu_seq_len,
                block_map_gpu,
                ssm_states,
                seq_size_per_block,
                chunk_size,
            )

            chunk_offset = 0
            for i in range(bs):
                prefix_offset = prefix_lengths[i] // seq_size_per_block
                for block_idx in range(block_num[i] - 1):
                    block_idx -= prefix_offset
                    if block_idx < 0:
                        continue
                    # last block is always in final states
                    chunk_idx = (
                        chunk_offset
                        + (block_idx + 1) * seq_size_per_block // chunk_size
                    )
                    torch.testing.assert_close(
                        ssm_states[block_map[i][block_idx + prefix_offset]],
                        h[chunk_idx],
                    )
                torch.testing.assert_close(
                    ssm_states[block_map[i][block_num[i] - 1]], final_states[i]
                )
                chunk_offset += chunk_lengths[i]

        for k_size in k_sizes:
            for head_num in head_nums:
                for bs in batch_size:
                    if bs > 1:
                        prefix_lengths = [
                            random.randint(1, 10) * seq_size_per_block
                            for _ in range(bs - 1)
                        ] + [0]
                    else:
                        prefix_lengths = [random.randint(1, 10) * seq_size_per_block]
                        input_lengths = [random.randint(10, 1024) for _ in range(bs)]
                        _test_one_case(
                            k_size, head_num, bs, prefix_lengths, input_lengths
                        )
                        # test input_length % seq_size_per_block == 0 case
                        input_lengths = [
                            random.randint(1, 10) * seq_size_per_block
                            for _ in range(bs)
                        ]
                        _test_one_case(
                            k_size, head_num, bs, prefix_lengths, input_lengths
                        )


if __name__ == "__main__":
    unittest.main()
