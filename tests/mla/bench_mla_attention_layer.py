import logging
import math
import os
import random
import unittest
from dataclasses import dataclass
from typing import List

import torch
from rotary_util import create_cos_sin_cache

os.environ["DEVICE_RESERVE_MEMORY_BYTES"] = "128000000"

logging.basicConfig(level="INFO", format="%(message)s")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def yarn_get_mscale(scale: float = 1, mscale: float = 1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


@dataclass
class DeepSeekConfig:
    head_num = 128
    nope_head_size = 128
    rope_head_size = 64
    nope_rope_size = 192
    v_head_size = 128
    q_lora_rank = 1536
    kv_lora = 512
    hidden_size = 7168
    mscale = yarn_get_mscale(40, 1.0)
    softmax_scale = 192**-0.5 * mscale * mscale


class TestMlaAttentionLayer(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        logging.info(f"cwd: {os.getcwd()}")
        torch.classes.load_library(
            os.environ["TEST_SRCDIR"] + "/rtp_llm/tests/libtest_ops.so"
        )
        self.config = DeepSeekConfig()
        self.mla_attn_layer_op = torch.classes.unittest.MlaAttnLayerOp(
            self.config.head_num,
            self.config.nope_head_size,
            self.config.rope_head_size,
            self.config.v_head_size,
            self.config.q_lora_rank,
            self.config.kv_lora,
            self.config.hidden_size,
            self.config.mscale * self.config.mscale,
        )

    def _test_one_case(
        self, sequence_lengths: List[int], head_num: int, page_size: int
    ):
        self.config.head_num = head_num
        self.mla_attn_layer_op.reset_head_num(head_num)
        batch_size = len(sequence_lengths)
        total_token_num = len(sequence_lengths)
        seq_page_sizes = [math.ceil(x / page_size) for x in sequence_lengths]
        kvcache_block_id = torch.zeros(
            [batch_size, max(seq_page_sizes)], dtype=torch.int32
        )
        bias = 0
        for i in range(batch_size):
            kvcache_block_id[i, : seq_page_sizes[i]] = torch.arange(
                bias,
                bias + seq_page_sizes[i],
                dtype=torch.int32,
                device=torch.device("cuda"),
            )
            bias += seq_page_sizes[i]
        # 根据MlaAttnLayerOp::forward构造输入
        hidden = torch.randn(
            [total_token_num, self.config.hidden_size],
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )

        # 构造权重列表
        weights = [
            # 0. kc_t_weight
            torch.randn(
                [self.config.head_num, self.config.nope_head_size, self.config.kv_lora],
                dtype=torch.bfloat16,
                device=torch.device("cuda"),
            ),
            # 1, vc_t_weight
            torch.randn(
                [self.config.head_num, self.config.kv_lora, self.config.v_head_size],
                dtype=torch.bfloat16,
                device=torch.device("cuda"),
            ),
            # 2. q_a_norm_weight_gamma
            torch.randn(
                [self.config.q_lora_rank],
                dtype=torch.bfloat16,
                device=torch.device("cuda"),
            ),
            # 3. q_a_norm_weight_beta
            torch.randn(
                [self.config.q_lora_rank],
                dtype=torch.bfloat16,
                device=torch.device("cuda"),
            ),
            # 4 mla_fusedqkrope_w
            torch.randn(
                [
                    self.config.hidden_size,
                    self.config.q_lora_rank
                    + self.config.kv_lora
                    + self.config.rope_head_size,
                ],
                dtype=torch.bfloat16,
                device=torch.device("cuda"),
            ),
            # 5. q_b_weight
            torch.randn(
                [
                    self.config.q_lora_rank,
                    self.config.head_num
                    * (self.config.nope_head_size + self.config.rope_head_size),
                ],
                dtype=torch.bfloat16,
                device=torch.device("cuda"),
            ),
            # 6. kv_a_norm_weight_gamma
            torch.randn(
                [self.config.kv_lora], dtype=torch.bfloat16, device=torch.device("cuda")
            ),
            # 7. kv_a_norm_weight_beta
            torch.randn(
                [self.config.kv_lora], dtype=torch.bfloat16, device=torch.device("cuda")
            ),
            # 8. output_weight
            torch.randn(
                [
                    self.config.head_num * self.config.v_head_size,
                    self.config.hidden_size,
                ],
                dtype=torch.bfloat16,
                device=torch.device("cuda"),
            ),
            # 9. cos_sin_cache
            create_cos_sin_cache(),
        ]

        total_page_num = sum(seq_page_sizes)
        ckv_cache = torch.randn(
            [
                total_page_num,
                page_size,
                self.config.kv_lora + self.config.rope_head_size,
            ],
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )
        kpe_cache = torch.randn(
            [total_page_num, page_size, 0],
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )

        sequence_lengths_mius_1 = [x - 1 for x in sequence_lengths]
        sequence_lengths_t = torch.tensor(sequence_lengths_mius_1, dtype=torch.int32)
        prefix_lengths_t = torch.zeros(
            len(sequence_lengths_t) - len(sequence_lengths), dtype=torch.int32
        )
        benchmark_times = 5
        for _ in range(benchmark_times):
            with torch.cuda.nvtx.range("mla_attn_layer_op"):
                ft_output = self.mla_attn_layer_op.forward(
                    hidden,
                    weights,
                    ckv_cache,
                    kpe_cache,
                    prefix_lengths_t,
                    sequence_lengths_t,
                    kvcache_block_id,
                    page_size,
                )

    def test_mla_attention_layer(self):
        set_seed(42)
        head_num = 128
        world_size = 32

        for seq_len in [4096, 8192, 16384]:
            for batch_size in [128, 256, 512]:
                for tp_size in [1, 2, 4]:
                    local_head_num = head_num // tp_size
                    local_batch_size = batch_size // (world_size // tp_size)
                    with torch.cuda.nvtx.range(
                        f"b_{batch_size}_s_{seq_len}_tp_{tp_size}_lbs_{local_batch_size}"
                    ):
                        self._test_one_case(
                            [seq_len] * local_batch_size, local_head_num, 64
                        )


if __name__ == "__main__":
    unittest.main()
