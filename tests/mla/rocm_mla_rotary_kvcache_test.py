import logging
import math
import os
import unittest

import torch

os.environ["DEVICE_RESERVE_MEMORY_BYTES"] = "128000000"

import random
from dataclasses import dataclass
from typing import List

from rotary_util import apply_rotary_pos_emb
from test_util import MlaOpsType, compare_tensor_diff_with_ratio

from rtp_llm.models.rotary_embedding.deepseek_rotary_embedding import (
    DeepseekV3YarnRotaryEmbedding,
)

logging.basicConfig(level="INFO", format="%(message)s")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


os.environ["ENABLE_TRTV1_FMHA"] = "OFF"
os.environ["ENABLE_TRT_FMHA"] = "OFF"


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


class RotaryConfig:
    max_position_embeddings: int = 163840
    scaling_factor: int = 40
    rope_theta: int = 10000
    original_max_position_embeddings: int = 4096
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    mscale_all_dim: float = 1.0


def trans(x: torch.Tensor):
    b, h, s, d = x.shape
    x = x.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    x = x.reshape(b * h, s * d).contiguous()
    return x


class TestRotaryKVcacheTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        torch.classes.load_library(
            os.environ["TEST_SRCDIR"] + "/rtp_llm/tests/librocm_test_ops.so"
        )
        self.config = DeepSeekConfig()
        # self.mla_ops_type = MlaOpsType.FLASH_MLA
        self.mla_ops_type = MlaOpsType.FLASH_INFER
        self.rotary_config = RotaryConfig()
        self.mla_rotary_kvcache_op = torch.classes.unittest.MlaRotaryKVCacheOp(
            self.mla_ops_type,
            self.config.head_num,
            self.config.nope_head_size,
            self.config.rope_head_size,
            self.config.v_head_size,
            self.config.q_lora_rank,
            self.config.kv_lora,
            self.config.hidden_size,
            self.config.mscale * self.config.mscale,
        )
        self._init_cos_sin_cache()

    def _init_cos_sin_cache(self):
        self.rotary_emb = (
            DeepseekV3YarnRotaryEmbedding(
                self.config.rope_head_size,
                max_position_embeddings=self.rotary_config.max_position_embeddings,
                scaling_factor=self.rotary_config.scaling_factor,
                base=self.rotary_config.rope_theta,
                original_max_position_embeddings=self.rotary_config.original_max_position_embeddings,
                beta_fast=self.rotary_config.beta_fast,
                beta_slow=self.rotary_config.beta_slow,
                mscale=self.rotary_config.mscale,
                mscale_all_dim=self.rotary_config.mscale_all_dim,
            )
            .to(torch.float32)
            .to(torch.device("cuda"))
        )
        cos_cache = self.rotary_emb.cos_cached[:, : self.config.rope_head_size // 2]
        sin_cache = self.rotary_emb.sin_cached[:, : self.config.rope_head_size // 2]
        self.cos_sin_cache = (
            torch.cat([cos_cache, sin_cache], dim=-1)
            .to(torch.float32)
            .to(torch.device("cuda"))
        )

    # q shape [token_num, head_num, rope_head_size]
    # k shape [token_num, rope_head_size]
    def do_rotary_ref(self, q: torch.Tensor, k: torch.Tensor, indexs: List[int]):
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.unsqueeze(0).unsqueeze(0)
        cos, sin = self.rotary_emb(
            torch.zeros([1]).float().cuda(), seq_len=indexs[-1] + 1
        )
        q_rope, k_rope = apply_rotary_pos_emb(
            q,
            k,
            cos,
            sin,
            torch.tensor(
                indexs, device=torch.device("cuda"), dtype=torch.int32
            ).unsqueeze(0),
        )
        q_rope = q_rope.squeeze(0).transpose(0, 1)
        k_rope = k_rope.squeeze(0).squeeze(0)
        return q_rope, k_rope

    def _test_one_case(
        self, sequence_lengths: List[int], intput_lengths: List[int], page_size: int
    ):
        logging.info(
            f"--------------------{self.mla_ops_type}, {sequence_lengths}, {intput_lengths}, {page_size}--------------------"
        )
        batch_page_sizes = [math.ceil(x / page_size) for x in sequence_lengths] + [
            math.ceil(x / page_size) for x in intput_lengths[len(sequence_lengths) :]
        ]
        total_page_num = max(batch_page_sizes) * len(intput_lengths)
        block_id_map = torch.zeros(
            [len(intput_lengths), max(batch_page_sizes)], dtype=torch.int32
        )
        for i in range(len(intput_lengths)):
            for j in range(max(batch_page_sizes)):
                block_id_map[i, j] = i * max(batch_page_sizes) + j
        block_id_map_device = block_id_map.to("cuda")
        token_num = len(sequence_lengths) + sum(intput_lengths[len(sequence_lengths) :])
        q = torch.randn(
            [
                token_num,
                self.config.head_num,
                self.config.nope_head_size + self.config.rope_head_size,
            ],
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )
        q_rope_clone = (
            q[:, :, self.config.nope_head_size :].contiguous().detach().clone()
        )
        fused_qkv = torch.randn(
            [
                token_num,
                self.config.q_lora_rank
                + self.config.kv_lora
                + self.config.rope_head_size,
            ],
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )
        ckv = fused_qkv[
            :, self.config.q_lora_rank : self.config.q_lora_rank + self.config.kv_lora
        ]
        k_rope = fused_qkv[:, self.config.q_lora_rank + self.config.kv_lora :]
        k_rope_clone = k_rope.detach().clone()
        # transpose
        # q[..., self.config.nope_head_size:] = q[..., self.config.nope_head_size:].reshape(token_num, self.config.head_num, self.config.rope_head_size // 2, 2).transpose(2, 3).reshape(token_num, self.config.head_num, self.config.rope_head_size).contiguous()
        # k_rope = k_rope.reshape(token_num, self.config.rope_head_size // 2, 2).transpose(1, 2).reshape(token_num, self.config.rope_head_size).contiguous()

        # TODO: CHECK COS SIN CACHE TYPE
        cos_sin_cache = self.cos_sin_cache
        sequence_length_mins_one = torch.tensor(
            [x - 1 for x in sequence_lengths], dtype=torch.int32
        )
        prefix_length_t = torch.zeros(
            (len(intput_lengths) - len(sequence_lengths)), dtype=torch.int32
        )
        self.mla_rotary_kvcache_op.init(
            prefix_length_t,
            sequence_length_mins_one,
            torch.tensor(intput_lengths, dtype=torch.int32),
            page_size,
            block_id_map,
            block_id_map_device,
        )

        kv_cache = torch.zeros(
            [
                total_page_num,
                page_size,
                self.config.kv_lora + self.config.rope_head_size,
            ],
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )
        ckv_cache = kv_cache[:, :, : self.config.kv_lora]
        kpe_cache = kv_cache[:, :, self.config.kv_lora :]
        self.mla_rotary_kvcache_op.applyRotaryKVCache(
            q,
            fused_qkv,
            self.config.q_lora_rank,
            kv_cache,
            torch.empty(tuple(), dtype=torch.bfloat16, device=torch.device("cuda")),
            cos_sin_cache,
        )
        torch.cuda.synchronize()

        # check kvcache not empty
        page_offset = 0
        decode_batch_size = len(sequence_lengths)

        # check decode write kvcache
        for i, seq_len in enumerate(sequence_length_mins_one):
            page_idx = math.floor(seq_len // page_size)
            page_offset = seq_len % page_size
            torch.testing.assert_close(
                ckv_cache.reshape(-1, self.config.kv_lora)[
                    block_id_map[i][page_idx] * page_size + page_offset
                ],
                ckv[i],
            )
            torch.testing.assert_close(
                kpe_cache.reshape(-1, self.config.rope_head_size)[
                    block_id_map[i][page_idx] * page_size + page_offset
                ],
                k_rope[i],
            )
        # check prefill write kvcache
        offset = decode_batch_size
        for i, input_len in enumerate(intput_lengths[decode_batch_size:]):
            for j in range(input_len):
                page_idx = math.floor(j // page_size)
                page_offset = j % page_size
                torch.testing.assert_close(
                    ckv_cache.reshape(-1, self.config.kv_lora)[
                        block_id_map[i + decode_batch_size][page_idx] * page_size
                        + page_offset
                    ],
                    ckv[offset + j],
                )
                torch.testing.assert_close(
                    kpe_cache.reshape(-1, self.config.rope_head_size)[
                        block_id_map[i + decode_batch_size][page_idx] * page_size
                        + page_offset
                    ],
                    k_rope[offset + j],
                )
            offset += input_len

        q_rope = q[:, :, self.config.nope_head_size :]
        torch.cuda.synchronize()
        # check rotary embedding
        for i, seq_len in enumerate(sequence_length_mins_one):
            q_rope_compare, k_rope_compare = self.do_rotary_ref(
                q_rope_clone[i].unsqueeze(0), k_rope_clone[i].unsqueeze(0), [seq_len]
            )
            compare_tensor_diff_with_ratio(
                q_rope[i],
                q_rope_compare.reshape(q_rope[i].shape),
                rel_threshold=1e-2,
                abs_threshold=1e-3,
                name="Decode Batch " + str(i),
            )
            compare_tensor_diff_with_ratio(
                k_rope[i],
                k_rope_compare.reshape(k_rope[i].shape),
                rel_threshold=1e-2,
                abs_threshold=1e-3,
                name="Decode Batch " + str(i),
            )

        offset = decode_batch_size
        for i, input_len in enumerate(intput_lengths[decode_batch_size:]):
            q_rope_compare, k_rope_compare = self.do_rotary_ref(
                q_rope_clone[offset : offset + input_len],
                k_rope_clone[offset : offset + input_len],
                list(range(input_len)),
            )
            compare_tensor_diff_with_ratio(
                q_rope[offset : offset + input_len],
                q_rope_compare.reshape(q_rope[offset : offset + input_len].shape),
                rel_threshold=1e-2,
                abs_threshold=1e-3,
                name="Prefill Batch " + str(i),
            )
            compare_tensor_diff_with_ratio(
                k_rope[offset : offset + input_len],
                k_rope_compare.reshape(k_rope[offset : offset + input_len].shape),
                rel_threshold=1e-2,
                abs_threshold=1e-3,
                name="Prefill Batch " + str(i),
            )
            offset += input_len

    def test_rotary_kvcache(self):
        set_seed(42)

        self._test_one_case([], [25], 16)
        self._test_one_case([], [1024], 16)
        self._test_one_case([30], [25], 16)
        self._test_one_case([26], [25], 64)
        self._test_one_case([26], [25, 65], 64)
        self._test_one_case([84], [25, 953], 64)
        self._test_one_case([30], [25, 300], 16)
        self._test_one_case([30], [25, 34, 65, 123, 2345], 64)
        self._test_one_case([130], [39], 64)
        self._test_one_case([129], [39], 64)
        self._test_one_case([127], [39], 64)
        self._test_one_case([128], [39], 64)
        self._test_one_case([129], [39], 64)
        self._test_one_case([126], [39], 64)


if __name__ == "__main__":
    unittest.main()
