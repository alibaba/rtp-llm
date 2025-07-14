import logging

logging.basicConfig(level="INFO", format="%(message)s")

import math
import os
import random
import unittest
from dataclasses import dataclass

import torch
from flash_attn import flash_attn_varlen_func
from test_util import MlaOpsType, compare_tensor_diff_with_ratio

os.environ["DEVICE_RESERVE_MEMORY_BYTES"] = "128000000"


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


def torch_attention(
    q: torch.Tensor,
    kv_a: torch.Tensor,
    k_rope: torch.Tensor,
    k_nope_weight: torch.Tensor,
    v_weight: torch.Tensor,
    seq_len: torch.Tensor,
    config: DeepSeekConfig,
):
    head_num = config.head_num
    nope_head_size = config.nope_head_size
    rope_head_size = config.rope_head_size
    nope_rope_size = config.nope_rope_size
    v_head_size = config.v_head_size
    # [token, kv_lora] [kv_lora, head * nope_size] -> [token, nope_size]
    k_nope = kv_a @ k_nope_weight
    # [token, kv_lora] [kv_lora, head * nope_size] -> [token, nope_size]
    v = kv_a @ v_weight
    token_num = q.shape[0]

    k_nope_tensor = k_nope.reshape(1, token_num, head_num, nope_head_size).permute(
        0, 2, 1, 3
    )
    k_rope_tensor = k_rope.reshape(1, token_num, 1, rope_head_size).permute(0, 2, 1, 3)

    query_states = q.reshape(1, token_num, head_num, nope_rope_size).permute(0, 2, 1, 3)
    key_states = query_states.new_empty(1, head_num, token_num, nope_rope_size)
    key_states[:, :, :, :nope_head_size] = k_nope_tensor
    key_states[:, :, :, nope_head_size:] = k_rope_tensor
    value_states = v.reshape(1, token_num, head_num, v_head_size).permute(0, 2, 1, 3)

    # call flash attn
    zero_paded = torch.zeros(
        1,
        head_num,
        token_num,
        rope_head_size,
        dtype=query_states.dtype,
        device=query_states.device,
    )
    value_states = torch.cat([value_states, zero_paded], dim=-1)
    query_states = query_states.transpose(1, 2).squeeze(0).contiguous()
    key_states = key_states.transpose(1, 2).squeeze(0).contiguous()
    value_states = value_states.transpose(1, 2).squeeze(0).contiguous()

    batch_size = seq_len.shape[0]
    cu_seq_len = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=query_states.device
    )
    total_seq_len = 0
    for i in range(batch_size):
        total_seq_len += seq_len[i]
        cu_seq_len[i + 1] = total_seq_len
    max_seq_len = int(torch.max(seq_len).item())

    attn_output: torch.Tensor = flash_attn_varlen_func(query_states, key_states, value_states, cu_seq_len, cu_seq_len, max_seq_len, max_seq_len, softmax_scale=config.softmax_scale, causal=True)  # type: ignore

    attn_output = attn_output.reshape(1, token_num, head_num, -1)

    return attn_output


class TestRope(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        torch.classes.load_library(
            os.environ["TEST_SRCDIR"] + "/rtp_llm/tests/libtest_ops.so"
        )
        self.config = DeepSeekConfig()

        # self.mla_ops_type = MlaOpsType.FLASH_MLA
        self.mla_ops_type = MlaOpsType.FLASH_INFER
        self.mla_context_attn_op = torch.classes.unittest.MlaContextAttnOp(
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

    def test_mla_context_attn(self):
        set_seed(0)

        with torch.no_grad():
            config = self.config
            head_num = config.head_num
            nope_head_size = config.nope_head_size
            rope_head_size = config.rope_head_size
            nope_rope_size = config.nope_rope_size
            v_head_size = config.v_head_size
            kv_lora = config.kv_lora
            q_lora = config.q_lora_rank

            dtype = torch.bfloat16
            device = torch.device("cuda")

            batch_size = 4
            seq_len = torch.randint(512, 1024, (batch_size,), dtype=torch.int32)
            token_nums = int(torch.sum(seq_len).item())  # type: ignore

            q = torch.randn(
                token_nums, head_num * nope_rope_size, dtype=dtype, device=device
            )
            fused_qkv = torch.randn(
                token_nums,
                q_lora + kv_lora + rope_head_size,
                dtype=dtype,
                device=device,
            )
            kv_a = fused_qkv[:, q_lora : q_lora + kv_lora].contiguous()
            k_rope = fused_qkv[:, q_lora + kv_lora :].contiguous()
            cos_sin_cache = (
                torch.concat(
                    [torch.ones([16384, 32]), torch.zeros([16384, 32])], dim=-1
                )
                .to(torch.device("cuda"))
                .to(torch.float32)
            )

            k_nope_weight = torch.randn(
                kv_lora, head_num * nope_head_size, dtype=dtype, device=device
            )
            v_weight = torch.randn(
                kv_lora, head_num * v_head_size, dtype=dtype, device=device
            )

            # [1, token_num, head_num, nope_size]
            attn_output_expect = torch_attention(
                q, kv_a, k_rope, k_nope_weight, v_weight, seq_len, config
            )
            attn_output_expect = attn_output_expect[:, :, :, :v_head_size].contiguous()
            print(attn_output_expect.shape)

            attn_output = self.mla_context_attn_op.forward(
                q, fused_qkv, q_lora, k_nope_weight, v_weight, cos_sin_cache, seq_len
            )
            attn_output = attn_output.reshape(1, token_nums, head_num, v_head_size)
            compare_tensor_diff_with_ratio(attn_output, attn_output_expect, 1e-2, 2e-3)
            torch.cuda.synchronize()


if __name__ == "__main__":
    logging.info("cwd: %s", os.getcwd())
    unittest.main()
