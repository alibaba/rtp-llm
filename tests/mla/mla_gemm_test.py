import os
import unittest

import torch
import torch.nn as nn

os.environ["DEVICE_RESERVE_MEMORY_BYTES"] = "128000000"


def deepseek_transpose(x: torch.Tensor) -> torch.Tensor:
    b, h, s, d = x.shape
    x = x.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    return x


class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        DeepseekV2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MlaConfig:
    hidden_size = 4096
    num_attention_heads = 32
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    q_lora_rank = 1536
    kv_lora_rank = 512
    attention_bias = False


class MlaQKVGemm(nn.Module):
    def __init__(self, config: MlaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True

        self.q_a_proj = nn.Linear(
            self.hidden_size, config.q_lora_rank, bias=config.attention_bias
        )
        self.q_a_layernorm = DeepseekV2RMSNorm(config.q_lora_rank)
        self.q_b_proj = nn.Linear(
            config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV2RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

    def set_weight(
        self,
        q_a_proj_weight: torch.Tensor,
        q_b_proj_weight: torch.Tensor,
        kv_a_proj_with_mqa_weight: torch.Tensor,
        kv_b_proj_weight: torch.Tensor,
        kv_a_layernorm_weight: torch.Tensor,
        q_a_layernorm_weight: torch.Tensor,
    ):
        self.q_a_proj.weight = nn.Parameter(q_a_proj_weight)
        self.q_b_proj.weight = nn.Parameter(q_b_proj_weight)
        self.kv_a_proj_with_mqa.weight = nn.Parameter(kv_a_proj_with_mqa_weight)
        self.kv_b_proj.weight = nn.Parameter(kv_b_proj_weight)
        self.kv_a_layernorm.weight = nn.Parameter(kv_a_layernorm_weight)
        self.q_a_layernorm.weight = nn.Parameter(q_a_layernorm_weight)

    def forward(self, hidden_states: torch.Tensor):
        bsz, q_len, _ = hidden_states.size()

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        q_pe = deepseek_transpose(q_pe)
        k_pe = deepseek_transpose(k_pe)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        qkv_states = k_pe.new_zeros(bsz, 3 * self.num_heads, q_len, self.q_head_dim)
        qkv_states[:, : self.num_heads, :, :] = query_states
        qkv_states[:, self.num_heads : 2 * self.num_heads, :, :] = key_states
        qkv_states[:, 2 * self.num_heads :, :, : self.v_head_dim] = value_states
        return qkv_states


class TestRope(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.config = MlaConfig()
        torch.classes.load_library(
            os.environ["TEST_SRCDIR"] + "/rtp_llm/tests/libtest_ops.so"
        )
        self.mla_qkv_gemm_op = torch.classes.unittest.MlaQKVGemmOP(
            self.config.num_attention_heads,
            self.config.q_lora_rank,
            self.config.kv_lora_rank,
            self.config.qk_nope_head_dim,
            self.config.qk_rope_head_dim,
            self.config.v_head_dim,
        )

    def test_mla_qkv_gemm(self):
        batch_sizes = [1, 2]
        seq_lens = range(512, 1025, 512)
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                config = self.config
                gemm = MlaQKVGemm(config)
                head_num = config.num_attention_heads
                nope_dim = config.qk_nope_head_dim
                rope_dim = config.qk_rope_head_dim
                vhead_dim = config.v_head_dim
                hidden_size = config.hidden_size
                kv_lora = config.kv_lora_rank

                q_a_weight = torch.randn(config.q_lora_rank, config.hidden_size)
                q_b_weight = torch.randn(
                    head_num * (nope_dim + rope_dim), config.q_lora_rank
                )
                kv_a_weight = torch.randn(
                    config.kv_lora_rank + rope_dim, config.hidden_size
                )
                kv_b_weight = torch.randn(
                    head_num * (nope_dim + vhead_dim), config.kv_lora_rank
                )
                kv_a_layernorm_weight = torch.randn(config.kv_lora_rank)
                q_a_layernorm_weight = torch.randn(config.q_lora_rank)

                gemm.set_weight(
                    q_a_weight,
                    q_b_weight,
                    kv_a_weight,
                    kv_b_weight,
                    kv_a_layernorm_weight,
                    q_a_layernorm_weight,
                )

                hidden_states = (
                    torch.randn(batch_size, seq_len, hidden_size).contiguous().cuda()
                )
                gemm.cuda()
                expect_output = gemm(hidden_states).transpose(1, 2)
                q_a = q_a_weight.transpose(0, 1).contiguous().cuda()
                q_b = q_b_weight.transpose(0, 1).contiguous().cuda()

                kv_a, k_rope = torch.split(
                    kv_a_weight.transpose(0, 1),
                    [config.kv_lora_rank, config.qk_rope_head_dim],
                    dim=-1,
                )
                kv_a = kv_a.contiguous().cuda()
                k_rope = k_rope.contiguous().cuda()

                k_nope, v = torch.split(
                    kv_b_weight.transpose(0, 1).reshape(
                        kv_lora, head_num, nope_dim + vhead_dim
                    ),
                    [nope_dim, vhead_dim],
                    dim=-1,
                )

                k_nope = (
                    k_nope.reshape(kv_lora, head_num * nope_dim).contiguous().cuda()
                )
                v = v.reshape(kv_lora, head_num * nope_dim).contiguous().cuda()

                q_layernorm = q_a_layernorm_weight.contiguous().cuda()
                kv_layernorm = kv_a_layernorm_weight.contiguous().cuda()
                output = (
                    torch.randn(
                        batch_size, seq_len, 3 * head_num, (nope_dim + rope_dim)
                    )
                    .contiguous()
                    .cuda()
                )
                hidden_states = (
                    hidden_states.reshape(batch_size * seq_len, hidden_size)
                    .contiguous()
                    .cuda()
                )
                print(hidden_states.shape, q_a.shape)
                self.mla_qkv_gemm_op.forward(
                    hidden_states,
                    output,
                    q_a,
                    q_b,
                    kv_a,
                    k_nope,
                    k_rope,
                    v,
                    q_layernorm,
                    kv_layernorm,
                )

                torch.testing.assert_close(output, expect_output, atol=1e-3, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
