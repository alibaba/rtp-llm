import itertools
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional
from unittest import SkipTest, TestCase, main

import torch

# CUR_PATH = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(str(CUR_PATH), "../../../"))
device = torch.device(f"cuda")

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models.rotary_embedding.deepseek_rotary_embedding import (
    DeepseekV3YarnRotaryEmbedding,
)
from rtp_llm.models_py.modules.fmha import (
    MlaFlashInferDecodeImpl,
    MlaFlashInferPrefillImpl,
)
from rtp_llm.models_py.modules.mla import DeepSeekV2Attention
from rtp_llm.models_py.modules.mla.mla_attention_ref import DeepseekV2AttentionRef
from rtp_llm.ops import KVCache, PyAttentionInputs
from rtp_llm.utils.model_weight import W


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads):
    bs_page_num, page_size, ckv_dim = ckv.shape
    page_num = bs_page_num // batch_size
    _, _, kpe_dim = kpe.shape
    ckv = ckv.view(batch_size, page_num * page_size, ckv_dim)
    kpe = kpe.view(batch_size, page_num * page_size, kpe_dim)
    ckv = ckv[:, :kv_len, :]
    kpe = kpe[:, :kv_len, :]
    k = (
        torch.cat([ckv, kpe], dim=-1)
        .view(-1, 1, ckv_dim + kpe_dim)
        .repeat_interleave(num_heads, dim=1)
    )
    v = ckv.repeat_interleave(num_heads, dim=1)

    return k, v


def create_cos_sin_cache():
    rotary_emb = DeepseekV3YarnRotaryEmbedding(
        64,
        163840,
        10000,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=0.707,
        mscale_all_dim=0.707,
    )
    half_rope_dim = 64 // 2
    cos_cache = rotary_emb.cos_cached[:, :half_rope_dim]
    sin_cache = rotary_emb.sin_cached[:, :half_rope_dim]
    # cos sin cache must be float32
    cos_sin_cache = (
        torch.cat([cos_cache, sin_cache], dim=-1)
        .contiguous()
        .to(device)
        .to(torch.float32)
    )
    return cos_sin_cache


class MLATest(TestCase):
    NUM_TOKENS = [7]
    HIDDEN_SIZES = [2048]
    PAGE_SIZE = [64]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device(device)

    def _run_mla_test(self, num_tokens: int, hidden_size: int, page_size: int):
        sequence_lengths = [2]

        batch_size = len(sequence_lengths)
        num_tokens = len(sequence_lengths)

        seq_page_sizes = [math.ceil(x / page_size) for x in sequence_lengths]
        kvcache_block_id = torch.zeros(
            [batch_size, max(seq_page_sizes)],
            dtype=torch.int32,
            device=torch.device("cpu"),
        )
        bias = 1
        for i in range(batch_size):
            kvcache_block_id[i, : seq_page_sizes[i]] = torch.arange(
                bias,
                bias + seq_page_sizes[i],
                dtype=torch.int32,
                device=torch.device("cpu"),
            )
            bias += seq_page_sizes[i]

        self.config = GptInitModelParameters(128, 16, 27, 1024, 102400)
        self.config.head_num = 16
        self.config.hidden_size = hidden_size
        self.config.nope_head_dim = 128
        self.config.rope_head_dim = 64
        self.config.kv_lora_rank = 512
        self.config.v_head_dim = 128
        self.config.q_lora_rank = 0
        self.config.seq_size_per_block = 64
        self.config.softmax_extra_scale = 1.0
        self.config.use_mla = True
        self.config.size_per_head = 192

        torch.manual_seed(0)
        sequence_lengths_mius_1 = [x - 1 for x in sequence_lengths]
        sequence_lengths_t = torch.tensor(
            sequence_lengths_mius_1, dtype=torch.int32, device=torch.device("cpu")
        )
        prefix_lengths_t = torch.zeros(
            len(sequence_lengths_t) - len(sequence_lengths) + 1,
            dtype=torch.int32,
            device=torch.device("cpu"),
        )

        attn_inputs: PyAttentionInputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.prefix_lengths = prefix_lengths_t
        attn_inputs.sequence_lengths = torch.tensor(
            [], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.input_lengths = sequence_lengths_t
        attn_inputs.kv_cache_block_id_host = kvcache_block_id

        weights = {}
        weights[W.mla_fusedqkrope_no_lora_w] = torch.randn(
            [
                self.config.hidden_size,
                self.config.size_per_head * self.config.head_num
                + self.config.kv_lora_rank
                + self.config.rope_head_dim,
            ],
            dtype=torch.float16,
            device=device,
        )

        weights[W.mla_kv_a_ln_gamma] = torch.randn(
            [self.config.kv_lora_rank], dtype=torch.float16, device=device
        )

        weights[W.mla_kc] = torch.randn(
            [self.config.head_num, self.config.nope_head_dim, self.config.kv_lora_rank],
            dtype=torch.float16,
            device=device,
        )

        weights[W.mla_vc] = torch.randn(
            [self.config.head_num, self.config.kv_lora_rank, self.config.v_head_dim],
            dtype=torch.float16,
            device=device,
        )

        weights[W.mla_v_w] = torch.randn(
            [self.config.kv_lora_rank, hidden_size],
            dtype=torch.float16,
            device=device,
        )

        weights[W.mla_k_nope_w] = torch.randn(
            [self.config.kv_lora_rank, hidden_size],
            dtype=torch.float16,
            device=device,
        )

        weights[W.attn_o_w] = torch.randn(
            [
                self.config.head_num * self.config.v_head_dim,
                self.config.hidden_size,
            ],
            dtype=torch.float16,
            device=device,
        )

        layer_weights: List[Dict[str, torch.Tensor]] = []
        layer_weights.append(weights)

        fmha_impl = MlaFlashInferPrefillImpl(
            self.config, attn_inputs, layer_weights, create_cos_sin_cache()
        )
        deepseekv2_mla = DeepSeekV2Attention(self.config, weights, 0)
        kv_cache: Optional[KVCache] = None
        deepseekv2_mla_ref = DeepseekV2AttentionRef(self.config, weights, 0)

        hidden = torch.randn(
            [num_tokens, self.config.hidden_size],
            dtype=torch.float16,
            device=device,
        )

        out = deepseekv2_mla(hidden, fmha_impl, kv_cache)
        out_ref = deepseekv2_mla_ref(hidden)

        self.assertTrue(torch.allclose(out, out_ref, atol=1, rtol=1))

    def test_mlp(self):
        for params in itertools.product(
            self.NUM_TOKENS, self.HIDDEN_SIZES, self.PAGE_SIZE
        ):
            with self.subTest(
                num_tokens=params[0], hidden_size=params[1], page_size=params[2]
            ):
                self._run_mla_test(*params)


if __name__ == "__main__":
    main()
