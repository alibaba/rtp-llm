import itertools
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional
from unittest import SkipTest, TestCase, main

import torch
import torch.nn.functional as F

device = torch.device(f"cuda")

import flashinfer.page as page

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models.rotary_embedding.deepseek_rotary_embedding import (
    DeepseekV3YarnRotaryEmbedding,
)
from rtp_llm.models_py.modules import LinearFactory
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashInferPrefillImpl,
)
from rtp_llm.models_py.modules.hybrid.test.mla_attention_ref import attention_ref
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs
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
    NUM_TOKENS = [7, 2000]
    HIDDEN_SIZES = [2048]
    PAGE_SIZE = [64]
    REUSE_LEN = [0, 128]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device(device)

    def _run_mla_test(
        self, num_tokens: int, hidden_size: int, page_size: int, reuse_len: int
    ):

        input_lengths = [num_tokens]
        mock_page_num = 2048
        page_num = math.ceil(reuse_len + num_tokens + page_size - 1 / page_size)
        block_list = [i for i in range(1, page_num + 1)]
        # print(f"block_list: {block_list}")
        kvcache_block_id = torch.tensor(
            [block_list],
            dtype=torch.int32,
            device=torch.device("cpu"),
        )

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
        self.scaling = (self.config.nope_head_dim + self.config.rope_head_dim) ** (-0.5)

        torch.manual_seed(0)
        input_lengths_t = torch.tensor(
            input_lengths, dtype=torch.int32, device=torch.device("cpu")
        )
        prefix_lengths_t = torch.tensor(
            [reuse_len],
            dtype=torch.int32,
            device=torch.device("cpu"),
        )

        attn_inputs: PyAttentionInputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.prefix_lengths = prefix_lengths_t
        attn_inputs.sequence_lengths = torch.tensor(
            [], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.input_lengths = input_lengths_t
        attn_inputs.kv_cache_block_id_host = kvcache_block_id

        weights = self._create_weights(self.config, hidden_size)
        layer_weights: List[Dict[str, torch.Tensor]] = [weights]

        cos_sin_cache = create_cos_sin_cache()
        fmha_impl = MlaFlashInferPrefillImpl(
            self.config, attn_inputs, layer_weights, cos_sin_cache
        )

        q = torch.randn(
            [
                num_tokens,
                self.config.head_num,
                self.config.nope_head_dim + self.config.rope_head_dim,
            ],
            dtype=torch.bfloat16,
            device=device,
        )

        compressed_kv = torch.randn(
            [num_tokens, self.config.kv_lora_rank],
            dtype=torch.bfloat16,
            device=device,
        )

        k_pe = torch.randn(
            [num_tokens, self.config.rope_head_dim],
            dtype=torch.bfloat16,
            device=device,
        )

        cache = torch.randn(
            [
                mock_page_num,
                page_size,
                self.config.kv_lora_rank + self.config.rope_head_dim,
            ],
            dtype=torch.bfloat16,
            device=device,
        )

        kv_cache: Optional[KVCache] = KVCache()
        kv_cache.k_cache_base = cache

        k_cache, v_cache = torch.split(
            kv_cache.k_cache_base,
            [self.config.kv_lora_rank, self.config.rope_head_dim],
            dim=-1,
        )
        page.append_paged_mla_kv_cache(
            compressed_kv,
            k_pe,
            fmha_impl.rope_params.batch_indice,
            fmha_impl.rope_params.positions,
            k_cache,
            v_cache,
            fmha_impl.rope_params.page_indice,
            fmha_impl.rope_params.decode_page_indptr,
            fmha_impl.rope_params.paged_kv_last_page_len,
        )

        out = fmha_impl.compute_prefill_context(q, compressed_kv, k_pe, kv_cache, 0)

        index_list = torch.empty(0, dtype=torch.int32, device=device)
        if fmha_impl.fmha_params.reuse_cache_page_indice is not None:
            index_list = fmha_impl.fmha_params.reuse_cache_page_indice.clone()
        selected_blocks = cache[index_list]
        selected_blocks = selected_blocks.view(-1, selected_blocks.size(-1))

        compressed_kv = torch.cat(
            [selected_blocks[:, : compressed_kv.size(1)], compressed_kv], dim=0
        )
        k_pe = k_pe.view(-1, self.config.rope_head_dim)
        k_pe = torch.cat([selected_blocks[:, compressed_kv.size(1) :], k_pe], dim=0)

        k_pe = k_pe.view(-1, 1, self.config.rope_head_dim)
        self.k_nope_proj = LinearFactory.create_linear_from_weights(
            layer_weights[0], W.mla_k_nope_w, W.mla_k_nope_s, None, self.config
        )

        self.v_proj = LinearFactory.create_linear_from_weights(
            layer_weights[0], W.mla_v_w, W.mla_v_s, None, self.config
        )

        k_nope = self.k_nope_proj(compressed_kv)
        value_states = self.v_proj(compressed_kv)

        k_nope = k_nope.view(-1, self.config.head_num, self.config.nope_head_dim)
        value_states = value_states.view(
            -1, self.config.head_num, self.config.v_head_dim
        )

        k = k_pe.new_empty(
            k_pe.size(0),
            self.config.head_num,
            self.config.rope_head_dim + self.config.nope_head_dim,
        )
        k[..., : self.config.nope_head_dim] = k_nope
        k[..., self.config.nope_head_dim :] = k_pe
        out_ref, _ = attention_ref(
            1,
            q,
            k,
            value_states,
            causal=True,
            sm_scale=self.scaling,
        )
        out_norm = out / (torch.norm(out) + 1e-8)
        out_ref_norm = out_ref / (torch.norm(out_ref) + 1e-8)
        self.assertTrue(torch.allclose(out_norm, out_ref_norm, atol=0.01, rtol=0.01))
        out_flat = out.flatten()
        out_ref_flat = out_ref.flatten()
        # 计算余弦相似度
        cosine_sim = F.cosine_similarity(
            out_flat.unsqueeze(0), out_ref_flat.unsqueeze(0), dim=1
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor(1.0).to(device).to(cosine_sim.dtype),
                cosine_sim,
                atol=0.01,
                rtol=0.01,
            )
        )

    def _create_weights(self, config, hidden_size):
        """创建测试权重"""
        weights = {}
        weights[W.mla_fusedqkrope_no_lora_w] = torch.randn(
            [
                config.hidden_size,
                config.size_per_head * config.head_num
                + config.kv_lora_rank
                + config.rope_head_dim,
            ],
            dtype=torch.bfloat16,
            device=device,
        )

        weights[W.mla_kv_a_ln_gamma] = torch.randn(
            [config.kv_lora_rank], dtype=torch.bfloat16, device=device
        )

        weights[W.mla_kc] = torch.randn(
            [config.head_num, config.nope_head_dim, config.kv_lora_rank],
            dtype=torch.bfloat16,
            device=device,
        )

        weights[W.mla_vc] = torch.randn(
            [config.head_num, config.kv_lora_rank, config.v_head_dim],
            dtype=torch.bfloat16,
            device=device,
        )

        weights[W.mla_v_w] = torch.randn(
            [config.kv_lora_rank, hidden_size],
            dtype=torch.bfloat16,
            device=device,
        )

        weights[W.mla_k_nope_w] = torch.randn(
            [config.kv_lora_rank, hidden_size],
            dtype=torch.bfloat16,
            device=device,
        )

        weights[W.mla_kc] = (
            weights[W.mla_k_nope_w]
            .view(config.kv_lora_rank, config.head_num, config.nope_head_dim)
            .transpose(0, 1)
            .transpose(1, 2)
        )
        weights[W.mla_vc] = (
            weights[W.mla_v_w]
            .view(config.kv_lora_rank, config.head_num, config.v_head_dim)
            .transpose(0, 1)
        )

        weights[W.attn_o_w] = torch.randn(
            [
                config.head_num * config.v_head_dim,
                config.hidden_size,
            ],
            dtype=torch.bfloat16,
            device=device,
        )

        return weights

    def test_mlp(self):
        for params in itertools.product(
            self.NUM_TOKENS, self.HIDDEN_SIZES, self.PAGE_SIZE, self.REUSE_LEN
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                page_size=params[2],
                reuse_len=params[3],
            ):
                self._run_mla_test(*params)


if __name__ == "__main__":
    main()
