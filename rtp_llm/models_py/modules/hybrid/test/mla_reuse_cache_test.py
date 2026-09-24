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

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.rotary_embedding.deepseek_rotary_embedding import (
    DeepseekV3YarnRotaryEmbedding,
)
from rtp_llm.models_py.modules import LinearFactory
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashInferPrefillImpl,
)
from rtp_llm.models_py.modules.hybrid.test.mla_attention_ref import attention_ref
from rtp_llm.ops import FMHAConfig, ParallelismConfig, compute_ops
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs
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
        page_num = math.ceil((reuse_len + num_tokens + page_size - 1) / page_size)
        block_list = [i for i in range(1, page_num + 1)]
        # print(f"block_list: {block_list}")
        kvcache_block_id = torch.tensor(
            [block_list],
            dtype=torch.int32,
            device=torch.device("cpu"),
        )

        self.config = ModelConfig()
        self.config.attn_config.head_num = 16
        self.config.hidden_size = hidden_size
        self.config.attn_config.nope_head_dim = 128
        self.config.attn_config.rope_head_dim = 64
        self.config.attn_config.kv_lora_rank = 512
        self.config.attn_config.v_head_dim = 128
        self.config.attn_config.q_lora_rank = 0
        self.config.attn_config.tokens_per_block = 64
        self.config.attn_config.kernel_tokens_per_block = 64
        self.config.attn_config.softmax_extra_scale = 1.0
        self.config.attn_config.use_mla = True
        self.config.attn_config.size_per_head = 192
        self.scaling = (
            self.config.attn_config.nope_head_dim
            + self.config.attn_config.rope_head_dim
        ) ** (-0.5)

        self.parallelism_config = ParallelismConfig()
        self.parallelism_config.tp_size = 1
        self.parallelism_config.tp_rank = 0

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
        attn_inputs.kv_cache_block_id = kvcache_block_id
        attn_inputs.kv_cache_block_id_device = kvcache_block_id.to(device)
        attn_inputs.kv_cache_kernel_block_id = kvcache_block_id
        attn_inputs.kv_cache_kernel_block_id_device = kvcache_block_id.to(device)

        weights = self._create_weights(self.config, hidden_size)
        layer_weights: List[Dict[str, torch.Tensor]] = [weights]

        cos_sin_cache = create_cos_sin_cache()

        fmha_impl = MlaFlashInferPrefillImpl(
            self.config.attn_config,
            attn_inputs,
            layer_weights,
            cos_sin_cache,
            quant_config=self.config.quant_config,
        )

        q = torch.randn(
            [
                num_tokens,
                self.config.attn_config.head_num,
                self.config.attn_config.nope_head_dim
                + self.config.attn_config.rope_head_dim,
            ],
            dtype=torch.bfloat16,
            device=device,
        )

        compressed_kv = torch.randn(
            [num_tokens, self.config.attn_config.kv_lora_rank],
            dtype=torch.bfloat16,
            device=device,
        )

        k_pe = torch.randn(
            [num_tokens, self.config.attn_config.rope_head_dim],
            dtype=torch.bfloat16,
            device=device,
        )

        cache = torch.randn(
            [
                mock_page_num,
                page_size,
                self.config.attn_config.kv_lora_rank
                + self.config.attn_config.rope_head_dim,
            ],
            dtype=torch.bfloat16,
            device=device,
        )

        kv_cache: Optional[LayerKVCache] = LayerKVCache()
        kv_cache.kv_cache_base = cache

        k_cache, v_cache = torch.split(
            kv_cache.kv_cache_base,
            [
                self.config.attn_config.kv_lora_rank,
                self.config.attn_config.rope_head_dim,
            ],
            dim=-1,
        )
        # NewMlaRotaryEmbeddingParams: flashinfer params live under .params
        compute_ops.concat_and_cache_mla(
            compressed_kv,
            k_pe,
            kv_cache.kv_cache_base,
            fmha_impl.rope_params.slot_mapping,
            "auto",
            torch.tensor(1.0, dtype=torch.float32, device=device),
        )

        out = fmha_impl.compute_prefill_context(q, compressed_kv, k_pe, kv_cache, 0)

        index_list = torch.empty(0, dtype=torch.int32, device=device)
        if fmha_impl.fmha_impl.reuse_cache_page_indice is not None:
            index_list = fmha_impl.fmha_impl.reuse_cache_page_indice.clone()
        selected_blocks = cache[index_list]
        selected_blocks = selected_blocks.view(-1, selected_blocks.size(-1))

        compressed_kv = torch.cat(
            [selected_blocks[:, : compressed_kv.size(1)], compressed_kv], dim=0
        )
        k_pe = k_pe.view(-1, self.config.attn_config.rope_head_dim)
        k_pe = torch.cat([selected_blocks[:, compressed_kv.size(1) :], k_pe], dim=0)

        k_pe = k_pe.view(-1, 1, self.config.attn_config.rope_head_dim)
        self.kv_b_proj = LinearFactory.create_linear_from_weights(
            layer_weights[0], W.mla_kv_b_w, W.mla_kv_b_s, None
        )

        kv = self.kv_b_proj(compressed_kv)
        kv = kv.view(
            -1,
            self.config.attn_config.head_num,
            self.config.attn_config.nope_head_dim + self.config.attn_config.v_head_dim,
        )
        k_nope = kv[:, :, : self.config.attn_config.nope_head_dim].contiguous()
        value_states = kv[:, :, self.config.attn_config.nope_head_dim :].contiguous()

        k = k_pe.new_empty(
            k_pe.size(0),
            self.config.attn_config.head_num,
            self.config.attn_config.rope_head_dim
            + self.config.attn_config.nope_head_dim,
        )
        k[..., : self.config.attn_config.nope_head_dim] = k_nope
        k[..., self.config.attn_config.nope_head_dim :] = k_pe
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
                config.attn_config.size_per_head * config.attn_config.head_num
                + config.attn_config.kv_lora_rank
                + config.attn_config.rope_head_dim,
            ],
            dtype=torch.bfloat16,
            device=device,
        )

        weights[W.mla_kv_a_ln_gamma] = torch.randn(
            [config.attn_config.kv_lora_rank], dtype=torch.bfloat16, device=device
        )

        weights[W.mla_kc] = torch.randn(
            [
                config.attn_config.head_num,
                config.attn_config.nope_head_dim,
                config.attn_config.kv_lora_rank,
            ],
            dtype=torch.bfloat16,
            device=device,
        )

        weights[W.mla_vc] = torch.randn(
            [
                config.attn_config.head_num,
                config.attn_config.kv_lora_rank,
                config.attn_config.v_head_dim,
            ],
            dtype=torch.bfloat16,
            device=device,
        )

        weights[W.mla_kv_b_w] = torch.randn(
            [
                config.attn_config.kv_lora_rank,
                config.attn_config.head_num
                * (config.attn_config.nope_head_dim + config.attn_config.v_head_dim),
            ],
            dtype=torch.bfloat16,
            device=device,
        )

        kv_b = weights[W.mla_kv_b_w].view(
            config.attn_config.kv_lora_rank,
            config.attn_config.head_num,
            config.attn_config.nope_head_dim + config.attn_config.v_head_dim,
        )
        weights[W.mla_kc] = (
            kv_b[:, :, : config.attn_config.nope_head_dim].permute(1, 2, 0).contiguous()
        )
        weights[W.mla_vc] = (
            kv_b[:, :, config.attn_config.nope_head_dim :].transpose(0, 1).contiguous()
        )

        weights[W.attn_o_w] = torch.randn(
            [
                config.attn_config.head_num * config.attn_config.v_head_dim,
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

    @staticmethod
    def _chunk_inputs(lengths, prefixes, blocks):
        inputs = PyAttentionInputs()
        inputs.is_prefill = True
        inputs.input_lengths = torch.tensor(lengths, dtype=torch.int32, device="cpu")
        inputs.prefix_lengths = torch.tensor(prefixes, dtype=torch.int32, device="cpu")
        inputs.sequence_lengths = torch.empty(0, dtype=torch.int32, device="cpu")
        inputs.kv_cache_block_id = torch.tensor(blocks, dtype=torch.int32, device="cpu")
        inputs.kv_cache_block_id_device = inputs.kv_cache_block_id.to(device)
        inputs.kv_cache_kernel_block_id = inputs.kv_cache_block_id
        inputs.kv_cache_kernel_block_id_device = inputs.kv_cache_block_id_device
        return inputs

    def _chunk_fixture(self):
        config = ModelConfig()
        attn = config.attn_config
        attn.head_num = 16
        attn.nope_head_dim = 128
        attn.rope_head_dim = 64
        attn.kv_lora_rank = 512
        attn.v_head_dim = 128
        attn.size_per_head = 192
        attn.tokens_per_block = attn.kernel_tokens_per_block = 64
        attn.use_mla = True
        config.hidden_size = 2048
        torch.manual_seed(17)
        weights = self._create_weights(config, config.hidden_size)
        # Fan-in scaling keeps logits and raw outputs in the usual BF16 range.
        for name in (W.mla_kv_b_w, W.mla_kc, W.mla_vc):
            weights[name] = weights[name] / math.sqrt(attn.kv_lora_rank)
        cos_sin = create_cos_sin_cache()
        return attn, weights, cos_sin

    @staticmethod
    def _sentinel_cache(num_pages, block_size):
        # The -7 fill makes writes to unowned slots visible.
        cache = LayerKVCache()
        cache.kv_cache_base = torch.full(
            (num_pages, block_size, 576), -7, dtype=torch.bfloat16, device=device
        )
        return cache

    @staticmethod
    def _written_mask(shape, segments, block_size):
        """Mask of the cache slots touched by (blocks, start, end) segments."""
        mask = torch.zeros(shape, dtype=torch.bool, device=device)
        for blocks, start, end in segments:
            for pos in range(start, end):
                mask[blocks[pos // block_size], pos % block_size] = True
        return mask

    def _assert_cache_isolation(self, cache, before, reference_cache, written):
        # Slots outside the chunk must stay untouched; written slots must match
        # the full-prefill reference bit for bit.
        torch.testing.assert_close(
            cache.kv_cache_base[~written], before[~written], rtol=0, atol=0
        )
        torch.testing.assert_close(
            cache.kv_cache_base[written],
            reference_cache.kv_cache_base[written],
            rtol=0,
            atol=0,
        )

    def _chunked_replay(
        self,
        attn,
        weights,
        cos_sin,
        fmha,
        blocks,
        block_size,
        q,
        ckv,
        kpe,
        budget,
        quant_config=None,
        check_chunk=None,
    ):
        """Replay one prefill in budget-sized chunks against a full-prefill reference.

        check_chunk(impl, start, end) runs before each chunk's forward; every
        chunk's cache writes are checked for isolation, and the concatenated
        outputs must match the reference in raw amplitude.
        """
        length = q.shape[0]
        reference_cache = self._sentinel_cache(max(blocks) + 2, block_size)
        reference_impl = MlaFlashInferPrefillImpl(
            attn,
            self._chunk_inputs([length], [0], [blocks]),
            [weights],
            cos_sin,
            fmha_config=fmha,
            quant_config=quant_config,
        )
        reference = reference_impl.forward(
            q.clone(), ckv, kpe.clone(), reference_cache, 0
        )
        chunk_cache = self._sentinel_cache(max(blocks) + 2, block_size)
        outputs = []
        for start in range(0, length, budget):
            end = min(start + budget, length)
            impl = MlaFlashInferPrefillImpl(
                attn,
                self._chunk_inputs([end - start], [start], [blocks]),
                [weights],
                cos_sin,
                fmha_config=fmha,
                quant_config=quant_config,
            )
            if check_chunk is not None:
                check_chunk(impl, start, end)
            before = chunk_cache.kv_cache_base.clone()
            outputs.append(
                impl.forward(
                    q[start:end].clone(),
                    ckv[start:end],
                    kpe[start:end].clone(),
                    chunk_cache,
                    0,
                )
            )
            written = self._written_mask(
                chunk_cache.kv_cache_base.shape[:2],
                [(blocks, start, end)],
                block_size,
            )
            self._assert_cache_isolation(chunk_cache, before, reference_cache, written)
        # Compare raw values, including amplitude, not normalized directions.
        torch.testing.assert_close(torch.cat(outputs), reference, rtol=0.01, atol=0.01)

    def test_chunked_forward(self):
        attn, weights, cos_sin = self._chunk_fixture()
        length, budget = 130, 64
        blocks = [5, 2, 9]
        q = torch.randn(length, 16, 192, dtype=torch.bfloat16, device=device)
        ckv = torch.randn(length, 512, dtype=torch.bfloat16, device=device)
        kpe = torch.randn(length, 64, dtype=torch.bfloat16, device=device)
        # The same short-tail case exercises both expanded and absorbed suffixes.
        for absorb_len in (0, 1024):
            with self.subTest(absorb_len=absorb_len):
                fmha = FMHAConfig()
                fmha.absorb_opt_len = absorb_len

                def check_chunk(impl, start, end, absorb_len=absorb_len):
                    self.assertEqual(
                        impl.absorb_fmha is not None,
                        start > 0 and end - start < absorb_len,
                    )

                self._chunked_replay(
                    attn,
                    weights,
                    cos_sin,
                    fmha,
                    blocks,
                    64,
                    q,
                    ckv,
                    kpe,
                    budget,
                    check_chunk=check_chunk,
                )

    def test_fp8_weight_prefill_keeps_quantized_kv_projection(self):
        from rtp_llm.config.quant_config import init_quant_config
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            is_deep_gemm_e8m0_used,
        )
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import requant_weight_ue8m0
        from rtp_llm.test.utils.numeric_util import per_block_cast_to_fp8

        attn, weights, cos_sin = self._chunk_fixture()
        quant = init_quant_config("FP8_PER_BLOCK")
        # Match load-time quantization: KV-B is quantized, while the absorbed
        # matrices keep the original BF16 checkpoint values.
        weight = weights[W.mla_kv_b_w].t().contiguous()
        quant_weight, scales = per_block_cast_to_fp8(weight, use_ue8m0=False)
        if is_deep_gemm_e8m0_used():
            quant_weight, scales = requant_weight_ue8m0(quant_weight, scales)
        else:
            quant_weight = quant_weight.reshape(weight.shape[1], weight.shape[0])
            scales = scales.reshape(scales.shape[1], scales.shape[0])
        weights[W.mla_kv_b_w] = quant_weight
        weights[W.mla_kv_b_s] = scales

        fmha = FMHAConfig()
        # Exercise the quantized KV projection in both full and chunked prefill.
        fmha.absorb_opt_len = 0
        length, budget = 130, 64
        blocks = [5, 2, 9]
        q = torch.randn(length, 16, 192, dtype=torch.bfloat16, device=device)
        ckv = torch.randn(length, 512, dtype=torch.bfloat16, device=device)
        kpe = torch.randn(length, 64, dtype=torch.bfloat16, device=device)

        def check_chunk(impl, _start, _end):
            self.assertIsNone(impl.absorb_fmha)

        self._chunked_replay(
            attn,
            weights,
            cos_sin,
            fmha,
            blocks,
            64,
            q,
            ckv,
            kpe,
            budget,
            quant_config=quant,
            check_chunk=check_chunk,
        )

    def test_chunked_batched_forward(self):
        attn, weights, cos_sin = self._chunk_fixture()
        lengths = (130, 193)
        blocks = ([5, 2, 9, 0], [1, 4, 8, 7])
        queries = [
            torch.randn(n, 16, 192, dtype=torch.bfloat16, device=device)
            for n in lengths
        ]
        compressed = [
            torch.randn(n, 512, dtype=torch.bfloat16, device=device) for n in lengths
        ]
        keys = [
            torch.randn(n, 64, dtype=torch.bfloat16, device=device) for n in lengths
        ]
        fmha = FMHAConfig()
        fmha.absorb_opt_len = 0
        cache = self._sentinel_cache(11, 64)
        reference_cache = self._sentinel_cache(11, 64)
        references = []
        for row, length in enumerate(lengths):
            impl = MlaFlashInferPrefillImpl(
                attn,
                self._chunk_inputs([length], [0], [blocks[row]]),
                [weights],
                cos_sin,
                fmha_config=fmha,
            )
            references.append(
                impl.forward(
                    queries[row].clone(),
                    compressed[row],
                    keys[row].clone(),
                    reference_cache,
                    0,
                )
            )
        prefixes = [0, 0]
        # Vary grants and batch membership; the third call has different prefixes.
        for batch in (
            [(0, 64), (1, 64)],
            [(1, 128)],
            [(1, 1), (0, 64)],
            [(0, 2)],
        ):
            batch_lengths = [n for _, n in batch]
            batch_prefixes = [prefixes[row] for row, _ in batch]
            inputs = self._chunk_inputs(
                batch_lengths,
                batch_prefixes,
                [blocks[row] for row, _ in batch],
            )
            impl = MlaFlashInferPrefillImpl(
                attn, inputs, [weights], cos_sin, fmha_config=fmha
            )
            params = impl.fmha_params
            kv_lengths = [p + n for p, n in zip(batch_prefixes, batch_lengths)]
            self.assertEqual(
                params.qo_indptr_h.tolist(),
                [0] + list(itertools.accumulate(batch_lengths)),
            )
            self.assertEqual(params.kvlen_h.tolist(), kv_lengths)
            self.assertEqual(
                params.positions_h.tolist(),
                [
                    pos
                    for prefix, length in zip(batch_prefixes, batch_lengths)
                    for pos in range(prefix, prefix + length)
                ],
            )
            slices = [
                (row, slice(prefixes[row], prefixes[row] + n)) for row, n in batch
            ]
            before = cache.kv_cache_base.clone()
            output = impl.forward(
                torch.cat([queries[row][s] for row, s in slices]),
                torch.cat([compressed[row][s] for row, s in slices]),
                torch.cat([keys[row][s] for row, s in slices]),
                cache,
                0,
            )
            expected = torch.cat([references[row][s] for row, s in slices])
            torch.testing.assert_close(output, expected, atol=0.01, rtol=0.01)
            written = self._written_mask(
                cache.kv_cache_base.shape[:2],
                [(blocks[row], prefixes[row], prefixes[row] + n) for row, n in batch],
                64,
            )
            for row, n in batch:
                prefixes[row] += n
            self._assert_cache_isolation(cache, before, reference_cache, written)

    def test_fp8_kv_chunked_batched_gather(self):
        from rtp_llm.ops import KvCacheDataType

        attn, weights, cos_sin = self._chunk_fixture()
        attn.kv_cache_dtype = KvCacheDataType.FP8
        lengths, blocks = [130, 65], [[5, 2, 9], [1, 4, 0]]
        ckv = [torch.randn(n, 512, dtype=torch.bfloat16) for n in lengths]
        kpe = [torch.randn(n, 64, dtype=torch.bfloat16) for n in lengths]
        cache, reference_cache = LayerKVCache(), LayerKVCache()
        for target in (cache, reference_cache):
            # Native layout: 512 CKV bytes, four FP32 scales, 64 BF16 KPE values.
            target.kv_cache_base = torch.full((11, 64, 656), 0xA5, dtype=torch.uint8)
        full = MlaFlashInferPrefillImpl(
            attn, self._chunk_inputs(lengths, [0, 0], blocks), [weights], cos_sin
        )
        full.kv_cache_write_op.forward(
            torch.cat(ckv), torch.cat(kpe), reference_cache, full.fmha_params
        )
        prefixes = [0, 0]
        for batch in ([(0, 64)], [(1, 64), (0, 64)], [(0, 2), (1, 1)]):
            inputs = self._chunk_inputs(
                [n for _, n in batch],
                [prefixes[row] for row, _ in batch],
                [blocks[row] for row, _ in batch],
            )
            impl = MlaFlashInferPrefillImpl(attn, inputs, [weights], cos_sin)
            slices = [
                (row, slice(prefixes[row], prefixes[row] + n)) for row, n in batch
            ]
            current_ckv = torch.cat([ckv[row][s] for row, s in slices])
            current_kpe = torch.cat([kpe[row][s] for row, s in slices])
            before = cache.kv_cache_base.clone()
            impl.kv_cache_write_op.forward(
                current_ckv, current_kpe, cache, impl.fmha_params
            )
            written = self._written_mask(
                cache.kv_cache_base.shape[:2],
                [(blocks[row], prefixes[row], prefixes[row] + n) for row, n in batch],
                64,
            )
            self._assert_cache_isolation(cache, before, reference_cache, written)
            gathered_ckv, gathered_kpe = impl.fmha_impl._reuse_kv_cache_indexed_batched(
                current_ckv, current_kpe, cache
            )
            if any(prefixes[row] for row, _ in batch):
                slots = [
                    blocks[row][pos // 64] * 64 + pos % 64
                    for row, n in batch
                    for pos in range(prefixes[row] + n)
                ]
                packed = reference_cache.kv_cache_base.view(-1, 656)[slots]
                quantized = packed[:, :512].contiguous().view(torch.float8_e4m3fn)
                scales = packed[:, 512:528].contiguous().view(torch.float32)
                expected_ckv = (
                    (quantized.float().reshape(-1, 4, 128) * scales.unsqueeze(-1))
                    .reshape(-1, 512)
                    .to(torch.bfloat16)
                )
                expected_kpe = packed[:, 528:].contiguous().view(torch.bfloat16)
            else:
                expected_ckv, expected_kpe = current_ckv, current_kpe
            torch.testing.assert_close(gathered_ckv, expected_ckv, rtol=0, atol=0)
            torch.testing.assert_close(gathered_kpe, expected_kpe, rtol=0, atol=0)
            for row, n in batch:
                prefixes[row] += n


if __name__ == "__main__":
    main()
