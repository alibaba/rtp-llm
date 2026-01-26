"""
Test SparseMlaFp8Op correctness for Decode stage.

This module contains unittest test cases for testing SparseMlaFp8Op with FP8 quantized
KV cache, following FlashMLA test design. Tests cover various configurations including:
- Basic decode configurations with different batch sizes and sequence lengths
- Production-like configurations
- Corner cases (invalid indices, zero sequence lengths)

Usage:
    python sparse_mla_decode_op_test.py
    python -m unittest sparse_mla_decode_op_test
    python -m unittest sparse_mla_decode_op_test.SparseMlaFp8DecodeOpTest.test_basic_decode_configurations
"""

import enum
import math
import sys
from typing import Tuple
from unittest import SkipTest, TestCase, main

import torch
import torch.nn.functional as F

sys.path.append("/data2/baowending.bwd/new/RTP-LLM/github-opensource/")

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
    SparseMlaFp8Op,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FP8KVCacheLayout(enum.Enum):
    """FP8 KV Cache Layout types following FlashMLA."""

    V32_FP8Sparse = 1
    MODEL1_FP8Sparse = 2

    def get_meta(self) -> Tuple[int, int, int, int, int]:
        """Return (d, d_nope, d_rope, tile_size, num_tiles)."""
        return {
            FP8KVCacheLayout.V32_FP8Sparse: (576, 512, 64, 128, 4),
            FP8KVCacheLayout.MODEL1_FP8Sparse: (512, 448, 64, 64, 7),
        }[self]


def _cast_scale_inv_to_ue8m0(
    scales_inv: torch.Tensor, out_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Cast scale to UE8M0 format (power of 2)."""
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil()).to(out_dtype)


def quantize_k_cache(
    input_k_cache: torch.Tensor,
    kvcache_layout: FP8KVCacheLayout,
) -> torch.Tensor:
    """
    Quantize K cache to FP8 format.

    Args:
        input_k_cache: [num_blocks, block_size, h_k, d]
        kvcache_layout: FP8 layout type

    Returns:
        Quantized K cache in FP8 format
    """
    d, d_nope, d_rope, tile_size, num_tiles = kvcache_layout.get_meta()
    assert input_k_cache.shape[-1] == d
    num_blocks, block_size, h_k, _ = input_k_cache.shape
    assert h_k == 1
    input_k_cache = input_k_cache.squeeze(2)
    input_elem_size = input_k_cache.element_size()

    if kvcache_layout == FP8KVCacheLayout.V32_FP8Sparse:
        bytes_per_token = d_nope + num_tiles * 4 + input_elem_size * d_rope
        result = torch.empty(
            (num_blocks, block_size + 1, bytes_per_token),
            dtype=torch.float8_e4m3fn,
            device=input_k_cache.device,
        )[:, :block_size, :]

        result_k_nope_part = result[..., :d_nope]
        result_k_scale_factor = result[..., d_nope : d_nope + num_tiles * 4].view(
            torch.float32
        )
        result_k_rope_part = result[..., d_nope + num_tiles * 4 :].view(
            input_k_cache.dtype
        )
        result_k_rope_part[:] = input_k_cache[..., d_nope:]

        for tile_idx in range(0, num_tiles):
            cur_scale_factors_inv = (
                torch.abs(
                    input_k_cache[
                        ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
                    ]
                )
                .max(dim=-1)
                .values.float()
                / 448.0
            )
            cur_scale_factors_inv = _cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
            result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv

            cur_scale_factors_inv.unsqueeze_(-1)
            cur_quantized_nope = (
                input_k_cache[
                    ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
                ].float()
                / cur_scale_factors_inv.float()
            ).to(torch.float8_e4m3fn)
            result_k_nope_part[
                ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
            ] = cur_quantized_nope

        result = result.view(num_blocks, block_size, 1, -1)
        return result

    elif kvcache_layout == FP8KVCacheLayout.MODEL1_FP8Sparse:
        bytes_per_token = d_nope + 2 * d_rope + num_tiles + 1
        size_per_block_padded = (block_size * bytes_per_token + 576 - 1) // 576 * 576
        result = torch.empty(
            (num_blocks, size_per_block_padded),
            dtype=torch.float8_e4m3fn,
            device=input_k_cache.device,
        )[:, : block_size * bytes_per_token]

        result_k_nope_rope_part = result[:, : block_size * (d_nope + 2 * d_rope)].view(
            num_blocks, block_size, d_nope + 2 * d_rope
        )
        result_k_nope = result_k_nope_rope_part[:, :, :d_nope]
        result_k_rope = result_k_nope_rope_part[:, :, d_nope:].view(input_k_cache.dtype)
        result_k_scale_factor = (
            result[:, block_size * (d_nope + 2 * d_rope) :]
            .view(num_blocks, block_size, 8)[:, :, :7]
            .view(torch.float8_e8m0fnu)
        )

        result_k_rope[:] = input_k_cache[..., d_nope:]
        for tile_idx in range(0, num_tiles):
            cur_scale_factors_inv = (
                torch.abs(
                    input_k_cache[
                        ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
                    ]
                )
                .max(dim=-1)
                .values.float()
                / 448.0
            )
            cur_scale_factors_inv = _cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
            result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv.to(
                torch.float8_e8m0fnu
            )

            cur_scale_factors_inv = cur_scale_factors_inv.view(
                num_blocks, block_size, 1
            )
            cur_quantized_nope = (
                input_k_cache[
                    ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
                ].float()
                / cur_scale_factors_inv.float()
            ).to(torch.float8_e4m3fn)
            result_k_nope[:, :, tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
                cur_quantized_nope
            )

        result = result.view(num_blocks, block_size, 1, -1)
        return result

    else:
        raise NotImplementedError(f"Unsupported kvcache_layout: {kvcache_layout}")


def dequantize_k_cache(
    quant_k_cache: torch.Tensor,
    kvcache_layout: FP8KVCacheLayout,
) -> torch.Tensor:
    """
    Dequantize K cache from FP8 format to bfloat16.

    Args:
        quant_k_cache: [num_blocks, block_size, 1, bytes_per_token]
        kvcache_layout: FP8 layout type

    Returns:
        Dequantized K cache in bfloat16
    """
    d, d_nope, d_rope, tile_size, num_tiles = kvcache_layout.get_meta()
    num_blocks, block_size, h_k, _ = quant_k_cache.shape
    assert h_k == 1
    result = torch.empty(
        (num_blocks, block_size, d), dtype=torch.bfloat16, device=quant_k_cache.device
    )

    if kvcache_layout == FP8KVCacheLayout.V32_FP8Sparse:
        quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)

        input_nope = quant_k_cache[..., :d_nope]
        input_scale = quant_k_cache[..., d_nope : d_nope + num_tiles * 4].view(
            torch.float32
        )
        input_rope = quant_k_cache[..., d_nope + num_tiles * 4 :].view(torch.bfloat16)
        result[..., d_nope:] = input_rope

        for tile_idx in range(0, num_tiles):
            cur_nope = input_nope[
                ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
            ].to(torch.float32)
            cur_scales = input_scale[..., tile_idx].unsqueeze(-1)
            result[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
                cur_nope * cur_scales
            )

    elif kvcache_layout == FP8KVCacheLayout.MODEL1_FP8Sparse:
        quant_k_cache = quant_k_cache.view(num_blocks, -1)

        input_nope_rope = quant_k_cache[:, : block_size * (d_nope + 2 * d_rope)].view(
            num_blocks, block_size, d_nope + 2 * d_rope
        )
        input_nope = input_nope_rope[:, :, :d_nope]
        input_rope = input_nope_rope[:, :, d_nope:].view(torch.bfloat16)
        input_scale = (
            quant_k_cache[:, block_size * (d_nope + 2 * d_rope) :]
            .view(num_blocks, block_size, 8)[:, :, :7]
            .view(torch.float8_e8m0fnu)
        )

        result[..., d_nope:] = input_rope
        for tile_idx in range(0, num_tiles):
            cur_nope = input_nope[
                ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
            ].to(torch.bfloat16)
            cur_scales = input_scale[:, :, tile_idx].to(torch.bfloat16).unsqueeze(-1)
            result[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
                cur_nope * cur_scales
            )

    else:
        raise NotImplementedError(f"Unsupported kvcache_layout: {kvcache_layout}")

    result = result.view(num_blocks, block_size, 1, d)
    return result


class TestParam:
    """Test parameters for decode stage."""

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        num_kv_heads: int,
        seq_len_kv: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        page_size: int,
        top_k: int,
        softmax_extra_scale: float = 1.0,
        seed: int = 42,
        is_all_indices_invalid: bool = False,
        have_zero_seqlen_k: bool = False,
    ):
        self.num_tokens = batch_size
        self.batch_size = batch_size
        self.total_cache_len = seq_len_kv * batch_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.page_size = page_size
        self.top_k = top_k
        self.softmax_extra_scale = softmax_extra_scale
        self.seed = seed
        self.seq_len_kv = seq_len_kv
        self.is_all_indices_invalid = is_all_indices_invalid
        self.have_zero_seqlen_k = have_zero_seqlen_k

    def __str__(self):
        return (
            f"TestParam(batch_size={self.batch_size}, "
            f"seq_len_kv={self.seq_len_kv}, "
            f"num_heads={self.num_heads}, "
            f"kv_lora_rank={self.kv_lora_rank}, "
            f"qk_head_dim={self.qk_head_dim}, "
            f"page_size={self.page_size}, "
            f"top_k={self.top_k}, "
            f"invalid={self.is_all_indices_invalid}, "
            f"zero_len={self.have_zero_seqlen_k})"
        )


class Testcase:
    """Test case data container."""

    def __init__(
        self,
        q: torch.Tensor,
        kv_fp8: torch.Tensor,
        kv_dequant: torch.Tensor,
        topk_indices: torch.Tensor,
        block_table: torch.Tensor,
        mla_params: rtp_llm_ops.FlashInferMlaAttnParams,
        scale: float,
        sequence_lengths: torch.Tensor,
    ):
        self.q = q
        self.kv_fp8 = kv_fp8
        self.kv_dequant = kv_dequant
        self.topk_indices = topk_indices
        self.block_table = block_table
        self.mla_params = mla_params
        self.scale = scale
        self.sequence_lengths = sequence_lengths


def generate_block_table(
    batch_size: int, seq_len_per_batch: int, page_size: int
) -> torch.Tensor:
    """Generate block table for paged KV cache."""
    num_blocks_per_seq = math.ceil(seq_len_per_batch / page_size)
    block_table = torch.zeros(
        [batch_size, num_blocks_per_seq],
        dtype=torch.int32,
        device=torch.device("cpu"),
    )

    bias = 0
    for i in range(batch_size):
        block_table[i, :] = torch.arange(
            bias,
            bias + num_blocks_per_seq,
            dtype=torch.int32,
            device=torch.device("cpu"),
        )
        bias += num_blocks_per_seq

    return block_table


def generate_testcase(p: TestParam) -> Testcase:
    """
    Generate test case with FP8 quantization flow.

    Flow:
    1. Generate random Q and KV cache
    2. Quantize KV to FP8 (for SparseMlaFp8Op)
    3. Dequantize KV back to bfloat16 (for reference implementation)
    """
    set_seed(p.seed)
    device = torch.device("cuda")

    q = (
        torch.randn(
            [p.num_tokens, p.num_heads, p.qk_head_dim],
            dtype=torch.bfloat16,
            device=device,
        )
        / 10.0
    )
    q.clamp_(-1, 1)

    block_table_host = generate_block_table(p.batch_size, p.seq_len_kv, p.page_size)
    block_table_device = block_table_host.to(device)
    num_blocks_per_seq = block_table_host.shape[1]
    total_blocks = p.batch_size * num_blocks_per_seq

    kv_original = (
        torch.randn(
            [total_blocks, p.page_size, p.num_kv_heads, p.qk_head_dim],
            dtype=torch.bfloat16,
            device=device,
        )
        / 10.0
    )
    kv_original.clamp_(-1, 1)

    if p.qk_head_dim == 576:
        fp8_layout = FP8KVCacheLayout.V32_FP8Sparse
    elif p.qk_head_dim == 512:
        fp8_layout = FP8KVCacheLayout.MODEL1_FP8Sparse
    else:
        raise ValueError(f"Unsupported qk_head_dim: {p.qk_head_dim}")

    kv_fp8 = quantize_k_cache(kv_original, fp8_layout)
    kv_dequant = dequantize_k_cache(kv_fp8, fp8_layout)
    kv_dequant = kv_dequant.view(-1, p.num_kv_heads, p.qk_head_dim)

    if p.is_all_indices_invalid:
        topk_indices = torch.full(
            [p.num_tokens, p.num_kv_heads, p.top_k],
            -1,
            dtype=torch.int32,
            device=device,
        )
    else:
        topk_indices_2d = torch.randint(
            0, p.seq_len_kv, [p.num_tokens, p.top_k], dtype=torch.int32, device=device
        )
        topk_indices = (
            topk_indices_2d.unsqueeze(1)
            .expand(p.num_tokens, p.num_kv_heads, p.top_k)
            .contiguous()
        )

    mla_params = rtp_llm_ops.FlashInferMlaAttnParams()

    if p.have_zero_seqlen_k:
        sequence_lengths = torch.tensor(
            [p.seq_len_kv if i % 2 == 0 else 0 for i in range(p.batch_size)],
            dtype=torch.int32,
            device=torch.device("cpu"),
        )
    else:
        sequence_lengths = torch.tensor(
            [p.seq_len_kv] * p.batch_size, dtype=torch.int32, device=torch.device("cpu")
        )

    input_lengths = torch.ones(
        p.batch_size, dtype=torch.int32, device=torch.device("cpu")
    )
    prefix_lengths = torch.tensor([], dtype=torch.int32, device=torch.device("cpu"))

    mla_params.fill_params(
        prefix_lengths, sequence_lengths, input_lengths, block_table_host, p.page_size
    )
    scale = (p.qk_head_dim**-0.5) * p.softmax_extra_scale

    return Testcase(
        q,
        kv_fp8,
        kv_dequant,
        topk_indices,
        block_table_device,
        mla_params,
        scale,
        sequence_lengths,
    )


def ref_sparse_mla_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_indices_global: torch.Tensor,
    scale: float,
    kv_lora_rank: int,
    sequence_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation of sparse MLA attention using PyTorch.
    Uses dequantized KV cache for ground truth.

    Args:
        q: [num_tokens, num_heads, qk_head_dim]
        kv: [total_cache_len, h_kv, d_qk]
        topk_indices_global: [num_tokens, h_kv, top_k]
        scale: softmax scale
        kv_lora_rank: output dimension
        sequence_lengths: [batch_size]

    Returns:
        output: [num_tokens, num_heads, kv_lora_rank]
    """
    num_tokens, num_heads, _ = q.shape
    h_kv = topk_indices_global.shape[1]

    q_fp32 = q.float()
    kv_fp32 = kv.float()

    output = torch.zeros(
        [num_tokens, num_heads, kv_lora_rank], dtype=torch.float32, device=q.device
    )

    for token_idx in range(num_tokens):
        batch_idx = token_idx
        seq_len = sequence_lengths[batch_idx].item()

        if seq_len == 0:
            continue

        indices = topk_indices_global[token_idx]

        for kv_head_idx in range(h_kv):
            indices_per_head = indices[kv_head_idx]

            valid_mask = (indices_per_head >= 0) & (indices_per_head < kv_fp32.shape[0])
            if not valid_mask.any():
                continue

            indices_clamped = torch.clamp(indices_per_head, 0, kv_fp32.shape[0] - 1)
            gathered_kv = kv_fp32[indices_clamped, kv_head_idx, :]

            q_token = q_fp32[token_idx, :, :]
            attn_scores = torch.matmul(q_token, gathered_kv.T) * scale
            attn_scores[:, ~valid_mask] = float("-inf")

            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, 0.0)

            gathered_v = gathered_kv[:, :kv_lora_rank]
            output_token = torch.matmul(attn_weights, gathered_v)

            if kv_head_idx == 0:
                output[token_idx] = output_token
            else:
                output[token_idx] += output_token

    return output.to(q.dtype)


class SparseMlaFp8DecodeOpTest(TestCase):
    """Test SparseMlaFp8Op for decode stage."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test environment once for all tests."""
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        cls.device = torch.device("cuda:0")
        torch.cuda.set_device(cls.device)

    def setUp(self) -> None:
        """Set up before each test."""
        torch.set_default_device(self.device)
        torch.set_default_dtype(torch.bfloat16)
        torch.cuda.empty_cache()

    def tearDown(self) -> None:
        """Clean up after each test."""
        torch.cuda.empty_cache()

    def _run_test(self, p: TestParam):
        """Run single test case."""
        # Log test parameters
        test_info = f"Running test: {p}"
        print(f"\n{'='*80}\n{test_info}\n{'='*80}")

        testcase = generate_testcase(p)

        sparse_mla_fp8_op = SparseMlaFp8Op(
            num_heads=p.num_heads,
            kv_lora_rank=p.kv_lora_rank,
            qk_rope_head_dim=p.qk_rope_head_dim,
            qk_nope_head_dim=p.qk_nope_head_dim,
            page_size=p.page_size,
            softmax_extra_scale=p.softmax_extra_scale,
            top_k=p.top_k,
        )

        sparse_mla_fp8_op.plan(testcase.mla_params, testcase.block_table)

        output = sparse_mla_fp8_op.forward(
            testcase.q, testcase.kv_fp8, testcase.topk_indices
        )
        torch.cuda.synchronize()

        # Convert local indices to global for reference implementation
        global_indices = sparse_mla_fp8_op._convert_topk_indices_to_global(  # type: ignore[attr-defined]
            testcase.topk_indices
        )

        ref_output = ref_sparse_mla_forward(
            testcase.q,
            testcase.kv_dequant,
            global_indices,
            testcase.scale,
            p.kv_lora_rank,
            testcase.sequence_lengths,
        )
        torch.cuda.synchronize()

        # Calculate error metrics
        abs_diff = torch.abs(output - ref_output)
        rel_diff = abs_diff / (torch.abs(ref_output) + 1e-8)
        max_abs_error = torch.max(abs_diff).item()
        max_rel_error = torch.max(rel_diff).item()
        mean_abs_error = torch.mean(abs_diff).item()

        output_flat = output.flatten()
        ref_output_flat = ref_output.flatten()

        output_is_zero = torch.allclose(
            output_flat, torch.zeros_like(output_flat), atol=1e-6
        )
        ref_is_zero = torch.allclose(
            ref_output_flat, torch.zeros_like(ref_output_flat), atol=1e-6
        )

        if output_is_zero and ref_is_zero:
            cosine_sim = 1.0
            print(f"\nCorrectness check:")
            print(f"  Zero output (all indices invalid or zero sequence length)")
            print(f"  ✓ Test passed!")
            return True
        elif output_is_zero or ref_is_zero:
            cosine_sim = 0.0
        else:
            cosine_sim = F.cosine_similarity(
                output_flat.unsqueeze(0), ref_output_flat.unsqueeze(0), dim=1
            ).item()

        print(f"\nCorrectness check:")
        print(f"  Max absolute error: {max_abs_error:.6f}")
        print(f"  Max relative error: {max_rel_error:.6f}")
        print(f"  Mean absolute error: {mean_abs_error:.6f}")
        print(f"  Cosine similarity: {cosine_sim:.6f}")

        if p.have_zero_seqlen_k:
            self.assertGreater(
                cosine_sim,
                0.5,
                f"Cosine similarity too low for zero-length case: {cosine_sim:.6f} "
                f"(max_abs_error={max_abs_error:.6f}, max_rel_error={max_rel_error:.6f})",
            )
        else:
            self.assertGreater(
                cosine_sim,
                0.95,
                f"Cosine similarity too low: {cosine_sim:.6f} "
                f"(max_abs_error={max_abs_error:.6f}, max_rel_error={max_rel_error:.6f}, "
                f"mean_abs_error={mean_abs_error:.6f})",
            )

        print(f"✓ Test passed!")

    def test_basic_decode_configurations(self):
        """
        Test basic decode configurations with FP8 quantization.

        Covers:
        - Different batch sizes (1, 2, 4)
        - Different sequence lengths (128, 512, 1024, 2048)
        - Different top_k values (128, 256)
        - Different num_heads (64, 128)
        - Different d_qk (512, 576) for different FP8 layouts
        """
        test_cases = [
            (1, 128, 128, 64, 448, 512, 64, "single_batch_small_cache_MODEL1"),
            (1, 512, 128, 64, 448, 512, 64, "single_batch_medium_cache_MODEL1"),
            (2, 512, 128, 64, 448, 512, 64, "2batch_medium_cache_MODEL1"),
            (4, 1024, 128, 64, 448, 512, 64, "4batch_large_cache_MODEL1"),
            (1, 512, 128, 64, 512, 512, 64, "d_qk_576_V32"),
            (2, 2048, 256, 128, 512, 512, 64, "large_cache_large_topk_V32"),
        ]

        for (
            batch_size,
            seq_len_kv,
            top_k,
            num_heads,
            qk_nope_dim,
            kv_lora_rank,
            page_size,
            desc,
        ) in test_cases:
            with self.subTest(desc=desc):
                p = TestParam(
                    batch_size=batch_size,
                    num_heads=num_heads,
                    num_kv_heads=1,
                    seq_len_kv=seq_len_kv,
                    kv_lora_rank=kv_lora_rank,
                    qk_rope_head_dim=64,
                    qk_nope_head_dim=qk_nope_dim,
                    page_size=page_size,
                    top_k=top_k,
                )
                self._run_test(p)

    def test_production_configurations(self):
        """
        Test production-like configurations with FP8 quantization.
        Based on FlashMLA production settings.
        """
        test_cases = [
            (2, 128, 1024, 256, 512, 64, "V32_simplified_2batch"),
            (4, 128, 1024, 256, 512, 64, "V32_simplified_4batch"),
            (2, 64, 1024, 128, 448, 64, "MODEL1_simplified_2batch"),
            (4, 64, 1024, 128, 448, 64, "MODEL1_simplified_4batch"),
        ]

        for (
            batch_size,
            num_heads,
            seq_len_kv,
            top_k,
            qk_nope_dim,
            page_size,
            desc,
        ) in test_cases:
            with self.subTest(desc=desc):
                p = TestParam(
                    batch_size=batch_size,
                    num_heads=num_heads,
                    num_kv_heads=1,
                    seq_len_kv=seq_len_kv,
                    kv_lora_rank=512,
                    qk_rope_head_dim=64,
                    qk_nope_head_dim=qk_nope_dim,
                    page_size=page_size,
                    top_k=top_k,
                )
                self._run_test(p)

    def test_corner_cases(self):
        """
        Test corner cases with FP8 quantization.

        Covers:
        - All invalid topk indices
        - Zero sequence lengths
        """
        test_cases = [
            (1, 512, 128, 64, 448, 64, True, False, "all_invalid_indices_MODEL1"),
            (2, 512, 128, 64, 448, 64, False, True, "zero_sequence_length_MODEL1"),
            (1, 512, 128, 64, 512, 64, True, False, "all_invalid_indices_V32"),
        ]

        for (
            batch_size,
            seq_len_kv,
            top_k,
            num_heads,
            qk_nope_dim,
            page_size,
            is_all_invalid,
            have_zero_len,
            desc,
        ) in test_cases:
            with self.subTest(desc=desc):
                p = TestParam(
                    batch_size=batch_size,
                    num_heads=num_heads,
                    num_kv_heads=1,
                    seq_len_kv=seq_len_kv,
                    kv_lora_rank=512,
                    qk_rope_head_dim=64,
                    qk_nope_head_dim=qk_nope_dim,
                    page_size=page_size,
                    top_k=top_k,
                    is_all_indices_invalid=is_all_invalid,
                    have_zero_seqlen_k=have_zero_len,
                )
                self._run_test(p)


if __name__ == "__main__":
    main()
