"""Unit tests for MhaRotaryEmbeddingOp"""

import math
import unittest
from typing import Tuple

import torch
from flashinfer import get_batch_indices_positions, get_seq_lens

from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
    MhaRotaryEmbeddingOp,
)
from rtp_llm.ops import AttentionConfigs, RopeStyle
from rtp_llm.ops.compute_ops import KVCache


class RopeParams:
    """Simple RoPE parameters container for testing"""

    def __init__(
        self,
        batch_indice_d: torch.Tensor,
        positions_d: torch.Tensor,
        page_indice_d: torch.Tensor,
        decode_page_indptr_d: torch.Tensor,
        paged_kv_last_page_len_d: torch.Tensor,
    ):
        self.batch_indice_d = batch_indice_d
        self.positions_d = positions_d
        self.page_indice_d = page_indice_d
        self.decode_page_indptr_d = decode_page_indptr_d
        self.paged_kv_last_page_len_d = paged_kv_last_page_len_d


def create_cos_sin_cache(
    head_dim: int, max_seq_len: int = 2048, base: float = 10000.0, device: str = "cuda"
) -> torch.Tensor:
    """Create cos_sin_cache for RoPE using standard implementation.

    Args:
        head_dim: Head dimension (RoPE dimension)
        max_seq_len: Maximum sequence length
        base: RoPE base frequency
        device: Device to create the cache on

    Returns:
        cos_sin_cache: [max_seq_len, head_dim] tensor with first half cos, second half sin
    """
    # Create inverse frequencies
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )

    # Create position IDs
    t = torch.arange(max_seq_len, dtype=torch.float32)

    # Compute frequencies
    freqs = torch.outer(t, inv_freq)  # [max_seq_len, head_dim/2]

    # Compute cos and sin
    cos = freqs.cos()  # [max_seq_len, head_dim/2]
    sin = freqs.sin()  # [max_seq_len, head_dim/2]

    # Interleave cos and sin to match C++ genBaseCache format
    # Stack [2, max_seq_len, head_dim/2], permute to [max_seq_len, head_dim/2, 2], reshape to [max_seq_len, head_dim]
    cos_sin_cache = (
        torch.stack([cos, sin], dim=0).permute(1, 2, 0).reshape(cos.size(0), -1)
    )

    return cos_sin_cache.contiguous().to(device).to(torch.float32)


def create_test_attn_config(
    head_num: int = 8,
    kv_head_num: int = 4,
    size_per_head: int = 64,
    tokens_per_block: int = 16,
) -> AttentionConfigs:
    """Create AttentionConfigs for testing.

    Args:
        head_num: Number of query heads
        kv_head_num: Number of key-value heads
        size_per_head: Dimension of each head
        tokens_per_block: Tokens per KV cache block

    Returns:
        AttentionConfigs object for testing
    """
    config = AttentionConfigs()
    config.head_num = head_num
    config.kv_head_num = kv_head_num
    config.size_per_head = size_per_head
    config.tokens_per_block = tokens_per_block
    config.rope_config.style = RopeStyle.Base
    config.rope_config.dim = size_per_head
    config.rope_config.base = 10000
    return config


def apply_rope_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation of RoPE (non-interleave style).

    Args:
        q: Query [num_tokens, num_heads, head_dim]
        k: Key [num_tokens, num_kv_heads, head_dim]
        cos_sin_cache: [max_seq_len, head_dim] with first half cos, second half sin
        positions: Position IDs [num_tokens]

    Returns:
        q_rope, k_rope: Rotated query and key tensors
    """
    head_dim = q.size(2)
    half_dim = head_dim // 2

    # Extract cos and sin for the given positions
    cos_sin = cos_sin_cache[positions]  # [num_tokens, head_dim]
    cos_half = cos_sin[:, :half_dim]  # [num_tokens, half_dim]
    sin_half = cos_sin[:, half_dim:]  # [num_tokens, half_dim]

    # Expand cos and sin to full head_dim by repeating for both halves
    # For non-interleave style: [cos, cos] and [sin, sin]
    cos = torch.cat([cos_half, cos_half], dim=-1)  # [num_tokens, head_dim]
    sin = torch.cat([sin_half, sin_half], dim=-1)  # [num_tokens, head_dim]

    # Apply RoPE (non-interleave style)
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Split and rotate for RoPE"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    # Apply to Q: q_rope = q * cos + rotate_half(q) * sin
    q_rope = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))

    # Apply to K: k_rope = k * cos + rotate_half(k) * sin
    k_rope = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))

    return q_rope, k_rope


class TestMhaRotaryEmbeddingOp(unittest.TestCase):
    """Test suite for MhaRotaryEmbeddingOp"""

    def setUp(self):
        """Set up test fixtures"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available - this test requires CUDA")

        self.device = torch.device("cuda")
        torch.manual_seed(42)

    def test_basic_rope_application(self):
        """Test basic RoPE application without KV cache"""
        # Setup
        num_tokens = 7
        num_heads = 8
        num_kv_heads = 4
        head_dim = 64
        max_seq_len = 2048
        token_per_block = 16

        # Create cos_sin_cache
        cos_sin_cache = create_cos_sin_cache(head_dim, max_seq_len, device="cuda")

        # Create attention config
        attn_config = create_test_attn_config(
            head_num=num_heads,
            kv_head_num=num_kv_heads,
            size_per_head=head_dim,
            tokens_per_block=token_per_block,
        )

        # Create MhaRotaryEmbeddingOp
        rope_op = MhaRotaryEmbeddingOp(
            head_size=head_dim,
            cos_sin_cache=cos_sin_cache,
            token_per_block=token_per_block,
            attn_config=attn_config,
            num_kv_heads=num_kv_heads,
        )

        # Create input tensors
        q = torch.randn(
            num_tokens, num_heads, head_dim, device=self.device, dtype=torch.float16
        )
        k = torch.randn(
            num_tokens, num_kv_heads, head_dim, device=self.device, dtype=torch.float16
        )
        v = torch.randn(
            num_tokens, num_kv_heads, head_dim, device=self.device, dtype=torch.float16
        )

        # Create position IDs (0, 1, 2, ..., num_tokens-1)
        positions = torch.arange(num_tokens, dtype=torch.int32, device=self.device)

        # Create rope_params (minimal for testing without KV cache)
        rope_params = RopeParams(
            batch_indice_d=torch.zeros(
                num_tokens, dtype=torch.int32, device=self.device
            ),
            positions_d=positions,
            page_indice_d=torch.tensor([], dtype=torch.int32, device=self.device),
            decode_page_indptr_d=torch.tensor(
                [0], dtype=torch.int32, device=self.device
            ),
            paged_kv_last_page_len_d=torch.tensor(
                [], dtype=torch.int32, device=self.device
            ),
        )

        # Store original values
        q_orig = q.clone()
        k_orig = k.clone()

        # Apply RoPE
        rope_op.forward(q, k, v, rope_params, kv_cache=None)

        # Verify Q and K are modified
        self.assertFalse(torch.allclose(q, q_orig, atol=1e-5))
        self.assertFalse(torch.allclose(k, k_orig, atol=1e-5))

        # Verify against reference implementation
        q_ref, k_ref = apply_rope_reference(
            q_orig.float(), k_orig.float(), cos_sin_cache, positions
        )

        # Compare (with some tolerance due to FP16)
        q_diff = (q.float() - q_ref).abs().max().item()
        k_diff = (k.float() - k_ref).abs().max().item()

        self.assertLess(q_diff, 1e-2, f"Q difference too large: {q_diff}")
        self.assertLess(k_diff, 1e-2, f"K difference too large: {k_diff}")

    def test_rope_with_kv_cache(self):
        """Test RoPE application with KV cache append"""
        # Setup
        num_tokens = 5
        num_heads = 8
        num_kv_heads = 4
        head_dim = 64
        max_seq_len = 2048
        token_per_block = 16

        # Create cos_sin_cache
        cos_sin_cache = create_cos_sin_cache(head_dim, max_seq_len, device="cuda")

        # Create attention config
        attn_config = create_test_attn_config(
            head_num=num_heads,
            kv_head_num=num_kv_heads,
            size_per_head=head_dim,
            tokens_per_block=token_per_block,
        )

        # Create MhaRotaryEmbeddingOp
        rope_op = MhaRotaryEmbeddingOp(
            head_size=head_dim,
            cos_sin_cache=cos_sin_cache,
            token_per_block=token_per_block,
            attn_config=attn_config,
            num_kv_heads=num_kv_heads,
        )

        # Create input tensors
        q = torch.randn(
            num_tokens, num_heads, head_dim, device=self.device, dtype=torch.float16
        )
        k = torch.randn(
            num_tokens, num_kv_heads, head_dim, device=self.device, dtype=torch.float16
        )
        v = torch.randn(
            num_tokens, num_kv_heads, head_dim, device=self.device, dtype=torch.float16
        )

        # Create paged KV cache (HND layout)
        num_pages = math.ceil(num_tokens / token_per_block)
        kv_cache_base = torch.zeros(
            num_pages,
            2,
            num_kv_heads,
            token_per_block,
            head_dim,
            device=self.device,
            dtype=torch.float16,
        )
        kv_cache = KVCache()
        kv_cache.kv_cache_base = kv_cache_base

        # Create rope_params for paged cache
        kv_len = [num_tokens]
        num_pages_per_req = torch.tensor(
            [math.ceil(len / token_per_block) for len in kv_len],
            dtype=torch.int32,
            device=self.device,
        )
        kv_append_length = torch.tensor(kv_len, dtype=torch.int32, device=self.device)
        kv_append_indptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=self.device),
                torch.cumsum(kv_append_length, dim=0),
            ]
        )
        kv_page_indptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=self.device),
                torch.cumsum(num_pages_per_req, dim=0),
            ]
        )
        kv_page_indices = torch.arange(
            int(sum(num_pages_per_req)), dtype=torch.int32, device=self.device
        )
        kv_last_page_len = torch.tensor(
            [
                (
                    len % token_per_block
                    if len % token_per_block != 0
                    else token_per_block
                )
                for len in kv_len
            ],
            dtype=torch.int32,
            device=self.device,
        )
        batch_indices, _positions_rope = get_batch_indices_positions(
            kv_append_indptr,
            get_seq_lens(kv_page_indptr, kv_last_page_len, token_per_block),
            num_tokens,
        )

        rope_params = RopeParams(
            batch_indice_d=batch_indices,
            positions_d=_positions_rope,
            page_indice_d=kv_page_indices,
            decode_page_indptr_d=kv_page_indptr,
            paged_kv_last_page_len_d=kv_last_page_len,
        )

        # Apply RoPE and append to cache
        rope_op.forward(q, k, v, rope_params, kv_cache=kv_cache)

        # Verify KV cache is not all zeros
        self.assertFalse(torch.all(kv_cache_base == 0))

        # Verify cache has the correct values for first few tokens
        for i in range(min(3, num_tokens)):
            page_idx = i // token_per_block
            pos_in_page = i % token_per_block

            # Extract K from cache
            k_cached = kv_cache_base[
                page_idx, 0, :, pos_in_page, :
            ]  # [num_kv_heads, head_dim]
            k_input = k[i]  # [num_kv_heads, head_dim]

            # Should be similar (with RoPE applied)
            diff = (k_cached - k_input).abs().max().item()
            self.assertLess(diff, 1e-2, f"K cache mismatch at token {i}")

            # Extract V from cache
            v_cached = kv_cache_base[
                page_idx, 1, :, pos_in_page, :
            ]  # [num_kv_heads, head_dim]
            v_input = v[i]  # [num_kv_heads, head_dim]

            # V should match exactly (no RoPE applied)
            diff_v = (v_cached - v_input).abs().max().item()
            self.assertLess(diff_v, 1e-4, f"V cache mismatch at token {i}")

    def test_cos_sin_cache_validation(self):
        """Test that cos_sin_cache is properly validated

        Note: get_rope_cache_once uses singleton pattern, so it returns the same cache
        for all calls. We use head_dim=128 to match other tests.
        """
        head_dim = 128  # Must match test_auto_generate_rope_cache due to singleton

        # Create attention config with explicit parameters
        attn_config = create_test_attn_config(
            head_num=8,
            kv_head_num=4,
            size_per_head=head_dim,
            tokens_per_block=16,
        )

        # Ensure rope_config.dim matches head_dim
        self.assertEqual(attn_config.rope_config.dim, head_dim)

        # Test None cos_sin_cache with rope_config auto-generates cache
        rope_op = MhaRotaryEmbeddingOp(
            head_size=head_dim,
            cos_sin_cache=None,
            token_per_block=16,
            attn_config=attn_config,
            num_kv_heads=4,
            max_position_embeddings=2048,
        )

        # Verify cache was auto-generated
        self.assertIsNotNone(rope_op.cos_sin_cache)
        self.assertEqual(
            rope_op.cos_sin_cache.shape[1],
            head_dim,
            f"Expected cache dim {head_dim}, got {rope_op.cos_sin_cache.shape[1]}",
        )  # head_dim

    def test_different_head_dimensions(self):
        """Test with different head dimensions (FlashInfer supports 64, 128, 256)"""
        for head_dim in [64, 128]:
            with self.subTest(head_dim=head_dim):
                num_tokens = 3
                num_heads = 4
                num_kv_heads = 2
                max_seq_len = 1024
                token_per_block = 16

                # Create cos_sin_cache
                cos_sin_cache = create_cos_sin_cache(
                    head_dim, max_seq_len, device="cuda"
                )

                # Create attention config
                attn_config = create_test_attn_config(
                    head_num=num_heads,
                    kv_head_num=num_kv_heads,
                    size_per_head=head_dim,
                    tokens_per_block=token_per_block,
                )

                # Create MhaRotaryEmbeddingOp
                rope_op = MhaRotaryEmbeddingOp(
                    head_size=head_dim,
                    cos_sin_cache=cos_sin_cache,
                    token_per_block=token_per_block,
                    attn_config=attn_config,
                    num_kv_heads=num_kv_heads,
                )

                # Create input tensors
                q = torch.randn(
                    num_tokens,
                    num_heads,
                    head_dim,
                    device=self.device,
                    dtype=torch.float16,
                )
                k = torch.randn(
                    num_tokens,
                    num_kv_heads,
                    head_dim,
                    device=self.device,
                    dtype=torch.float16,
                )
                v = torch.randn(
                    num_tokens,
                    num_kv_heads,
                    head_dim,
                    device=self.device,
                    dtype=torch.float16,
                )

                positions = torch.arange(
                    num_tokens, dtype=torch.int32, device=self.device
                )

                rope_params = RopeParams(
                    batch_indice_d=torch.zeros(
                        num_tokens, dtype=torch.int32, device=self.device
                    ),
                    positions_d=positions,
                    page_indice_d=torch.tensor(
                        [], dtype=torch.int32, device=self.device
                    ),
                    decode_page_indptr_d=torch.tensor(
                        [0], dtype=torch.int32, device=self.device
                    ),
                    paged_kv_last_page_len_d=torch.tensor(
                        [], dtype=torch.int32, device=self.device
                    ),
                )

                # Should not raise
                rope_op.forward(q, k, v, rope_params, kv_cache=None)

    def test_auto_generate_rope_cache(self):
        """Test automatic RoPE cache generation via get_rope_cache_once."""
        num_tokens = 10
        num_kv_heads = 4
        head_dim = 128
        token_per_block = 16
        max_position_embeddings = 2048

        # Create attention config with RoPE
        attn_config = create_test_attn_config(
            head_num=num_kv_heads,
            kv_head_num=num_kv_heads,
            size_per_head=head_dim,
            tokens_per_block=token_per_block,
        )
        attn_config.max_seq_len = max_position_embeddings

        # Create input tensors
        q = torch.randn(
            num_tokens, num_kv_heads, head_dim, dtype=torch.float16, device=self.device
        )
        k = torch.randn(
            num_tokens, num_kv_heads, head_dim, dtype=torch.float16, device=self.device
        )
        v = torch.randn(
            num_tokens, num_kv_heads, head_dim, dtype=torch.float16, device=self.device
        )

        # Create RoPE parameters
        positions = torch.arange(num_tokens, dtype=torch.int32, device=self.device)
        rope_params = RopeParams(
            batch_indice_d=torch.zeros(
                num_tokens, dtype=torch.int32, device=self.device
            ),
            positions_d=positions,
            page_indice_d=torch.tensor([], dtype=torch.int32, device=self.device),
            decode_page_indptr_d=torch.tensor(
                [0], dtype=torch.int32, device=self.device
            ),
            paged_kv_last_page_len_d=torch.tensor(
                [], dtype=torch.int32, device=self.device
            ),
        )

        # Test 1: Without cos_sin_cache (should auto-generate via get_rope_cache_once)
        rope_op_auto = MhaRotaryEmbeddingOp(
            head_size=head_dim,
            cos_sin_cache=None,  # Let it auto-generate
            token_per_block=token_per_block,
            attn_config=attn_config,
            num_kv_heads=num_kv_heads,
            max_position_embeddings=attn_config.max_seq_len,
        )

        # Store originals for comparison
        q_auto = q.clone()
        k_auto = k.clone()
        v_auto = v.clone()

        # Apply RoPE with auto-generated cache (no KV cache for simplicity)
        rope_op_auto.forward(q_auto, k_auto, v_auto, rope_params, kv_cache=None)

        # Test 2: With manually created cos_sin_cache (reference)
        cos_sin_cache = create_cos_sin_cache(
            head_dim, max_position_embeddings, attn_config.rope_config.base
        )
        rope_op_manual = MhaRotaryEmbeddingOp(
            head_size=head_dim,
            cos_sin_cache=cos_sin_cache,
            token_per_block=token_per_block,
            attn_config=attn_config,
            num_kv_heads=num_kv_heads,
            max_position_embeddings=attn_config.max_seq_len,
        )

        # Store originals for comparison
        q_manual = q.clone()
        k_manual = k.clone()
        v_manual = v.clone()

        # Apply RoPE with manual cache
        rope_op_manual.forward(q_manual, k_manual, v_manual, rope_params, kv_cache=None)

        # Compare results - they should be identical
        print(f"\nAuto-generated cache result shape: {q_auto.shape}")
        print(f"Manual cache result shape: {q_manual.shape}")
        print(f"Max difference in Q: {torch.max(torch.abs(q_auto - q_manual)).item()}")
        print(f"Max difference in K: {torch.max(torch.abs(k_auto - k_manual)).item()}")

        # Verify results are close (allowing for small floating point differences)
        self.assertTrue(
            torch.allclose(q_auto, q_manual, rtol=1e-3, atol=1e-3),
            "Auto-generated cache results should match manual cache results for Q",
        )
        self.assertTrue(
            torch.allclose(k_auto, k_manual, rtol=1e-3, atol=1e-3),
            "Auto-generated cache results should match manual cache results for K",
        )

        print("✓ Auto-generated RoPE cache produces correct results")


if __name__ == "__main__":
    unittest.main()
