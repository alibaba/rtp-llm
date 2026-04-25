"""Regression tests for ROCm prefill geometry helpers.

These helpers underlie every ROCm prefill path:

    * ``unpad_kv_vectorized``       — flash_attn_varlen_func K/V unpad
    * ``split_qkv_fp8``             — FP8 prefill QKV split
    * ``split_raw_qkv``             — kv_cache=None (encoder-only) QKV split
    * ``reshape_kv_cache_vectorized`` — paged-prefill 5D layout (ASM + V1)

All math is plain torch and runs on CPU; the actual attention kernels are
covered by end-to-end model tests on real ROCm devices.
"""

import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.rocm_impl._attn_utils import (
    reshape_kv_cache_vectorized,
    split_qkv_fp8,
    split_raw_qkv,
    unpad_kv_vectorized,
)

# ---------------------------------------------------------------------------
# unpad_kv_vectorized
# ---------------------------------------------------------------------------


def _unpad_kv_loop(k_padded, v_padded, cu_seqlens_k):
    """Reference implementation: the original Python for-loop version."""
    batch_size = cu_seqlens_k.shape[0] - 1
    kv_lengths = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).cpu()
    key_chunks = []
    value_chunks = []
    for i in range(batch_size):
        seq_len_i = int(kv_lengths[i].item())
        key_chunks.append(k_padded[i, :, :seq_len_i, :].transpose(0, 1).contiguous())
        value_chunks.append(v_padded[i, :, :seq_len_i, :].transpose(0, 1).contiguous())
    return torch.cat(key_chunks, dim=0), torch.cat(value_chunks, dim=0)


def _make_padded(kv_lengths, num_kv_heads, head_dim, dtype):
    """Random padded K/V; out-of-range positions are sentinel values so any
    indexing bug becomes visible (zero-init would silently leak)."""
    batch_size = len(kv_lengths)
    max_seqlen_k = max(kv_lengths) if kv_lengths else 0
    k = torch.full(
        (batch_size, num_kv_heads, max_seqlen_k, head_dim),
        fill_value=999.0,
        dtype=dtype,
    )
    v = torch.full_like(k, fill_value=-999.0)
    for i, seq_len in enumerate(kv_lengths):
        k[i, :, :seq_len, :] = torch.randn(num_kv_heads, seq_len, head_dim, dtype=dtype)
        v[i, :, :seq_len, :] = torch.randn(num_kv_heads, seq_len, head_dim, dtype=dtype)
    cu = torch.zeros(batch_size + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(kv_lengths, dtype=torch.int32).cumsum(0)
    return k, v, cu


class TestUnpadKvVectorized(unittest.TestCase):
    def _check(self, kv_lengths, num_kv_heads=4, head_dim=8, dtype=torch.float32):
        k, v, cu = _make_padded(kv_lengths, num_kv_heads, head_dim, dtype)
        k_vec, v_vec = unpad_kv_vectorized(k, v, cu)
        k_ref, v_ref = _unpad_kv_loop(k, v, cu)
        torch.testing.assert_close(k_vec, k_ref)
        torch.testing.assert_close(v_vec, v_ref)
        self.assertEqual(k_vec.shape, (sum(kv_lengths), num_kv_heads, head_dim))
        self.assertEqual(v_vec.shape, (sum(kv_lengths), num_kv_heads, head_dim))

    def test_uniform_lengths(self):
        self._check([16, 16, 16, 16])

    def test_varied_lengths(self):
        self._check([3, 17, 1, 9, 32])

    def test_single_batch(self):
        self._check([23])

    def test_batch_with_empty_sequence(self):
        self._check([5, 7, 0, 11])

    def test_all_empty_sequences(self):
        cu = torch.zeros(4, dtype=torch.int32)
        k = torch.zeros((3, 2, 0, 4), dtype=torch.float32)
        v = torch.zeros_like(k)
        k_vec, v_vec = unpad_kv_vectorized(k, v, cu)
        self.assertEqual(k_vec.shape, (0, 2, 4))
        self.assertEqual(v_vec.shape, (0, 2, 4))

    def test_dtype_bfloat16(self):
        self._check([2, 5, 3], dtype=torch.bfloat16)

    def test_dtype_float16(self):
        self._check([4, 4, 4], dtype=torch.float16)

    def test_int64_cu_seqlens(self):
        kv_lengths = [2, 5, 3]
        k, v, _ = _make_padded(kv_lengths, 2, 4, torch.float32)
        cu = torch.tensor([0, 2, 7, 10], dtype=torch.int64)
        k_vec, v_vec = unpad_kv_vectorized(k, v, cu)
        k_ref, v_ref = _unpad_kv_loop(k, v, cu)
        torch.testing.assert_close(k_vec, k_ref)
        torch.testing.assert_close(v_vec, v_ref)


# ---------------------------------------------------------------------------
# split_qkv_fp8 — FP8 prefill path
# ---------------------------------------------------------------------------


class TestSplitQkvFp8(unittest.TestCase):
    """Verify FP8 QKV split returns the correct slices and shares storage."""

    def _check(self, token_num, head_num, head_num_kv, head_dim):
        # Simulate the C++ FP8 buffer: token-major, heads concatenated. We use
        # int8 here to dodge requiring an FP8-capable build for the test.
        total_heads = head_num + 2 * head_num_kv
        full = torch.arange(
            token_num * total_heads * head_dim, dtype=torch.float32
        ).reshape(token_num, total_heads * head_dim)

        q, k, v = split_qkv_fp8(full, head_num, head_num_kv, head_dim)

        self.assertEqual(q.shape, (token_num, head_num, head_dim))
        self.assertEqual(k.shape, (token_num, head_num_kv, head_dim))
        self.assertEqual(v.shape, (token_num, head_num_kv, head_dim))

        # Reference: reshape and slice manually.
        ref = full.reshape(token_num, total_heads, head_dim)
        torch.testing.assert_close(q, ref[:, :head_num, :])
        torch.testing.assert_close(k, ref[:, head_num : head_num + head_num_kv, :])
        torch.testing.assert_close(
            v, ref[:, head_num + head_num_kv : head_num + 2 * head_num_kv, :]
        )

        # Returned tensors must be views — modifying them mutates the source.
        # (The C++ FP8 path relies on this: no extra allocation.)
        original = full.clone()
        q.add_(1.0)
        self.assertFalse(torch.equal(full, original))

    def test_mha(self):
        # head_num == head_num_kv — multi-head attention shape.
        self._check(token_num=8, head_num=8, head_num_kv=8, head_dim=64)

    def test_gqa(self):
        # Grouped-query attention: H_q > H_kv (the common LLaMA / Qwen layout).
        self._check(token_num=16, head_num=32, head_num_kv=4, head_dim=128)

    def test_single_token(self):
        self._check(token_num=1, head_num=2, head_num_kv=2, head_dim=16)

    def test_mqa(self):
        # Multi-query attention: H_kv == 1.
        self._check(token_num=4, head_num=8, head_num_kv=1, head_dim=64)


# ---------------------------------------------------------------------------
# split_raw_qkv — kv_cache=None (encoder-only) path
# ---------------------------------------------------------------------------


class TestSplitRawQkv(unittest.TestCase):
    """Verify the raw QKV split + token slicing used by the kv_cache=None path
    (encoder-only models like BERT, where Q and K/V token counts may differ)."""

    def _check(
        self,
        token_num,
        head_num,
        head_num_kv,
        head_dim,
        token_q_num=None,
        token_kv_num=None,
    ):
        token_q_num = token_q_num if token_q_num is not None else token_num
        token_kv_num = token_kv_num if token_kv_num is not None else token_num
        q_size = head_num * head_dim
        kv_size = head_num_kv * head_dim
        full = torch.randn(token_num, q_size + 2 * kv_size, dtype=torch.float32)

        q, k, v = split_raw_qkv(
            full, head_num, head_num_kv, head_dim, token_q_num, token_kv_num
        )

        self.assertEqual(q.shape, (token_q_num, head_num, head_dim))
        self.assertEqual(k.shape, (token_kv_num, head_num_kv, head_dim))
        self.assertEqual(v.shape, (token_kv_num, head_num_kv, head_dim))

        # Reference: explicit split + view + slice.
        q_ref, k_ref, v_ref = torch.split(full, [q_size, kv_size, kv_size], dim=-1)
        torch.testing.assert_close(
            q, q_ref.view(token_num, head_num, head_dim)[:token_q_num].contiguous()
        )
        torch.testing.assert_close(
            k, k_ref.view(token_num, head_num_kv, head_dim)[:token_kv_num].contiguous()
        )
        torch.testing.assert_close(
            v, v_ref.view(token_num, head_num_kv, head_dim)[:token_kv_num].contiguous()
        )

        # Outputs must be contiguous — flash_attn_varlen_func expects packed
        # row-major K/V; non-contiguous strides would crash the kernel.
        self.assertTrue(q.is_contiguous())
        self.assertTrue(k.is_contiguous())
        self.assertTrue(v.is_contiguous())

    def test_full_token_count(self):
        self._check(token_num=10, head_num=4, head_num_kv=4, head_dim=32)

    def test_truncated_q_only(self):
        # Q has fewer active tokens than K/V — covers asymmetric BERT-style
        # batching where padded slots are filled but Q is shorter.
        self._check(
            token_num=10,
            head_num=4,
            head_num_kv=4,
            head_dim=32,
            token_q_num=7,
            token_kv_num=10,
        )

    def test_truncated_both(self):
        self._check(
            token_num=16,
            head_num=8,
            head_num_kv=2,
            head_dim=64,
            token_q_num=11,
            token_kv_num=13,
        )

    def test_gqa_split(self):
        self._check(token_num=4, head_num=32, head_num_kv=4, head_dim=128)


# ---------------------------------------------------------------------------
# reshape_kv_cache_vectorized — ASM + V1 paged layouts
# ---------------------------------------------------------------------------


class TestReshapeKvCacheVectorized(unittest.TestCase):
    """Verify the 5D layout produced for mha_batch_prefill_func on both ASM
    and V1 kernel write conventions.

    ASM kernel writes V via templated ``getVLocalIdx<BASE>`` → already
    vectorized ``[ps//vs, hd, vs]`` — needs only a ``view``.
    V1 kernel writes V via non-template ``getVLocalIdx`` → linear ``[hd, ps]``
    — needs reshape + permute to reach the same target.
    Both K layouts are vectorized in the kernel and only need a ``view``.
    """

    def _make_kv_cache(
        self, num_blocks, head_num_kv, tokens_per_block, head_dim, dtype
    ):
        elems = 2 * head_num_kv * tokens_per_block * head_dim
        return torch.arange(num_blocks * elems, dtype=dtype).reshape(num_blocks, elems)

    def test_asm_layout_shape(self):
        num_blocks, hk, ps, hd, dtype = 4, 8, 16, 128, torch.float16
        vs = 16 // torch.tensor([], dtype=dtype).element_size()  # = 8
        kv_base = self._make_kv_cache(num_blocks, hk, ps, hd, dtype)

        k, v = reshape_kv_cache_vectorized(kv_base, hk, ps, hd, v1_kv_layout=False)
        # K: [num_blocks, hk, hd//vs, ps, vs]
        self.assertEqual(k.shape, (num_blocks, hk, hd // vs, ps, vs))
        # V (ASM): [num_blocks, hk, ps//vs, hd, vs]
        self.assertEqual(v.shape, (num_blocks, hk, ps // vs, hd, vs))

    def test_v1_layout_shape(self):
        num_blocks, hk, ps, hd, dtype = 4, 8, 16, 128, torch.float16
        vs = 16 // torch.tensor([], dtype=dtype).element_size()
        kv_base = self._make_kv_cache(num_blocks, hk, ps, hd, dtype)

        k, v = reshape_kv_cache_vectorized(kv_base, hk, ps, hd, v1_kv_layout=True)
        # K layout is identical for ASM and V1 (both vectorized writes).
        self.assertEqual(k.shape, (num_blocks, hk, hd // vs, ps, vs))
        # V (V1, after permute): [num_blocks, hk, ps//vs, hd, vs]
        self.assertEqual(v.shape, (num_blocks, hk, ps // vs, hd, vs))

    def test_v1_permute_correctness(self):
        """V1 path goes through a ``permute(0,1,3,2,4) + contiguous``;
        verify the permute lands the elements in the right slots."""
        num_blocks, hk, ps, hd, dtype = 2, 2, 8, 16, torch.float16
        vs = 16 // torch.tensor([], dtype=dtype).element_size()  # = 8
        kv_base = self._make_kv_cache(num_blocks, hk, ps, hd, dtype)

        _, v_v1 = reshape_kv_cache_vectorized(kv_base, hk, ps, hd, v1_kv_layout=True)

        # Build the same value via the documented manual sequence.
        expected_elems = 2 * hk * ps * hd
        flat = kv_base[:, :expected_elems].reshape(num_blocks, 2, hk, ps * hd)
        v_linear = flat[:, 1, :, :].view(num_blocks, hk, hd, ps)
        v_ref = (
            v_linear.reshape(num_blocks, hk, hd, ps // vs, vs)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        torch.testing.assert_close(v_v1, v_ref)

    def test_asm_view_no_copy(self):
        """ASM path uses pure ``view``; verify no copy happens (the result
        shares storage with kv_base)."""
        num_blocks, hk, ps, hd, dtype = 2, 2, 8, 16, torch.float16
        kv_base = self._make_kv_cache(num_blocks, hk, ps, hd, dtype)

        k, v = reshape_kv_cache_vectorized(kv_base, hk, ps, hd, v1_kv_layout=False)
        self.assertEqual(k.data_ptr(), kv_base.data_ptr())
        # V (ASM) starts after the K half within each block stride.
        self.assertGreater(v.data_ptr(), kv_base.data_ptr())

    def test_oversized_stride_truncated(self):
        """Hybrid-cache buffers carry extra stride for linear-attention layers;
        the helper must slice down to ``2*hk*ps*hd`` and ignore the tail."""
        num_blocks, hk, ps, hd, dtype = 2, 4, 8, 32, torch.float16
        expected_elems = 2 * hk * ps * hd
        # Allocate with extra padding columns beyond expected_elems.
        kv_base = torch.arange(num_blocks * (expected_elems + 64), dtype=dtype).reshape(
            num_blocks, expected_elems + 64
        )
        k, v = reshape_kv_cache_vectorized(kv_base, hk, ps, hd, v1_kv_layout=False)
        # Result must reflect only the first expected_elems per block.
        vs = 16 // kv_base.element_size()
        self.assertEqual(k.shape, (num_blocks, hk, hd // vs, ps, vs))
        self.assertEqual(v.shape, (num_blocks, hk, ps // vs, hd, vs))
        # First K element should be kv_base[0, 0] — proves we sliced from
        # offset 0, not from somewhere inside the padding.
        self.assertEqual(k[0, 0, 0, 0, 0].item(), kv_base[0, 0].item())


if __name__ == "__main__":
    unittest.main()
