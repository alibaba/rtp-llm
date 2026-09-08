import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.paged_kv_cache_write import (
    write_asymmetric_paged_kv_cache,
)


def _reference_write(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    page_indices: torch.Tensor,
    page_indptr: torch.Tensor,
) -> None:
    page_size = k_cache.shape[2]
    for token_idx in range(key.shape[0]):
        batch_idx = int(batch_indices[token_idx])
        position = int(positions[token_idx])
        logical_page = position // page_size
        slot = position % page_size
        page_start = int(page_indptr[batch_idx])
        page_end = int(page_indptr[batch_idx + 1])
        if 0 <= logical_page < page_end - page_start:
            physical_page = int(page_indices[page_start + logical_page])
        else:
            physical_page = 0
        if not 0 <= physical_page < k_cache.shape[0]:
            physical_page = 0
        k_cache[physical_page, :, slot, :] = key[token_idx]
        v_cache[physical_page, :, slot, :] = value[token_idx]


class TestPagedKVCacheWrite(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        self.device = torch.device("cuda:0")

    def _make_strided_mimo_tensors(
        self,
        num_tokens: int,
        num_pages: int,
        num_heads: int = 2,
        page_size: int = 8,
        k_dim: int = 192,
        v_dim: int = 128,
        dtype: torch.dtype = torch.bfloat16,
    ):
        torch.manual_seed(20260901 + num_tokens)
        packed = torch.randn(
            num_tokens,
            64 + num_heads * (k_dim + v_dim),
            dtype=torch.bfloat16,
            device=self.device,
        ).to(dtype)
        key = packed[:, 64 : 64 + num_heads * k_dim].view(num_tokens, num_heads, k_dim)
        value = packed[:, 64 + num_heads * k_dim :].view(num_tokens, num_heads, v_dim)

        cache_pool = torch.full(
            (num_pages, num_heads, page_size, k_dim + v_dim),
            -7.0,
            dtype=torch.bfloat16,
            device=self.device,
        ).to(dtype)
        k_cache = cache_pool[..., :k_dim]
        v_cache = cache_pool[..., k_dim:]
        self.assertFalse(key.is_contiguous())
        self.assertFalse(value.is_contiguous())
        self.assertFalse(k_cache.is_contiguous())
        self.assertFalse(v_cache.is_contiguous())
        self.assertEqual(k_cache.stride(), v_cache.stride())
        return key, value, k_cache, v_cache

    def test_mimo_prefill_strided_views(self) -> None:
        key, value, k_cache, v_cache = self._make_strided_mimo_tensors(10, 8)
        batch_indices = torch.zeros(10, dtype=torch.int32, device=self.device)
        positions = torch.arange(6, 16, dtype=torch.int32, device=self.device)
        page_indices = torch.tensor([2, 4], dtype=torch.int32, device=self.device)
        page_indptr = torch.tensor([0, 2], dtype=torch.int32, device=self.device)

        k_ref = k_cache.clone()
        v_ref = v_cache.clone()
        _reference_write(
            key,
            value,
            k_ref,
            v_ref,
            batch_indices,
            positions,
            page_indices,
            page_indptr,
        )
        write_asymmetric_paged_kv_cache(
            key,
            value,
            k_cache,
            v_cache,
            batch_indices,
            positions,
            page_indices,
            page_indptr,
        )

        self.assertTrue(torch.equal(k_cache, k_ref))
        self.assertTrue(torch.equal(v_cache, v_ref))

    def test_ragged_multi_request(self) -> None:
        key, value, k_cache, v_cache = self._make_strided_mimo_tensors(5, 8)
        batch_indices = torch.tensor(
            [0, 0, 0, 1, 1], dtype=torch.int32, device=self.device
        )
        positions = torch.tensor([6, 7, 8, 1, 2], dtype=torch.int32, device=self.device)
        page_indices = torch.tensor([1, 3, 5], dtype=torch.int32, device=self.device)
        page_indptr = torch.tensor([0, 2, 3], dtype=torch.int32, device=self.device)

        k_ref = k_cache.clone()
        v_ref = v_cache.clone()
        _reference_write(
            key,
            value,
            k_ref,
            v_ref,
            batch_indices,
            positions,
            page_indices,
            page_indptr,
        )
        write_asymmetric_paged_kv_cache(
            key,
            value,
            k_cache,
            v_cache,
            batch_indices,
            positions,
            page_indices,
            page_indptr,
        )

        self.assertTrue(torch.equal(k_cache, k_ref))
        self.assertTrue(torch.equal(v_cache, v_ref))

    def test_fp8_cache_dtype(self) -> None:
        key, value, k_cache, v_cache = self._make_strided_mimo_tensors(
            6, 5, dtype=torch.float8_e4m3fn
        )
        batch_indices = torch.zeros(6, dtype=torch.int32, device=self.device)
        positions = torch.arange(2, 8, dtype=torch.int32, device=self.device)
        page_indices = torch.tensor([1], dtype=torch.int32, device=self.device)
        page_indptr = torch.tensor([0, 1], dtype=torch.int32, device=self.device)

        k_ref = k_cache.clone()
        v_ref = v_cache.clone()
        _reference_write(
            key,
            value,
            k_ref,
            v_ref,
            batch_indices,
            positions,
            page_indices,
            page_indptr,
        )
        write_asymmetric_paged_kv_cache(
            key,
            value,
            k_cache,
            v_cache,
            batch_indices,
            positions,
            page_indices,
            page_indptr,
        )

        self.assertTrue(torch.equal(k_cache, k_ref))
        self.assertTrue(torch.equal(v_cache, v_ref))

    def test_swa_null_and_out_of_range_pages_use_reserved_page(self) -> None:
        key, value, k_cache, v_cache = self._make_strided_mimo_tensors(
            4, 6, page_size=4
        )
        batch_indices = torch.zeros(4, dtype=torch.int32, device=self.device)
        positions = torch.tensor([0, 1, 8, 12], dtype=torch.int32, device=self.device)
        page_indices = torch.tensor([-1, -1, 3], dtype=torch.int32, device=self.device)
        page_indptr = torch.tensor([0, 3], dtype=torch.int32, device=self.device)

        before_k = k_cache.clone()
        before_v = v_cache.clone()
        write_asymmetric_paged_kv_cache(
            key,
            value,
            k_cache,
            v_cache,
            batch_indices,
            positions,
            page_indices,
            page_indptr,
        )

        # Position 8 is the only valid materialized SWA page. NULL and logical
        # pages past this request's page span may modify reserved page 0 only.
        self.assertTrue(torch.equal(k_cache[3, :, 0, :], key[2]))
        self.assertTrue(torch.equal(v_cache[3, :, 0, :], value[2]))
        self.assertTrue(torch.equal(k_cache[1:3], before_k[1:3]))
        self.assertTrue(torch.equal(v_cache[1:3], before_v[1:3]))
        self.assertTrue(torch.equal(k_cache[4:], before_k[4:]))
        self.assertTrue(torch.equal(v_cache[4:], before_v[4:]))


if __name__ == "__main__":
    unittest.main()
