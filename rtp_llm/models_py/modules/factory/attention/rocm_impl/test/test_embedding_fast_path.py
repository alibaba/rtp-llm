"""Regression tests for _try_embedding_fast_path."""
import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter import (
    _try_embedding_fast_path,
)

HAS_GPU = torch.cuda.is_available()


class FakeFmhaImpl:
    def __init__(self, head_num=14, head_num_kv=2, head_dim=64, is_causal=False):
        self.head_num = head_num
        self.head_num_kv = head_num_kv
        self.head_dim = head_dim
        self.is_causal = is_causal


class FakeFmhaParams:
    def __init__(self, seqlens, device="cpu"):
        cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=device)
        cu[1:] = torch.tensor(seqlens, dtype=torch.int32, device=device).cumsum(0)
        self.cu_seqlens_q = cu
        self.cu_seqlens_k = cu.clone()
        self.max_seqlen_q = max(seqlens)
        self.max_seqlen_k = max(seqlens)
        total = sum(seqlens)
        self.token_q_num = total
        self.token_kv_num = total


class TestEmbeddingFastPathSkip(unittest.TestCase):
    """Cases where fast path must NOT trigger."""

    def _hidden(self, impl):
        return (impl.head_num + 2 * impl.head_num_kv) * impl.head_dim

    def test_fp8_input_skipped(self):
        impl = FakeFmhaImpl()
        params = FakeFmhaParams([4])
        qkv = torch.randn(4, self._hidden(impl)).to(torch.float8_e4m3fnuz)
        self.assertIsNone(_try_embedding_fast_path(qkv, impl, params))

    def test_3d_input_skipped(self):
        impl = FakeFmhaImpl()
        params = FakeFmhaParams([4])
        self.assertIsNone(_try_embedding_fast_path(torch.randn(1, 4, 100), impl, params))

    def test_tuple_with_valid_kv_skipped(self):
        impl = FakeFmhaImpl()
        params = FakeFmhaParams([4])
        q = torch.randn(4, impl.head_num, impl.head_dim)
        k = torch.randn(4, impl.head_num_kv, impl.head_dim)
        v = torch.randn(4, impl.head_num_kv, impl.head_dim)
        self.assertIsNone(_try_embedding_fast_path((q, k, v), impl, params))


@unittest.skipUnless(HAS_GPU, "requires GPU")
class TestEmbeddingFastPathTrigger(unittest.TestCase):
    """Cases where fast path must trigger and produce correct shapes."""

    def _hidden(self, impl):
        return (impl.head_num + 2 * impl.head_num_kv) * impl.head_dim

    def test_2d_packed_qkv(self):
        impl = FakeFmhaImpl()
        params = FakeFmhaParams([6], device="cuda")
        qkv = torch.randn(6, self._hidden(impl), device="cuda", dtype=torch.bfloat16)
        result = _try_embedding_fast_path(qkv, impl, params)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (6, impl.head_num * impl.head_dim))

    def test_tuple_packed_none_none(self):
        impl = FakeFmhaImpl()
        params = FakeFmhaParams([4], device="cuda")
        qkv = torch.randn(4, self._hidden(impl), device="cuda", dtype=torch.bfloat16)
        result = _try_embedding_fast_path((qkv, None, None), impl, params)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (4, impl.head_num * impl.head_dim))

    def test_tuple_packed_empty_tensors(self):
        impl = FakeFmhaImpl()
        params = FakeFmhaParams([4], device="cuda")
        qkv = torch.randn(4, self._hidden(impl), device="cuda", dtype=torch.bfloat16)
        result = _try_embedding_fast_path((qkv, torch.Tensor(), torch.Tensor()), impl, params)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (4, impl.head_num * impl.head_dim))

    def test_variable_length_trimming(self):
        """Unequal seqlens: output rows == token_q_num, not input rows."""
        impl = FakeFmhaImpl()
        seqlens = [3, 5, 2]
        params = FakeFmhaParams(seqlens, device="cuda")
        # Allocate extra rows beyond token_q_num
        qkv = torch.randn(12, self._hidden(impl), device="cuda", dtype=torch.bfloat16)
        params.token_q_num = 10
        params.token_kv_num = 10
        result = _try_embedding_fast_path(qkv, impl, params)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 10, "output should be trimmed to token_q_num")

    def test_single_token(self):
        impl = FakeFmhaImpl()
        params = FakeFmhaParams([1], device="cuda")
        qkv = torch.randn(1, self._hidden(impl), device="cuda", dtype=torch.bfloat16)
        result = _try_embedding_fast_path(qkv, impl, params)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (1, impl.head_num * impl.head_dim))


if __name__ == "__main__":
    unittest.main(verbosity=2)
