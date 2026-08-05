"""CUDA ``per_token_group_quant_fp4`` vs deep_gemm reference (Blackwell SM100+).

Covers:
  - Python wrapper ``per_token_group_quant_fp4`` (fp4_kernel)
  - Low-level op ``rtp_llm_ops.per_token_group_quant_fp4``
  - cast_back round-trip numerical sanity
"""

from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.models_py.kernels.cuda.fp4_kernel import (
    create_per_token_group_quant_fp4_output_scale,
    per_token_group_quant_fp4,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


def _is_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() >= (10, 0)


def _per_token_cast_to_fp4_ref(x: torch.Tensor):
    from deep_gemm.utils import per_token_cast_to_fp4

    return per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)


def _cast_back(packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    from deep_gemm.utils import cast_back_from_fp4

    return cast_back_from_fp4(packed, sf, gran_k=32, use_packed_ue8m0=True)


class PerTokenGroupQuantFp4Test(TestCase):
    GRAN_K = 32
    HD = 128
    EPS = 1e-4

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        if not _is_blackwell():
            raise SkipTest("FP4 quant kernel tests require Blackwell SM100+")
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        torch.manual_seed(0)

    def _assert_byte_equal_to_ref(
        self, q: torch.Tensor, s: torch.Tensor, x: torch.Tensor
    ) -> None:
        x_flat = x.reshape(-1, x.shape[-1])
        ref_q, ref_s = _per_token_cast_to_fp4_ref(x_flat)
        ref_q = ref_q.view(torch.int8).reshape_as(q)
        ref_s = ref_s.reshape_as(s)
        self.assertTrue(torch.equal(q, ref_q))
        self.assertTrue(torch.equal(s, ref_s))

    def _run_cuda_op(self, x: torch.Tensor):
        x_q = torch.empty(
            x.shape[:-1] + (x.shape[-1] // 2,), device=x.device, dtype=torch.int8
        )
        x_s = create_per_token_group_quant_fp4_output_scale(
            x_shape=x.shape,
            device=x.device,
            group_size=self.GRAN_K,
            use_packed_ue8m0=True,
        )
        rtp_llm_ops.per_token_group_quant_fp4(x, x_q, x_s, self.GRAN_K, self.EPS, True)
        torch.cuda.synchronize()
        return x_q, x_s

    def test_wrapper_matches_deep_gemm_single_token(self):
        x = torch.randn(1, self.HD, dtype=torch.bfloat16, device=self.device)
        q, s = per_token_group_quant_fp4(x, group_size=self.GRAN_K)
        self._assert_byte_equal_to_ref(q, s, x)

    def test_wrapper_matches_deep_gemm_batch(self):
        x = torch.randn(17, self.HD, dtype=torch.bfloat16, device=self.device)
        q, s = per_token_group_quant_fp4(x, group_size=self.GRAN_K)
        self._assert_byte_equal_to_ref(q, s, x)

    def test_wrapper_matches_deep_gemm_3d(self):
        x = torch.randn(2, 5, self.HD, dtype=torch.bfloat16, device=self.device)
        q, s = per_token_group_quant_fp4(x, group_size=self.GRAN_K)
        self._assert_byte_equal_to_ref(q, s, x)

    def test_wrapper_fp16_input(self):
        x = torch.randn(3, self.HD, dtype=torch.float16, device=self.device)
        q, s = per_token_group_quant_fp4(x, group_size=self.GRAN_K)
        self._assert_byte_equal_to_ref(q, s, x)

    def test_cuda_op_matches_deep_gemm(self):
        x = torch.randn(11, self.HD, dtype=torch.bfloat16, device=self.device)
        q, s = self._run_cuda_op(x)
        self._assert_byte_equal_to_ref(q, s, x)

    def test_wrapper_and_cuda_op_agree(self):
        x = torch.randn(8, self.HD, dtype=torch.bfloat16, device=self.device)
        q_wrap, s_wrap = per_token_group_quant_fp4(x, group_size=self.GRAN_K)
        q_op, s_op = self._run_cuda_op(x)
        self.assertTrue(torch.equal(q_wrap, q_op))
        self.assertTrue(torch.equal(s_wrap, s_op))

    def test_indexer_like_multi_head_shape(self):
        num_tokens = 6
        n_heads = 4
        x = torch.randn(
            num_tokens * n_heads,
            self.HD,
            dtype=torch.bfloat16,
            device=self.device,
        )
        q, s = per_token_group_quant_fp4(x, group_size=self.GRAN_K)
        self.assertEqual(q.shape, (num_tokens * n_heads, self.HD // 2))
        self.assertEqual(s.shape, (num_tokens * n_heads, 1))
        self._assert_byte_equal_to_ref(q, s, x)

    def test_cast_back_roundtrip(self):
        x = torch.randn(5, self.HD, dtype=torch.bfloat16, device=self.device)
        q, s = per_token_group_quant_fp4(x, group_size=self.GRAN_K)
        x_back = _cast_back(q.view(torch.uint8), s)
        max_err = (x.float() - x_back.float()).abs().max().item()
        # FP4 e2m1 is coarse; bound should stay well below bf16 noise floor.
        self.assertLess(max_err, 0.5)

    def test_empty_batch(self):
        x = torch.empty(0, self.HD, dtype=torch.bfloat16, device=self.device)
        q, s = per_token_group_quant_fp4(x, group_size=self.GRAN_K)
        self.assertEqual(q.shape, (0, self.HD // 2))
        self.assertEqual(s.shape, (0, 1))


if __name__ == "__main__":
    main()
