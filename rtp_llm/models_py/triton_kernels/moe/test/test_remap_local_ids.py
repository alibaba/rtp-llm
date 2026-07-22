"""
Tests for remap_to_local_ids Triton kernel.
Verifies equivalence with reference torch implementation across edge cases.
Requires ROCm or CUDA GPU; skipped when no GPU is available.
"""
import unittest

import torch

HAS_GPU = torch.cuda.is_available()

if HAS_GPU:
    from rtp_llm.models_py.triton_kernels.moe.remap_local_ids_kernel import (
        remap_to_local_ids,
    )


def remap_to_local_ids_ref(
    dispatch_ids: torch.Tensor,
    dispatch_weights: torch.Tensor,
    local_start: int,
    local_end: int,
) -> tuple:
    """Reference implementation in pure PyTorch."""
    is_local = (dispatch_ids >= local_start) & (dispatch_ids < local_end)
    local_ids = torch.where(is_local, dispatch_ids - local_start, torch.zeros_like(dispatch_ids))
    local_weights = torch.where(is_local, dispatch_weights, torch.zeros_like(dispatch_weights))
    return local_ids.to(torch.int32), local_weights.to(torch.float32)


DEVICE = torch.device("cuda") if HAS_GPU else None


@unittest.skipUnless(HAS_GPU, "No GPU available, skipping Triton kernel tests")
class TestRemapToLocalIds(unittest.TestCase):
    """Parameterized tests for remap_to_local_ids kernel."""

    def test_empty_dispatch_n0(self):
        """P1: N=0 should not launch kernel, return empty tensors."""
        ids = torch.empty((0, 4), dtype=torch.int32, device=DEVICE)
        weights = torch.empty((0, 4), dtype=torch.float32, device=DEVICE)
        local_ids, local_weights = remap_to_local_ids(ids, weights, 0, 8)
        self.assertEqual(local_ids.shape, (0, 4))
        self.assertEqual(local_weights.shape, (0, 4))
        self.assertEqual(local_ids.dtype, torch.int32)
        self.assertEqual(local_weights.dtype, torch.float32)

    def test_all_local(self):
        """All experts belong to this rank."""
        ids = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.int32, device=DEVICE)
        weights = torch.tensor([[0.5, 0.3, 0.1, 0.1], [0.4, 0.3, 0.2, 0.1]], dtype=torch.float32, device=DEVICE)
        local_ids, local_weights = remap_to_local_ids(ids, weights, 0, 4)
        ref_ids, ref_weights = remap_to_local_ids_ref(ids, weights, 0, 4)
        self.assertTrue(torch.equal(local_ids, ref_ids))
        self.assertTrue(torch.allclose(local_weights, ref_weights))

    def test_all_non_local(self):
        """No experts belong to this rank — all weights should be zeroed."""
        ids = torch.tensor([[8, 9, 10, 11]], dtype=torch.int32, device=DEVICE)
        weights = torch.tensor([[0.3, 0.3, 0.2, 0.2]], dtype=torch.float32, device=DEVICE)
        local_ids, local_weights = remap_to_local_ids(ids, weights, 0, 8)
        self.assertTrue((local_ids == 0).all())
        self.assertTrue((local_weights == 0.0).all())

    def test_mixed_local_non_local(self):
        """Mix of local and non-local experts."""
        ids = torch.tensor([[4, 5, 12, 15], [6, 7, 0, 1]], dtype=torch.int32, device=DEVICE)
        weights = torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.4, 0.3, 0.2, 0.1]], dtype=torch.float32, device=DEVICE)
        local_start, local_end = 4, 8
        local_ids, local_weights = remap_to_local_ids(ids, weights, local_start, local_end)
        ref_ids, ref_weights = remap_to_local_ids_ref(ids, weights, local_start, local_end)
        self.assertTrue(torch.equal(local_ids, ref_ids))
        self.assertTrue(torch.allclose(local_weights, ref_weights))

    def test_weight_dtype_fp16(self):
        """Kernel should handle fp16 input weights (output is always fp32)."""
        ids = torch.tensor([[2, 5, 7, 10]], dtype=torch.int32, device=DEVICE)
        weights = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float16, device=DEVICE)
        local_ids, local_weights = remap_to_local_ids(ids, weights, 4, 8)
        ref_ids, ref_weights = remap_to_local_ids_ref(ids, weights, 4, 8)
        self.assertEqual(local_weights.dtype, torch.float32)
        self.assertTrue(torch.equal(local_ids, ref_ids))
        self.assertTrue(torch.allclose(local_weights, ref_weights, atol=1e-3))

    def test_weight_dtype_bf16(self):
        """Kernel should handle bf16 input weights (output is always fp32)."""
        ids = torch.tensor([[2, 5, 7, 10]], dtype=torch.int32, device=DEVICE)
        weights = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.bfloat16, device=DEVICE)
        local_ids, local_weights = remap_to_local_ids(ids, weights, 4, 8)
        ref_ids, ref_weights = remap_to_local_ids_ref(ids, weights, 4, 8)
        self.assertEqual(local_weights.dtype, torch.float32)
        self.assertTrue(torch.equal(local_ids, ref_ids))
        self.assertTrue(torch.allclose(local_weights, ref_weights, atol=1e-3))

    def test_weight_dtype_fp32(self):
        """Kernel should handle fp32 input weights."""
        ids = torch.tensor([[2, 5, 7, 10]], dtype=torch.int32, device=DEVICE)
        weights = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32, device=DEVICE)
        local_ids, local_weights = remap_to_local_ids(ids, weights, 4, 8)
        ref_ids, ref_weights = remap_to_local_ids_ref(ids, weights, 4, 8)
        self.assertEqual(local_weights.dtype, torch.float32)
        self.assertTrue(torch.equal(local_ids, ref_ids))
        self.assertTrue(torch.allclose(local_weights, ref_weights))

    def test_topk_1(self):
        """topk=1."""
        M, topk, num_experts = 16, 1, 64
        local_start, local_end = 8, 16
        ids = torch.randint(0, num_experts, (M, topk), dtype=torch.int32, device=DEVICE)
        weights = torch.randn(M, topk, dtype=torch.float32, device=DEVICE).abs()
        local_ids, local_weights = remap_to_local_ids(ids, weights, local_start, local_end)
        ref_ids, ref_weights = remap_to_local_ids_ref(ids, weights, local_start, local_end)
        self.assertTrue(torch.equal(local_ids, ref_ids))
        self.assertTrue(torch.allclose(local_weights, ref_weights))

    def test_topk_8(self):
        """topk=8."""
        M, topk, num_experts = 16, 8, 64
        local_start, local_end = 8, 16
        ids = torch.randint(0, num_experts, (M, topk), dtype=torch.int32, device=DEVICE)
        weights = torch.randn(M, topk, dtype=torch.float32, device=DEVICE).abs()
        local_ids, local_weights = remap_to_local_ids(ids, weights, local_start, local_end)
        ref_ids, ref_weights = remap_to_local_ids_ref(ids, weights, local_start, local_end)
        self.assertTrue(torch.equal(local_ids, ref_ids))
        self.assertTrue(torch.allclose(local_weights, ref_weights))

    def test_large_batch(self):
        """Stress test with large token count (exceeds single Triton block)."""
        M, topk, num_experts = 4096, 8, 128
        local_start, local_end = 64, 80
        ids = torch.randint(0, num_experts, (M, topk), dtype=torch.int32, device=DEVICE)
        weights = torch.randn(M, topk, dtype=torch.float32, device=DEVICE).abs()
        local_ids, local_weights = remap_to_local_ids(ids, weights, local_start, local_end)
        ref_ids, ref_weights = remap_to_local_ids_ref(ids, weights, local_start, local_end)
        self.assertTrue(torch.equal(local_ids, ref_ids))
        self.assertTrue(torch.allclose(local_weights, ref_weights))

    def test_single_element(self):
        """Edge case: single token, single expert."""
        ids = torch.tensor([[5]], dtype=torch.int32, device=DEVICE)
        weights = torch.tensor([[1.0]], dtype=torch.float32, device=DEVICE)
        local_ids, local_weights = remap_to_local_ids(ids, weights, 4, 8)
        self.assertEqual(local_ids.item(), 1)  # 5 - 4 = 1
        self.assertEqual(local_weights.item(), 1.0)

    def test_boundary_expert_ids(self):
        """Expert IDs exactly at local_start and local_end boundaries."""
        ids = torch.tensor([[3, 4, 7, 8]], dtype=torch.int32, device=DEVICE)
        weights = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32, device=DEVICE)
        local_start, local_end = 4, 8
        local_ids, local_weights = remap_to_local_ids(ids, weights, local_start, local_end)
        expected_ids = torch.tensor([[0, 0, 3, 0]], dtype=torch.int32, device=DEVICE)
        expected_weights = torch.tensor([[0.0, 0.25, 0.25, 0.0]], dtype=torch.float32, device=DEVICE)
        self.assertTrue(torch.equal(local_ids, expected_ids))
        self.assertTrue(torch.allclose(local_weights, expected_weights))


if __name__ == "__main__":
    unittest.main()
