"""Unit tests for fused QK-RMSNorm kernel.

Validates that the single-launch Triton kernel produces results matching
two separate RMSNorm calls (one for Q heads, one for K heads).

Run with pytest:
    python -m pytest rtp_llm/models_py/triton_kernels/common/test/test_fused_qk_rmsnorm.py -v -s
"""

import unittest

import torch

from rtp_llm.models_py.modules.base.cuda.norm import FusedQKRMSNorm


def _ref_qk_rmsnorm(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    head_num: int,
    kv_head_num: int,
    size_per_head: int,
    eps: float,
) -> torch.Tensor:
    """Reference: two separate RMSNorm calls in fp32."""
    m, n = qkv.shape
    qkv_3d = qkv.clone().reshape(m, head_num + kv_head_num * 2, size_per_head)
    q = qkv_3d[:, :head_num, :].float()
    k = qkv_3d[:, head_num : head_num + kv_head_num, :].float()

    # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    def rmsnorm(x, w):
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(var + eps) * w.float().unsqueeze(0)).to(qkv.dtype)

    qkv_3d[:, :head_num, :] = rmsnorm(q, q_weight)
    qkv_3d[:, head_num : head_num + kv_head_num, :] = rmsnorm(k, k_weight)
    return qkv_3d.reshape(m, n)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFusedQKRMSNorm(unittest.TestCase):

    def _check(self, T: int, head_num: int, kv_head_num: int, size_per_head: int = 128):
        torch.manual_seed(42 + T + head_num)
        eps = 1e-6
        total_dim = (head_num + kv_head_num * 2) * size_per_head
        qkv = torch.randn(T, total_dim, dtype=torch.bfloat16, device="cuda")
        q_weight = torch.randn(size_per_head, dtype=torch.bfloat16, device="cuda")
        k_weight = torch.randn(size_per_head, dtype=torch.bfloat16, device="cuda")

        ref = _ref_qk_rmsnorm(
            qkv.clone(), q_weight, k_weight, head_num, kv_head_num, size_per_head, eps
        )

        op = FusedQKRMSNorm(
            q_weight, k_weight, head_num, kv_head_num, size_per_head, eps
        )
        actual = op(qkv.clone())

        max_diff = (actual.float() - ref.float()).abs().max().item()
        atol = 2e-2
        self.assertTrue(
            torch.allclose(actual.float(), ref.float(), atol=atol, rtol=2e-3),
            f"T={T}, heads={head_num}+{kv_head_num}: max_diff={max_diff:.4e}",
        )
        return max_diff

    def test_qwen35_config(self):
        """Qwen3.5: head_num=16, kv_head_num=4, size_per_head=128."""
        for T in [1, 4, 16, 128, 1024]:
            with self.subTest(T=T):
                max_diff = self._check(T, head_num=16, kv_head_num=4)
                print(f"  qk_rmsnorm T={T:5d}  max_diff={max_diff:.3e}  OK")

    def test_various_configs(self):
        for T in [1, 32, 256]:
            for head_num, kv_head_num in [(32, 8), (8, 2), (4, 4)]:
                with self.subTest(T=T, heads=f"{head_num}+{kv_head_num}"):
                    self._check(T, head_num, kv_head_num)

    def test_v_unchanged(self):
        """V portion of qkv must not be modified."""
        T, head_num, kv_head_num, spd = 16, 16, 4, 128
        total_dim = (head_num + kv_head_num * 2) * spd
        qkv = torch.randn(T, total_dim, dtype=torch.bfloat16, device="cuda")
        v_start = (head_num + kv_head_num) * spd
        v_original = qkv[:, v_start:].clone()

        q_weight = torch.randn(spd, dtype=torch.bfloat16, device="cuda")
        k_weight = torch.randn(spd, dtype=torch.bfloat16, device="cuda")
        op = FusedQKRMSNorm(q_weight, k_weight, head_num, kv_head_num, spd, 1e-6)
        out = op(qkv)

        v_after = out[:, v_start:]
        self.assertTrue(
            torch.equal(v_original, v_after),
            "V portion was modified by FusedQKRMSNorm",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
