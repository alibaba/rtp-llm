import itertools
from unittest import SkipTest, TestCase, main

import torch
from torch import dtype as _dtype

from rtp_llm.models_py.modules import FusedQKRMSNorm, QKRMSNorm
from rtp_llm.ops.compute_ops import rtp_llm_ops


class FusedQKRMSNormTest(TestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HEAD_NUM = [40]
    KV_HEAD_NUM = [40, 8, 4]
    SIZE_PER_HEAD = [128]

    # DTYPES = [torch.bfloat16]
    # NUM_TOKENS = [4096]
    # HEAD_NUM = [40]
    # KV_HEAD_NUM =  [40]
    # SIZE_PER_HEAD = [128]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_fused_qk_rmsnorm_test(
        self,
        num_tokens: int,
        head_num: int,
        kv_head_num: int,
        size_per_head: int,
        dtype: _dtype,
    ):
        torch.manual_seed(0)

        hidden_size = head_num * size_per_head + 2 * kv_head_num * size_per_head

        q_weight = torch.randn(size_per_head, dtype=dtype)
        k_weight = torch.randn(size_per_head, dtype=dtype)

        qkrmsnorm = QKRMSNorm(q_weight, k_weight, head_num, kv_head_num, size_per_head)
        fused_qkrmsnorm = FusedQKRMSNorm(
            q_weight, k_weight, head_num, kv_head_num, size_per_head
        )

        x = torch.randn(num_tokens, hidden_size, dtype=dtype)

        # for _ in range(5):
        #     # out = qkrmsnorm(x)
        #     out = fused_qkrmsnorm(x)
        # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     for _ in range(10):
        #         # out = qkrmsnorm(x)
        #         out = fused_qkrmsnorm(x)
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

        self.assertTrue(
            torch.allclose(qkrmsnorm(x), fused_qkrmsnorm(x), atol=1e-2, rtol=1e-2)
        )

    def test_fusedqkrmsnorm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HEAD_NUM,
            self.KV_HEAD_NUM,
            self.SIZE_PER_HEAD,
            self.DTYPES,
        ):
            with self.subTest(
                num_tokens=params[0],
                head_num=params[1],
                kv_head_num=params[2],
                size_per_head=params[3],
                dtype=params[4],
            ):
                self._run_fused_qk_rmsnorm_test(*params)


class FusedQKRMSNormV2Test(TestCase):
    """Precision regression for the opt-in V2 kernel (fused_qk_rmsnorm_v2.cu).

    The V2 kernel only specializes on head_dim=256 (Qwen3.5-9B Full-Attn);
    other dims fall through to V1 inside invokeFusedQkRmsNormV2 (so the
    head_dim=128 cases below should be bit-identical to V1, while head_dim=256
    exercises the warp-per-(token,head) wave64 path).

    We bypass the FusedQKRMSNorm Python wrapper (which reads the env var at
    module-import time and locks one path per process) and call both ops
    directly, so a single test run covers V1 vs V2 head-to-head.
    """

    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [1, 7, 83, 4096]
    # Qwen3.5-9B Full-Attn rank shape is (head_num=8, kv=2, head_dim=256);
    # also include a wider Q/K split to catch index-arithmetic bugs.
    HEAD_CONFIGS = [(8, 2), (8, 8), (40, 8)]
    HEAD_DIMS = [128, 256]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA/ROCm is not available")
        torch.set_default_device("cuda")

    @staticmethod
    def _call_v1(io: torch.Tensor, qg: torch.Tensor, kg: torch.Tensor,
                 head_num: int, kv_head_num: int, head_dim: int, eps: float):
        m, n = io.shape
        rtp_llm_ops.fused_qk_rmsnorm(io, qg, kg, eps, head_num, kv_head_num,
                                     m, n, head_dim)

    @staticmethod
    def _call_v2(io: torch.Tensor, qg: torch.Tensor, kg: torch.Tensor,
                 head_num: int, kv_head_num: int, head_dim: int, eps: float):
        m, n = io.shape
        rtp_llm_ops.fused_qk_rmsnorm_v2(io, qg, kg, eps, head_num, kv_head_num,
                                        m, n, head_dim)

    def _run_v1_v2_compare(
        self,
        num_tokens: int,
        head_num: int,
        kv_head_num: int,
        head_dim: int,
        dtype: _dtype,
    ):
        torch.manual_seed(0)
        eps = 1e-6
        hidden = (head_num + 2 * kv_head_num) * head_dim
        q_gamma = torch.randn(head_dim, dtype=dtype)
        k_gamma = torch.randn(head_dim, dtype=dtype)
        x = torch.randn(num_tokens, hidden, dtype=dtype)

        # V1 (in-place); take a copy so V2 starts from the same input.
        io_v1 = x.clone()
        io_v2 = x.clone()
        self._call_v1(io_v1, q_gamma, k_gamma, head_num, kv_head_num, head_dim, eps)
        self._call_v2(io_v2, q_gamma, k_gamma, head_num, kv_head_num, head_dim, eps)

        # Q+K region must match the normalized output; the V tail is untouched
        # by both kernels — assert it stayed bit-identical to the input.
        qk_size = (head_num + kv_head_num) * head_dim
        torch.testing.assert_close(
            io_v2[:, :qk_size], io_v1[:, :qk_size], atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(io_v2[:, qk_size:], x[:, qk_size:],
                                   atol=0, rtol=0)

    def test_v1_vs_v2(self):
        for num_tokens, (head_num, kv_head_num), head_dim, dtype in itertools.product(
            self.NUM_TOKENS, self.HEAD_CONFIGS, self.HEAD_DIMS, self.DTYPES
        ):
            with self.subTest(
                num_tokens=num_tokens,
                head_num=head_num,
                kv_head_num=kv_head_num,
                head_dim=head_dim,
                dtype=dtype,
            ):
                self._run_v1_v2_compare(
                    num_tokens, head_num, kv_head_num, head_dim, dtype
                )

    def test_v2_fallback_path_is_bit_identical(self):
        """head_dim != 256 falls back to V1 inside invokeFusedQkRmsNormV2 —
        the two outputs must be bit-identical (same kernel, no rounding diff)."""
        torch.manual_seed(1)
        eps = 1e-6
        head_num, kv_head_num, head_dim = 8, 2, 128
        num_tokens = 83
        hidden = (head_num + 2 * kv_head_num) * head_dim
        for dtype in self.DTYPES:
            with self.subTest(dtype=dtype):
                q_gamma = torch.randn(head_dim, dtype=dtype)
                k_gamma = torch.randn(head_dim, dtype=dtype)
                x = torch.randn(num_tokens, hidden, dtype=dtype)
                io_v1 = x.clone()
                io_v2 = x.clone()
                self._call_v1(io_v1, q_gamma, k_gamma,
                              head_num, kv_head_num, head_dim, eps)
                self._call_v2(io_v2, q_gamma, k_gamma,
                              head_num, kv_head_num, head_dim, eps)
                # Fallback dispatches the V1 kernel directly — bit-exact.
                torch.testing.assert_close(io_v2, io_v1, atol=0, rtol=0)


if __name__ == "__main__":
    main()
