"""RTX PRO 5000 gate for the long-context SM120 sparse-MLA path."""

from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
    quantize_and_insert_k_cache,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear import (
    CudaFp8VllmBlockwiseLinear,
)
from rtp_llm.models_py.utils.arch import is_sm120


class Sm120SparseMlaHardwareTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available() or not is_sm120():
            self.skipTest("requires an SM120 CUDA device")
        self.device = torch.device("cuda", torch.cuda.current_device())
        torch.manual_seed(7)

    def _packed_cache(self) -> torch.Tensor:
        cache = torch.zeros((2, 64, 584), dtype=torch.uint8, device=self.device)
        source = torch.randn((1, 512), dtype=torch.bfloat16, device=self.device)
        quantize_and_insert_k_cache(
            source,
            cache,
            torch.zeros(1, dtype=torch.int64, device=self.device),
        )
        return cache

    def test_cutlass_blockwise_linear_eager_and_cuda_graph(self) -> None:
        weight = (
            torch.randn((128, 128), dtype=torch.float32, device=self.device) * 0.05
        ).to(torch.float8_e4m3fn)
        weight_scale = torch.ones((1, 1), dtype=torch.float32, device=self.device)
        linear = CudaFp8VllmBlockwiseLinear(weight, weight_scale)
        activation = torch.randn((4, 128), dtype=torch.bfloat16, device=self.device)

        eager = linear(activation)
        reference = activation.float() @ weight.float().transpose(0, 1)
        torch.testing.assert_close(
            eager.float(),
            reference,
            rtol=2e-1,
            atol=1.5e-1,
        )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = linear(activation)
        activation.copy_(torch.randn_like(activation))
        graph.replay()
        torch.cuda.synchronize(self.device)
        replay_output = graph_output.clone()
        eager_after_update = linear(activation)
        torch.testing.assert_close(replay_output, eager_after_update)

    def test_hca_8192_eager_and_cuda_graph_replay(self) -> None:
        cache = self._packed_cache()
        extra_cache = self._packed_cache()
        query = torch.randn((1, 1, 16, 512), dtype=torch.bfloat16, device=self.device)
        sink = torch.zeros(16, dtype=torch.float32, device=self.device)
        swa_indices = torch.full((1, 1, 128), -1, dtype=torch.int32, device=self.device)
        swa_indices[..., 0] = 0
        swa_length = torch.ones(1, dtype=torch.int32, device=self.device)
        # HCA emits one compressed entry per 128 source tokens.  Width 8192 is
        # therefore the static FlashInfer instance needed by a 1M context.
        extra_indices = torch.full(
            (1, 1, 8192), -1, dtype=torch.int32, device=self.device
        )
        extra_indices[..., 0] = 0
        extra_length = torch.ones(1, dtype=torch.int32, device=self.device)
        op = SparseAttnV4DecodeFp8Op(16, 512, 512**-0.5)

        def forward() -> torch.Tensor:
            return op._forward_sm120_flashinfer(
                query,
                cache,
                sink,
                swa_indices,
                swa_length,
                extra_cache,
                extra_indices,
                extra_length,
            )

        # Materialize every grow-only workspace before capture, then verify a
        # replay consumes updated input values instead of replaying eager data.
        forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = forward()

        query.copy_(torch.randn_like(query))
        graph.replay()
        torch.cuda.synchronize(self.device)
        replay_output = graph_output.clone()
        eager_output = forward()

        self.assertTrue(torch.isfinite(replay_output).all())
        self.assertGreater(replay_output.float().abs().max().item(), 0.0)
        torch.testing.assert_close(
            replay_output,
            eager_output,
            rtol=2e-2,
            atol=2e-2,
        )


if __name__ == "__main__":
    unittest.main()
