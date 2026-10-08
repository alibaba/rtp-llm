"""Numerical and CUDA Graph checks for the K3 router projection."""

import unittest
import torch

from rtp_llm.models_py.modules.kimi_k3.router import KimiK3RouterProjection


def check_fp32_checkpoint_values(device):
    weight = torch.tensor([[1.0001, -0.0001], [0., 0.]], device=device)
    projection = KimiK3RouterProjection(weight)
    hidden = torch.tensor([[1., 0.]], dtype=torch.bfloat16, device=device)
    result = projection(hidden)
    assert torch.equal(result, weight[:1])
    assert not torch.equal(result, weight[:1].bfloat16().float())


def check_dyadic_oracle_shape_invariance_and_graph_replay(rows):
    generator = torch.Generator().manual_seed(9401)
    # Products and their complete sums fit exactly in FP32. This supplies an
    # independent zero-error oracle without an arbitrary tolerance.
    full = torch.randint(-4, 5, (700, 7168), generator=generator).float().div(16).bfloat16().cuda()
    weight = torch.randint(-4, 5, (896, 7168), generator=generator).float().div(16).bfloat16().cuda().t()
    assert not weight.is_contiguous()
    flags = (torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    projection = KimiK3RouterProjection(weight)
    baseline = projection(full)
    reference = (full[31:31+rows].double() @ weight.double()).float()
    # Force a non-unit last-dimension stride in the smaller input.
    storage = torch.zeros((rows, 7168*2), device='cuda', dtype=torch.bfloat16)
    hidden = storage[:, ::2]
    hidden.copy_(full[31:31+rows])
    actual = projection(hidden)
    assert torch.equal(actual, reference)
    assert torch.equal(actual, baseline[31:31+rows])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            projection(hidden)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = projection(hidden)
    for scale in [1., .5, 1.]:
        hidden.copy_(full[31:31+rows] * scale)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, reference * scale)
    assert flags == (torch.backends.cuda.matmul.allow_tf32,
                     torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)


def check_empty_projection():
    projection = KimiK3RouterProjection(torch.zeros(17, 65, device='cuda', dtype=torch.bfloat16))
    assert projection(torch.empty(0, 17, device='cuda', dtype=torch.bfloat16)).shape == (0, 65)


class KimiK3RouterProjectionTest(unittest.TestCase):
    def test_cpu_fp32_values(self):
        check_fp32_checkpoint_values('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_fp32_values(self):
        check_fp32_checkpoint_values('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_dyadic_oracle_and_graph(self):
        for rows in [1, 2, 3, 7, 8, 9, 117, 629]:
            with self.subTest(rows=rows):
                check_dyadic_oracle_shape_invariance_and_graph_replay(rows)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_nondyadic_fp32_accuracy_across_shapes(self):
        generator = torch.Generator().manual_seed(9402)
        x = torch.randn(700, 7168, generator=generator).bfloat16().cuda()
        w = torch.randn(896, 7168, generator=generator).bfloat16().cuda().t()
        projection = KimiK3RouterProjection(w)
        for rows in [1, 2, 3, 7, 8, 9, 117, 629]:
            with self.subTest(rows=rows):
                hidden = x[31:31+rows]
                actual = projection(hidden)
                self.assertEqual(actual.dtype, torch.float32)
                self.assertTrue(torch.equal(actual, projection(hidden)))
                # Native GEMM may select different reduction layouts across
                # batch shapes. Check real-valued dots against an independent
                # FP64 oracle instead of imposing cross-shape bitwise identity.
                # The dyadic test above retains exact shape/Graph assertions.
                indices = torch.arange(min(rows, 9), device='cuda')
                sample = hidden[indices].double()
                oracle = sample @ w.double()
                error = (actual[indices].double() - oracle).abs()
                # Standard dot-product forward-error bound gamma_K * |x|@|w|.
                # Native cuBLAS does not guarantee a balanced reduction tree;
                # a constant-epsilon bound is invalid even for its FP32 path.
                # Real checkpoint parity and exact dyadic tests cover the
                # tighter K3 precision contract separately.
                u = torch.finfo(torch.float32).eps / 2
                ku = hidden.shape[1] * u
                bound = (ku / (1 - ku)) * (sample.abs() @ w.double().abs())
                self.assertTrue(bool(torch.all(error <= bound)))
                print('Native router FP64 rows=%d max_abs=%g relative_l2=%g' %
                      (rows, error.max().item(),
                       (error.norm()/oracle.norm()).item()), flush=True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_empty(self):
        check_empty_projection()


if __name__ == '__main__':
    unittest.main(verbosity=2)
