import unittest

import torch

from rtp_llm.models_py.triton_kernels.dspark_swa import (
    DSparkGemmaRMSNorm,
    dspark_gemma_rms_norm,
)


def reference(x, raw, eps=1e-6):
    value = x.float()
    return (
        value
        * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        * (1 + raw.float())
    ).bfloat16()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class DSparkGemmaNormTest(unittest.TestCase):
    def test_fp32_scale_equation_and_strided_rows(self):
        torch.manual_seed(930)
        for dim in (128, 513, 6144):
            for rows in (1, 7, 33):
                with self.subTest(dim=dim, rows=rows):
                    storage = torch.randn(
                        rows, dim * 3, device="cuda", dtype=torch.bfloat16
                    )
                    x = storage[:, dim : dim * 2]
                    # Include near -1 (small scales) and positive raw weights
                    # whose +1 BF16 rounding was the old loss of precision.
                    raw = (
                        torch.randn(dim, device="cuda", dtype=torch.bfloat16) * 0.4
                        - 0.7
                    )
                    raw[::4] = 0.37109375
                    raw[1::4] = -0.99609375
                    before = storage.clone()
                    actual = dspark_gemma_rms_norm(x, raw, 1e-6)
                    expected = reference(x, raw)
                    torch.testing.assert_close(
                        actual, expected, rtol=0.008, atol=0.0001
                    )
                    error = (
                        actual.float() - expected.float()
                    ).norm() / expected.float().norm()
                    self.assertLess(error.item(), 0.001)
                    torch.testing.assert_close(storage, before, rtol=0, atol=0)
                    old_raw = (raw + 1).bfloat16().float() - 1
                    old = reference(x, old_raw)
                    self.assertGreater((old != expected).sum().item(), 0)
                    self.assertLess(
                        error.item(),
                        (
                            (old.float() - expected.float()).norm()
                            / expected.float().norm()
                        ).item(),
                    )

    def test_zero_empty_and_validation(self):
        raw = torch.zeros(128, device="cuda", dtype=torch.bfloat16)
        module = DSparkGemmaRMSNorm(raw, 1e-6)
        self.assertEqual(module.weight.data_ptr(), raw.data_ptr())
        empty = module(torch.empty(0, 128, device="cuda", dtype=torch.bfloat16))
        self.assertEqual(empty.shape, (0, 128))
        zero = module(torch.zeros(3, 128, device="cuda", dtype=torch.bfloat16))
        torch.testing.assert_close(zero, torch.zeros_like(zero), rtol=0, atol=0)
        with self.assertRaises(TypeError):
            module(torch.zeros(3, 128, device="cuda"))
        with self.assertRaises(ValueError):
            module(torch.zeros(3, 127, device="cuda", dtype=torch.bfloat16))

    def test_graph_replay_reads_updated_input_and_raw_weight(self):
        x = torch.randn(7, 6144, device="cuda", dtype=torch.bfloat16)
        raw = torch.randn(6144, device="cuda", dtype=torch.bfloat16)
        module = DSparkGemmaRMSNorm(raw, 1e-6)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                module(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = module(x)
        address = output.data_ptr()
        for scale in (0.5, 1.5):
            x.normal_()
            raw.fill_(scale)
            graph.replay()
            torch.testing.assert_close(
                output, reference(x, raw), rtol=0.008, atol=0.0001
            )
            self.assertEqual(output.data_ptr(), address)


if __name__ == "__main__":
    unittest.main()
