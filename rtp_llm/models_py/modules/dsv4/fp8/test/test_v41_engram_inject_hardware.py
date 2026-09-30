"""SM120 Engram injection accuracy and graph gate; no hardware skips."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.v41_engram_inject import gated_engram_residual


class V41EngramInjectHardwareTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "Engram hardware gate requires CUDA")
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))

    def inputs(self, tokens, dtype):
        generator = torch.Generator().manual_seed(4114)
        values = [
            torch.randn(shape, generator=generator).to(device="cuda", dtype=dtype)
            for shape in ((tokens, 4, 5120), (tokens, 5 * 5120), (4, 5120), (4, 5120))
        ]
        return (*values, torch.arange(tokens, device="cuda") % 3 != 0)

    def assert_result(self, args, output):
        cpu = [arg.cpu() for arg in args]
        expected = gated_engram_residual(*cpu, eps=1e-6, backend="eager")
        actual = output.cpu()
        rtol, atol = {
            torch.bfloat16: (8e-3, 8e-3),
            torch.float16: (1e-3, 1e-3),
            torch.float32: (2e-5, 2e-6),
        }[actual.dtype]
        torch.testing.assert_close(
            actual, expected, rtol=rtol, atol=atol, equal_nan=True
        )
        torch.testing.assert_close(
            actual[~cpu[-1]].view(torch.uint8), cpu[0][~cpu[-1]].view(torch.uint8)
        )

    def test_checkpoint_geometry_dtypes_and_exact_hidden_alias(self):
        with torch.inference_mode():
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                for tokens in (1, 17):
                    args = self.inputs(tokens, dtype)
                    args[-1][-1] = True
                    args[1][~args[-1]] = float("nan")
                    output = gated_engram_residual(*args, eps=1e-6)
                    self.assert_result(args, output)
                    before = args[0].clone()
                    alias = gated_engram_residual(*args, eps=1e-6, out=args[0])
                    self.assertIs(alias, args[0])
                    self.assert_result((before, *args[1:]), alias)

    def test_dead_rows_preserve_signed_zero_and_nan_payloads(self):
        args = self.inputs(3, torch.bfloat16)
        args[-1].zero_()
        args[0].view(torch.int16)[0] = -32768  # negative zero
        args[0].view(torch.int16)[1] = 32705  # NaN payload
        args[0].view(torch.int16)[2] = 32641  # signaling NaN payload
        args[1].fill_(float("nan"))
        output = gated_engram_residual(*args, eps=1e-6)
        torch.testing.assert_close(output.view(torch.uint8), args[0].view(torch.uint8))

    def test_graph_replay_updates_residual_projection_and_mask(self):
        args = self.inputs(17, torch.bfloat16)
        output = torch.empty_like(args[0])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                gated_engram_residual(*args, eps=1e-6, out=output)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gated_engram_residual(*args, eps=1e-6, out=output)
        for sign, valid in ((1, True), (-1, True), (1, False)):
            args[0].fill_(sign)
            args[1].fill_(sign * 0.25)
            args[-1].fill_(valid)
            graph.replay()
            self.assert_result(args, output)


if __name__ == "__main__":
    unittest.main()
