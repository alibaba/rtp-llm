"""SP0 and gamma-3 verification must agree on shared small-batch rows."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.compressor import _linear_bf16_bf16_fp32
from rtp_llm.models_py.modules.dsv4.moe.gate import Gate
from rtp_llm.models_py.utils.arch import is_sm120
from rtp_llm.utils.model_weight import W


class SmallBatchProjectionsTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or not is_sm120():
            self.skipTest("SM120 required")
        torch.manual_seed(20260930)
        self.x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)

    def evaluate(self, fn, x, graph):
        if not graph:
            return fn(x)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn(x)
        torch.cuda.current_stream().wait_stream(stream)
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            out = fn(x)
        # Change the inputs after capture to verify replay, not capture output.
        original = x.clone()
        x.neg_()
        capture.replay()
        torch.testing.assert_close(out, fn(x), rtol=0, atol=0)
        x.copy_(original)
        capture.replay()
        return out.clone()

    def test_gate_shared_rows(self):
        for hashed in (False, True):
            weights = {
                W.v4_router_w: torch.randn(256, 4096, device="cuda").to(torch.bfloat16)
                * 0.03,
                W.v4_router_bias: torch.randn(256, device="cuda") * 0.01,
                W.v4_router_tid2eid: torch.stack(
                    [torch.randperm(256, device="cuda")[:6] for _ in range(8)]
                ),
            }
            gate = Gate(
                0,
                4096,
                256,
                6,
                n_hash_layers=int(hashed),
                vocab_size=8,
                layer_weights=weights,
            )
            for fp32 in ("0", "1"):
                with patch.dict("os.environ", {"DSV4_GATE_FP32": fp32}):

                    def forward(x):
                        ids = torch.arange(x.shape[0], device=x.device)
                        values, routes = gate(x, ids)
                        return torch.cat((values, routes.float()), dim=-1)

                    for graph in (False, True):
                        ref = self.evaluate(forward, self.x, graph)
                        for rows in (1, 2, 3, 4, 6, 7, 8):
                            with self.subTest(
                                hashed=hashed, fp32=fp32, graph=graph, rows=rows
                            ):
                                actual = self.evaluate(
                                    forward, self.x[:rows].clone(), graph
                                )
                                torch.testing.assert_close(
                                    actual, ref[:rows], rtol=0, atol=0
                                )
                    values, routes = gate(self.x[:0])
                    self.assertEqual(values.shape, (0, 6))
                    self.assertEqual(routes.shape, (0, 6))

    def test_compressor_shared_rows(self):
        for width in (256, 2048):
            weight = torch.randn(width, 4096, device="cuda").to(torch.bfloat16) * 0.03

            def forward(x):
                return _linear_bf16_bf16_fp32(x, weight)

            for graph in (False, True):
                ref = self.evaluate(forward, self.x.reshape(2, 4, -1), graph).reshape(
                    8, width
                )
                for batch, query in (
                    (1, 1),
                    (2, 1),
                    (1, 3),
                    (1, 4),
                    (2, 3),
                    (1, 7),
                    (2, 4),
                ):
                    with self.subTest(
                        width=width, graph=graph, batch=batch, query=query
                    ):
                        x = self.x[: batch * query].reshape(batch, query, -1).clone()
                        actual = self.evaluate(forward, x, graph)
                        self.assertEqual(actual.shape, (batch, query, width))
                        self.assertEqual(actual.dtype, torch.float32)
                        torch.testing.assert_close(
                            actual.reshape(-1, width),
                            ref[: batch * query],
                            rtol=0,
                            atol=0,
                        )


if __name__ == "__main__":
    unittest.main()
