"""Fixed-reduction gates: positions, layouts, replay, and explicit phase scope."""

import os
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.models_py.triton_kernels.common.bf16_gate_linear import (
    maybe_bf16_gate_linear,
)
from rtp_llm.models_py.triton_kernels.common.scalar_linear import (
    maybe_bf16_scalar_linear,
)
from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.env import (
    fusion_phase,
    is_decode_phase,
)


class GateLinearTest(unittest.TestCase):
    @torch.inference_mode()
    def test_positions_strides_bias_and_graph_replay(self):
        torch.manual_seed(2510)
        for n in (128, 512):
            for strided in (False, True):
                stride = 2 if strided else 1
                w = torch.randn(n, 4096, device="cuda", dtype=torch.bfloat16) / 8
                if strided:
                    w = w.T.contiguous().T
                samples = torch.randn(
                    3, 4096 * stride, device="cuda", dtype=torch.bfloat16
                )[:, ::stride]
                for bias in (None, torch.randn(n, device="cuda", dtype=torch.bfloat16)):
                    reference = torch.cat(
                        [maybe_bf16_gate_linear(row[None], w, bias) for row in samples]
                    )
                    gold = F.linear(
                        samples.double(),
                        w.double(),
                        None if bias is None else bias.double(),
                    )
                    torch.testing.assert_close(
                        reference.double(), gold, rtol=0.004, atol=0.002
                    )
                    for m in (
                        1,
                        3,
                        17,
                        31,
                        64,
                        65,
                        97,
                        128,
                        251,
                        256,
                        257,
                        509,
                        512,
                        769,
                        1024,
                    ):
                        x = torch.randn(
                            m, 4096 * stride, device="cuda", dtype=torch.bfloat16
                        )[:, ::stride]
                        positions = sorted({0, min(m - 1, 63), min(m - 1, 64), m - 1})
                        maybe_bf16_gate_linear(x, w, bias)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            y = maybe_bf16_gate_linear(x, w, bias)
                        for sample in range(3):
                            for pos in positions:
                                x[pos].copy_(samples[sample])
                            y.fill_(float("nan"))
                            graph.replay()
                            for pos in positions:
                                self.assertTrue(
                                    torch.equal(y[pos], reference[sample]),
                                    (n, strided, m, sample, pos),
                                )

    @torch.inference_mode()
    def test_decode_scope_and_option_snapshot(self):
        flags = {"RTP_BF16_GATE_KERNEL": "1", "RTP_QWEN35_DECODE_FUSION": "0"}
        self.assertFalse(is_decode_phase())
        with patch.dict(os.environ, flags):
            for n in (1, 128, 512):
                w = torch.randn(n, 4096, device="cuda", dtype=torch.bfloat16)
                op = CudaF16Linear(w.T)
                x = torch.randn(513, 4096, device="cuda", dtype=torch.bfloat16)
                native = F.linear(x, w)
                with patch.dict(os.environ, {"RTP_BF16_GATE_KERNEL": "0"}):
                    self.assertTrue(torch.equal(op(x), native))
                    with fusion_phase(is_prefill=True):
                        self.assertFalse(is_decode_phase())
                        self.assertTrue(torch.equal(op(x), native))
                    with fusion_phase(is_prefill=False):
                        self.assertTrue(is_decode_phase())
                        expected = (
                            maybe_bf16_scalar_linear(x, w)
                            if n == 1
                            else maybe_bf16_gate_linear(x, w)
                        )
                        self.assertTrue(torch.equal(op(x), expected))
                        with fusion_phase(is_prefill=True):
                            self.assertFalse(is_decode_phase())
                            self.assertTrue(torch.equal(op(x), native))
                        self.assertTrue(is_decode_phase())
                    self.assertFalse(is_decode_phase())
                    disabled = CudaF16Linear(w.T)
                    with fusion_phase(is_prefill=False):
                        self.assertTrue(torch.equal(disabled(x), native))

    @torch.inference_mode()
    def test_disabled_or_unset_ignores_removed_options(self):
        legacy = {
            "RTP_BF16_SCALAR_LINEAR": "1",
            "RTP_BF16_ROUTER_LINEAR": "1",
            "RTP_BF16_GDN_LINEAR": "1",
            "RTP_BF16_SCALAR_LINEAR_MAX_ROWS": "not-a-number",
        }
        module = "rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear"
        x = torch.randn(96, 4096, device="cuda", dtype=torch.bfloat16)
        for value in (None, "0"):
            with patch.dict(os.environ, legacy):
                if value is None:
                    os.environ.pop("RTP_BF16_GATE_KERNEL", None)
                else:
                    os.environ["RTP_BF16_GATE_KERNEL"] = value
                for n in (1, 128, 512):
                    w = torch.randn(n, 4096, device="cuda", dtype=torch.bfloat16)
                    op = CudaF16Linear(w.T)
                    native = F.linear(x, w)
                    # A later env change must not alter this module's graph path.
                    with patch.dict(os.environ, {"RTP_BF16_GATE_KERNEL": "1"}), patch(
                        module + ".maybe_bf16_scalar_linear"
                    ) as scalar, patch(
                        module + ".maybe_bf16_gate_linear"
                    ) as gate, patch(
                        module + ".maybe_bf16_router_linear"
                    ) as router, patch(
                        module + ".maybe_bf16_gdn_linear"
                    ) as gdn:
                        self.assertTrue(torch.equal(op(x), native))
                        for prefill in (False, True):
                            with fusion_phase(is_prefill=prefill):
                                self.assertTrue(torch.equal(op(x), native))
                        for helper in (scalar, gate, router, gdn):
                            helper.assert_not_called()

    def test_phase_restored_after_exception(self):
        self.assertFalse(is_decode_phase())
        with self.assertRaisesRegex(RuntimeError, "test phase"):
            with fusion_phase(is_prefill=False):
                self.assertTrue(is_decode_phase())
                raise RuntimeError("test phase")
        self.assertFalse(is_decode_phase())

    @torch.inference_mode()
    def test_ordered_partial_sums_at_bf16_midpoint(self):
        # Sequential FP32 additions and a balanced tree can round to opposite
        # sides of a BF16 midpoint, despite each K-partition being identical.
        from itertools import permutations

        parts = sorted(set(permutations((1.0, 2.0**-8, 2.0**-24, 2.0**-24))))
        w = torch.zeros(128, 4096, device="cuda", dtype=torch.bfloat16)
        values = torch.tensor(
            [parts[i % len(parts)] for i in range(128)], dtype=torch.float32
        )
        for j in range(4):
            w[:, j * 1024] = values[:, j].to(device="cuda", dtype=torch.bfloat16)
        expected = (
            ((values[:, 0] + values[:, 1]) + values[:, 2]) + values[:, 3]
        ).bfloat16()
        balanced = (
            (values[:, 0] + values[:, 1]) + (values[:, 2] + values[:, 3])
        ).bfloat16()
        self.assertFalse(torch.equal(expected, balanced))
        for m in (1, 65, 128, 257, 512):
            x = torch.ones(m, 4096, device="cuda", dtype=torch.bfloat16)
            y = maybe_bf16_gate_linear(x, w)
            self.assertTrue(torch.equal(y, expected.cuda().expand(m, -1)), m)

    def test_empty_and_unsupported(self):
        w = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(0, 4096, device="cuda", dtype=torch.bfloat16)
        self.assertEqual(maybe_bf16_gate_linear(x, w).shape, (0, 128))
        self.assertIsNone(maybe_bf16_gate_linear(x.cpu(), w.cpu()))
        self.assertIsNone(maybe_bf16_gate_linear(x.float(), w.float()))
        self.assertIsNone(maybe_bf16_gate_linear(x, w[:64]))
        self.assertIsNone(maybe_bf16_gate_linear(x[:, :2048], w[:, :2048]))
        self.assertIsNone(
            maybe_bf16_gate_linear(
                x, w, torch.zeros(256, device="cuda", dtype=torch.bfloat16)[::2]
            )
        )


if __name__ == "__main__":
    unittest.main()
