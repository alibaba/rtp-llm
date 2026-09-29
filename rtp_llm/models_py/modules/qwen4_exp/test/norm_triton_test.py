"""Qwen4 fused Q/K norm parity and CUDA Graph replay tests."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.norm import Qwen4ExpFusedQKRMSNorm


class Qwen4FusedQKNormTest(unittest.TestCase):
    HEADS = 3
    KV_HEADS = 1
    DIM = 256

    def setUp(self):
        torch.manual_seed(21)
        self.q_gamma = (0.1 * torch.randn(self.DIM, device="cuda")).bfloat16()
        self.k_gamma = (0.1 * torch.randn(self.DIM, device="cuda")).bfloat16()
        self.norm = Qwen4ExpFusedQKRMSNorm(
            self.q_gamma, self.k_gamma, self.HEADS, self.KV_HEADS, self.DIM
        )

    def _input(self, rows):
        return torch.randn(
            rows,
            (self.HEADS + 2 * self.KV_HEADS) * self.DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )

    def _run(self, value, fused):
        with patch.dict(
            "os.environ", {"RTP_LLM_QWEN4_FUSED_QK_NORM": "1" if fused else "0"}
        ):
            return self.norm(value.clone())

    def test_matches_torch_and_preserves_v(self):
        for rows in (1, 3, 7, 128):
            with self.subTest(rows=rows):
                source = self._input(rows)
                reference = self._run(source, False)
                fused = self._run(source, True)
                torch.testing.assert_close(
                    fused.float(), reference.float(), atol=0.016, rtol=0.01
                )
                v_start = (self.HEADS + self.KV_HEADS) * self.DIM
                self.assertTrue(torch.equal(fused[:, v_start:], source[:, v_start:]))

    def test_gamma_plus_one_is_not_rounded_to_bf16(self):
        self.q_gamma.fill_(0.003)
        self.k_gamma.fill_(0.003)
        source = self._input(5)
        reference = self._run(source, False)
        fused = self._run(source, True)
        torch.testing.assert_close(
            fused.float(), reference.float(), atol=0.016, rtol=0.01
        )

    def test_replay_reads_new_inputs(self):
        static = self._input(4)
        with patch.dict("os.environ", {"RTP_LLM_QWEN4_FUSED_QK_NORM": "1"}):
            self.norm(static)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.norm(static)
            for _ in range(2):
                source = self._input(4)
                reference = self._run(source, False)
                static.copy_(source)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    static.float(), reference.float(), atol=0.016, rtol=0.01
                )


if __name__ == "__main__":
    unittest.main()
