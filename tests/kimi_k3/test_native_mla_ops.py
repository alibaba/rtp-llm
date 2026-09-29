import unittest

import torch

from rtp_llm.models_py.modules.kimi_k3.native_mla_ops import (
    fused_q_kv_rmsnorm,
    gate_sigmoid_mul,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class NativeMLAOpsTest(unittest.TestCase):
    def test_strided_norm_and_gate_changed_input_graph(self):
        qw = ((torch.arange(1536, device="cuda") % 8) * 0.125).bfloat16()
        kw = ((torch.arange(512, device="cuda") % 8) * 0.125).bfloat16()
        for rows in (1, 3, 8, 9):
            fused = torch.ones((rows, 1536 + 512 + 64 + 1536),
                               device="cuda", dtype=torch.bfloat16)
            q, kv, gate = fused[:, :1536], fused[:, 1536:2048], fused[:, -1536:]
            gate.zero_()
            attn = torch.ones_like(gate)
            def run():
                qo, ko = fused_q_kv_rmsnorm(q, kv, qw, kw, 3.0)
                return qo, ko, gate_sigmoid_mul(attn, gate)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
                run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                qo, ko, gated = run()
            for sign in (-1, 0, 1):
                q.fill_(sign)
                kv.fill_(-sign)
                attn.fill_(sign)
                graph.replay()
                self.assertTrue(torch.equal(qo, (qw * (0.5 * sign)).expand(rows, -1)))
                self.assertTrue(torch.equal(ko, (kw * (-0.5 * sign)).expand(rows, -1)))
                self.assertTrue(torch.equal(gated, torch.full_like(gated, sign * 0.5)))

    def test_empty_and_invalid_contracts(self):
        q = torch.ones((2, 1536), device="cuda", dtype=torch.bfloat16)
        kv = torch.ones((2, 512), device="cuda", dtype=torch.bfloat16)
        qw, kw = q[0].clone(), kv[0].clone()
        empty = fused_q_kv_rmsnorm(q[:0], kv[:0], qw, kw, 1e-5)
        self.assertEqual([x.shape for x in empty], [q[:0].shape, kv[:0].shape])
        for bad_q, bad_kv, bad_qw in ((q.float(), kv, qw), (q, kv[:1], qw),
                                     (q[:, ::2], kv, qw[::2]), (q, kv, qw[:-1])):
            with self.assertRaises(ValueError):
                fused_q_kv_rmsnorm(bad_q, bad_kv, bad_qw, kw, 1e-5)


if __name__ == "__main__":
    unittest.main()
