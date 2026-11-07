"""K3.1 DSA decode prologue (DSA_DCP_PROLOGUE) pieces on one GPU: weights, BF16x3 head gate, StreamDag.

- The upstream loader replicates decode q_b and kc independently of the prologue switch.
- split_bf16x3: the three BF16 parts sum to the FP32 head-gate weight, and one BF16 GEMM with FP32
  output over them matches the FP64 product as closely as the FP32 GEMM the dev path runs.
- StreamDag: a DAG over the main and two side streams gives the serial result, eagerly and when
  replayed from a CUDA graph with new inputs.
"""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.kimi_k3.dsa import split_bf16x3
from rtp_llm.models_py.utils.stream_dag import StreamDag


class LoaderTest(unittest.TestCase):
    def test_decode_query_heads_replicate_independently_of_prologue(self):
        from rtp_llm.models.kimi_k3.kimi_k31_weight import KimiK31MtpWeight, KimiK31Weight
        from rtp_llm.ops import RoleType

        for weights, mode, role, replicated in (
            (KimiK31Weight, "serial", RoleType.DECODE, True),
            (KimiK31Weight, "multi_stream", RoleType.DECODE, True),
            (KimiK31Weight, "dev", RoleType.DECODE, True),
            (KimiK31Weight, "serial", RoleType.PREFILL, False),
            (KimiK31MtpWeight, "serial", RoleType.DECODE, True),
        ):
            manifest = object.__new__(weights)
            manifest.prefill_query_cp, manifest.role_type, manifest.tp_size = False, role, 8
            with self.subTest(weights=weights.__name__, mode=mode, role=role), \
                    patch.dict(os.environ, {"DSA_DCP_PROLOGUE": mode}):
                self.assertEqual(manifest._replicate_mla_query_heads(), replicated)


class HeadGateTest(unittest.TestCase):
    def test_split_is_exact_and_gemm_matches_fp32(self):
        torch.manual_seed(7)
        hidden, heads = 7168, 32
        weight = torch.randn(hidden, heads, device="cuda") * hidden**-0.5
        parts = split_bf16x3(weight)
        total = parts[:, :heads].float() + parts[:, heads : 2 * heads].float() + parts[:, 2 * heads :].float()
        self.assertTrue(torch.equal(total, weight))
        for rows in (1, 8, 74, 512):
            with self.subTest(rows=rows):
                x = torch.randn(rows, hidden, device="cuda").bfloat16()
                gate = torch.mm(x, parts, out_dtype=torch.float32).view(rows, 3, heads)
                gate = gate[:, 0] + gate[:, 1] + gate[:, 2]
                exact = (x.double() @ weight.double()).float()
                dev = x.float() @ weight
                scale = exact.abs().max()
                self.assertLess(((gate - exact).abs().max() / scale).item(), 1e-5)
                self.assertLess(((dev - exact).abs().max() / scale).item(), 1e-5)


class StreamDagTest(unittest.TestCase):
    def build(self):
        dag = StreamDag(torch.cuda.current_device(), side_streams=("a", "b"))
        dag.add("x", lambda s: setattr(s, "x", s.input * 2))
        dag.add("ya", lambda s: setattr(s, "ya", s.x @ s.w1), stream="a", after=("x",))
        dag.add("yb", lambda s: setattr(s, "yb", s.x @ s.w2), stream="b", after=("x",))
        dag.add("z", lambda s: setattr(s, "z", s.x.sum(-1, keepdim=True)))
        dag.add("out", lambda s: setattr(s, "out", s.ya + s.yb + s.z), after=("ya", "yb", "z"))
        dag.finalize()
        return dag

    def state(self, seed):
        from types import SimpleNamespace

        gen = torch.Generator(device="cuda").manual_seed(seed)
        return SimpleNamespace(**{name: torch.randn(*shape, generator=gen, device="cuda")
                                  for name, shape in (("input", (256, 512)), ("w1", (512, 512)), ("w2", (512, 512)))})

    def test_eager(self):
        dag = self.build()
        serial, multi = self.state(1), self.state(1)
        dag.run(serial, multi_stream=False)
        dag.run(multi, multi_stream=True)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(serial.out, multi.out))

    def test_graph_replay(self):
        dag = self.build()
        state = self.state(2)
        dag.run(state, multi_stream=True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dag.run(state, multi_stream=True)
        for seed in (3, 4):
            fresh = self.state(seed)
            for name in ("input", "w1", "w2"):
                getattr(state, name).copy_(getattr(fresh, name))
            graph.replay()
            dag.run(fresh, multi_stream=False)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(state.out, fresh.out))


if __name__ == "__main__":
    unittest.main()
