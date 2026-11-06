"""Native FP4 Linear: online weights, dynamic activations and Graph replay."""

import unittest
from types import SimpleNamespace as NS

import torch

from rtp_llm.models.kimi_k3.nvfp4_tensor import (
    OnlineNvfp4Projection,
    nvfp4_global_scale,
)


def decode(codes, swizzled, outer):
    n, k = codes.shape[0], codes.shape[1] * 2
    rows, columns = (n + 127) // 128 * 128, (k // 16 + 3) // 4 * 4
    scales = swizzled.view(torch.float8_e4m3fn).reshape(
        rows // 128, columns // 4, 32, 4, 4
    )
    scales = scales.permute(0, 3, 2, 1, 4).reshape(rows, columns)[:n, : k // 16]
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=codes.device,
    )
    indices = torch.stack((codes & 15, codes >> 4), dim=-1).flatten(-2).long()
    return table[indices] * scales.float().repeat_interleave(16, -1) * outer


@unittest.skipUnless(
    torch.cuda.is_available(), "requires selected CUDA13 SM100/SM103 host"
)
class Nvfp4KernelReplayTest(unittest.TestCase):
    def projection(self, n=64, k=64, module="dsa.input", dynamic=True):
        from rtp_llm.config.quant_config import ModelOptFp4Config
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear import (
            CudaFp4GEMMLinear,
        )

        original = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) / 8
        p = OnlineNvfp4Projection(NS(name="w"), module=module)
        tensors = p._postprocess({"w": original}, "cuda", None)
        linear = CudaFp4GEMMLinear(
            weight=tensors["w"],
            weight_scales=tensors[p.local_key],
            weight_scale_2=tensors[p.global_key],
            input_scales=None if dynamic else torch.tensor([0.01], device="cuda"),
            quant_config=ModelOptFp4Config(
                bits=4,
                group_size=16,
                is_quanted=False,
                activation_scheme="dynamic" if dynamic else "static",
            ),
        )
        weight = decode(tensors["w"], tensors[p.local_key], tensors[p.global_key])
        quant_error = (weight[:n] - original.T.float()).norm() / original.float().norm()
        self.assertLess(quant_error.item(), 0.15)
        return linear, weight, int(tensors[p.logical_n_key])

    def check(self, linear, weight, x, actual=None):
        from flashinfer import fp4_quantize

        outer = (
            nvfp4_global_scale(x) if linear.dynamic_input_scale else linear.input_scale
        )
        codes, scales = fp4_quantize(x.contiguous(), outer.reciprocal())
        activation = decode(codes, scales, outer)
        quant_error = (activation - x.float()).norm() / x.float().norm().clamp_min(1e-8)
        self.assertLess(quant_error.item(), 0.15)
        expected = activation @ weight.T
        actual = linear(x) if actual is None else actual
        relative = (actual.float() - expected).norm() / expected.norm().clamp_min(1e-8)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertLess(relative.item(), 0.005)
        self.assertEqual(actual.dtype, torch.bfloat16)

    def test_dynamic_strided_zero_and_empty_inputs(self):
        linear, weight, _ = self.projection()
        for m in (1, 7, 33, 128):
            x = torch.randn(m, 128, device="cuda", dtype=torch.bfloat16)[:, ::2]
            self.check(linear, weight, x)
        self.check(
            linear, weight, torch.zeros(7, 64, device="cuda", dtype=torch.bfloat16)
        )
        self.check(
            linear, weight, torch.randn(7, 64, device="cuda", dtype=torch.float16)
        )
        self.assertEqual(
            linear(torch.empty(0, 64, device="cuda", dtype=torch.bfloat16)).shape,
            (0, 64),
        )
        self.assertEqual(
            linear(torch.empty(0, 64, device="cuda", dtype=torch.float16)).dtype,
            torch.bfloat16,
        )

    def test_kda_padding_and_graph_recomputes_activation_scale(self):
        linear, weight, logical_n = self.projection(n=580, module="kda.input")
        self.assertEqual((logical_n, linear.output_size), (580, 608))
        x = torch.randn(7, 64, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(3):
                linear(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = linear(x)
        for scale in (0.0, 0.001, 1.0, 100.0):
            x.copy_(torch.randn_like(x) * scale)
            graph.replay()
            self.check(linear, weight, x, actual)

    def test_static_linear_retains_existing_scale_contract(self):
        linear, weight, _ = self.projection(dynamic=False)
        self.check(
            linear, weight, torch.randn(7, 64, device="cuda", dtype=torch.bfloat16)
        )

    def test_dynamic_scale_on_decode_side_stream_graph_replay(self):
        from rtp_llm.models_py.utils.stream_dag import StreamDag

        first, first_weight, _ = self.projection()
        query, query_weight, _ = self.projection()
        state = NS(input=torch.randn(7, 64, device="cuda", dtype=torch.bfloat16))
        dag = StreamDag(torch.cuda.current_device(), side_streams=("query",))
        dag.add("input", lambda s: setattr(s, "hidden", first(s.input)))
        dag.add("query", lambda s: setattr(s, "query", query(s.hidden)),
                stream="query", after=("input",))
        dag.add("join", lambda s: setattr(s, "output", s.query.clone()), after=("query",))
        dag.finalize()
        for _ in range(3):
            dag.run(state, multi_stream=True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dag.run(state, multi_stream=True)
        for scale in (0.0, 0.001, 1.0, 100.0):
            state.input.copy_(torch.randn_like(state.input) * scale)
            graph.replay()
            self.check(first, first_weight, state.input, state.hidden)
            self.check(query, query_weight, state.hidden, state.output)

    def test_natural_q_gate_and_kv_keep_the_original_nvfp4_operands(self):
        from rtp_llm.models_py.modules.kimi_k3.natural_dsa import (
            split_natural_projection,
            NaturalQGateProjection,
        )

        linear, _, _ = self.projection(n=416)
        q, kv, gate = split_natural_projection(linear, 64, 96, 256, 0)
        merged = NaturalQGateProjection(q, gate)
        self.assertEqual(merged.nvfp4.weight.dtype, torch.uint8)
        self.assertEqual(kv.weight.dtype, torch.uint8)
        x = torch.randn(7, 64, device="cuda", dtype=torch.bfloat16)
        expected = linear(x).float()
        q_out, g_out = merged(x)
        actual = torch.cat((q_out, kv(x), g_out), -1).float()
        self.assertLess(((actual - expected).norm() / expected.norm()).item(), 0.005)


if __name__ == "__main__":
    unittest.main()
