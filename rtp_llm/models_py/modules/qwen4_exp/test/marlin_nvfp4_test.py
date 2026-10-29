import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.qwen4_exp.moe.marlin_nvfp4 import (
    Nvfp4MarlinMoE,
    Qwen4DecodeMarlinExecutor,
    _linear_scales,
    align,
)
from rtp_llm.models_py.modules.qwen4_exp.weight_views import PreparedWeightDict
from rtp_llm.utils.model_weight import W


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "requires SM120",
)
class MarlinNvfp4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(918)
        cls.n, cls.k, cls.experts = 640, 2560, 4
        cls.q1 = torch.randint(256, (4, 1280, 1280), device="cuda", dtype=torch.uint8)
        cls.q2 = torch.randint(256, (4, 2560, 320), device="cuda", dtype=torch.uint8)
        cls.s1 = (2.0 ** torch.randint(-9, 9, (4, 1280, 160), device="cuda")).to(
            torch.float8_e4m3fn
        )
        cls.s2 = (2.0 ** torch.randint(-9, 9, (4, 2560, 40), device="cuda")).to(
            torch.float8_e4m3fn
        )
        cls.s1[0].zero_()
        cls.g1 = torch.tensor([1e-5, 1e-4, 2e-4, 5e-5], device="cuda")
        cls.g2 = torch.tensor([2e-4, 5e-5, 1e-4, 1e-5], device="cuda")
        cls.models = [
            Nvfp4MarlinMoE(cls.q1, cls.s1, cls.g1, cls.q2, cls.s2, cls.g2, 2, rank)
            for rank in (0, 1)
        ]
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device="cuda",
        )

        def dequant(q, sf, glob):
            codes = torch.stack((q & 15, q >> 4), -1).flatten(-2).long()
            return (
                lut[codes] * sf.float().repeat_interleave(16, -1) * glob[:, None, None]
            )

        cls.w1 = dequant(cls.q1, cls.s1, cls.g1)
        cls.w2 = dequant(cls.q2, cls.s2, cls.g2)

    def inputs(self, rows, rank=0, local=None):
        x = torch.randn(rows, self.k, device="cuda", dtype=torch.bfloat16) * 0.2
        ids = torch.rand(rows, 8, device="cuda").topk(3, -1).indices.int()
        if local is not None:
            ids = (
                torch.tensor(local, device="cuda", dtype=torch.int32)
                .expand(rows, -1)
                .contiguous()
            )
        weights = torch.randn(rows, 3, device="cuda").softmax(-1)
        return x, ids, weights

    def reference(self, x, ids, weights, rank):
        result = torch.zeros_like(x, dtype=torch.float32)
        for row in range(x.shape[0]):
            for slot in range(ids.shape[1]):
                expert = int(ids[row, slot]) - rank * self.experts
                if not 0 <= expert < self.experts:
                    continue
                upgate = self.w1[expert] @ x[row].float()
                up, gate = upgate.chunk(2)
                result[row] += (
                    self.w2[expert] @ (up * torch.nn.functional.silu(gate))
                ) * weights[row, slot]
        return result

    def assert_reference(self, actual, expected):
        self.assertTrue(bool(torch.isfinite(actual).all()))
        if not bool(expected.any()):
            self.assertTrue(bool((actual == 0).all()))
            return
        relative_rmse = (
            ((actual.float() - expected).square().mean() / expected.square().mean())
            .sqrt()
            .item()
        )
        self.assertLess(relative_rmse, 0.02)

    def test_scale_range_zero_expert_and_fp32_weight_reference(self):
        for rank in (0, 1):
            args = self.inputs(3, rank, [rank * 4, rank * 4 + 1, rank * 4 + 3])
            self.assert_reference(self.models[rank](*args), self.reference(*args, rank))

    def test_ep_global_ids_and_all_nonlocal_routes(self):
        for rank in (0, 1):
            for rows in (1, 4, 8, 32, 64):
                args = self.inputs(rows, rank)
                self.assert_reference(
                    self.models[rank](*args), self.reference(*args, rank)
                )
            args = self.inputs(4, rank, [4, 5, 6] if rank == 0 else [0, 1, 2])
            self.assertTrue(bool((self.models[rank](*args) == 0).all()))

    def test_graph_refreshes_values_ids_weights(self):
        for rank in (0, 1):
            for rows in (1, 4, 8):
                x, ids, weights = self.inputs(rows, rank)
                model = self.models[rank]
                model(x, ids, weights)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = model(x, ids, weights)
                for _ in range(3):
                    new = self.inputs(rows, rank)
                    for dst, src in zip((x, ids, weights), new):
                        dst.copy_(src)
                    expected = model(x, ids, weights)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_alignment_stable_order_padding_and_nonlocal_routes(self):
        for rows in (1, 4, 8, 9, 32):
            for first in (0, 256):
                for mode in ("random", "same", "nonlocal", "boundary"):
                    ids = torch.randint(
                        0, 512, (rows, 10), device="cuda", dtype=torch.int32
                    )
                    if mode == "same":
                        ids.fill_(first + 3)
                    elif mode == "nonlocal":
                        ids.fill_(256 - first)
                    elif mode == "boundary":
                        ids.flatten()[::2] = first
                        ids.flatten()[1::2] = first + 255
                    routes = ids.numel()
                    active = min(routes, 256)
                    capacity = active + (routes - active) // 8
                    flat = ids.flatten().tolist()
                    expected_ids, expected_experts = [], []
                    for expert in range(256):
                        selected = [
                            i for i, value in enumerate(flat) if value == first + expert
                        ]
                        count = (len(selected) + 7) // 8
                        expected_ids.extend(
                            selected + [routes] * (count * 8 - len(selected))
                        )
                        expected_experts.extend([expert] * count)
                    padded = len(expected_ids)
                    expected_ids.extend([routes] * (capacity * 8 - padded))
                    expected_experts.extend([0] * (capacity - len(expected_experts)))
                    actual = align(ids, 256, first)
                    for tensor, expected in zip(
                        actual, (expected_ids, expected_experts, [padded])
                    ):
                        self.assertEqual(tensor.tolist(), expected)

    def test_concurrent_stream_workspaces(self):
        model = self.models[0]
        args = [self.inputs(8) for _ in range(2)]
        expected = [model(*a) for a in args]
        streams = [torch.cuda.Stream() for _ in range(2)]
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
        actual = []
        for stream, a in zip(streams, args):
            with torch.cuda.stream(stream):
                actual.append(model(*a))
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=0, atol=0)

    def test_repack_reuses_original_storage_and_executor_contract(self):
        def swizzle(sf):
            e, n, k = sf.shape
            return (
                sf.view(torch.uint8)
                .reshape(e, n // 128, 4, 32, k // 4, 4)
                .permute(0, 1, 4, 3, 2, 5)
                .contiguous()
                .reshape(e, n, k)
                .view(torch.int32)
            )

        q1, q2 = self.q1.clone(), self.q2.clone()
        sf1, sf2 = swizzle(self.s1), swizzle(self.s2)
        torch.testing.assert_close(
            _linear_scales(sf1, 4, 1280, 160).float(), self.s1.float(), rtol=0, atol=0
        )
        native = SimpleNamespace(
            config=None,
            quant_config=None,
            weights=PreparedWeightDict(),
            ep_size=2,
            ep_rank=1,
            tp_size=1,
            tp_rank=0,
            w1=q1.view(torch.int64),
            w2=q2.view(torch.int64),
            quant_scales=[None, sf1, None, None, sf2, None],
        )
        native.weights.update(
            {
                W.moe_w1: q1,
                W.moe_w2: q2,
                W.moe_s1: sf1,
                W.moe_s2: sf2,
                W.moe_w1_s2: self.g1,
                W.moe_w2_s2: self.g2,
            }
        )
        ptrs = [v.data_ptr() for v in (q1, q2, sf1, sf2)]
        executor = Qwen4DecodeMarlinExecutor(native)
        self.assertEqual(
            ptrs,
            [
                v.data_ptr()
                for v in (
                    executor.moe.w1,
                    executor.moe.w2,
                    executor.moe.s1,
                    executor.moe.s2,
                )
            ],
        )
        args = self.inputs(4, 1)
        payload = ExpertForwardPayload(
            expert_x=args[0], expert_topk_ids=args[1], expert_topk_weights=args[2]
        )
        actual = executor.execute(
            payload, "silu", None, None, False, None
        ).fused_expert_output
        self.assert_reference(actual, self.reference(*args, 1))
        self.assertTrue(
            all(
                v.update_requires_restart
                for v in native.weights.consumer_views.values()
            )
        )
        del executor
        # A later model can reuse the packed tensors after the first model dies.
        # It must rebuild views and workspaces without repacking their bytes.
        for _ in range(2):
            executor = Qwen4DecodeMarlinExecutor(native)
            repeated = executor.execute(
                payload, "silu", None, None, False, None
            ).fused_expert_output
            torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
            one_row = self.inputs(1, 1)
            one_payload = ExpertForwardPayload(
                expert_x=one_row[0],
                expert_topk_ids=one_row[1],
                expert_topk_weights=one_row[2],
            )
            one_output = executor.execute(
                one_payload, "silu", None, None, False, None
            ).fused_expert_output
            self.assert_reference(one_output, self.reference(*one_row, 1))
        for activation, expert_map, apply_input in [
            ("relu", None, False),
            ("silu", torch.zeros(8, device="cuda"), False),
            ("silu", None, True),
        ]:
            with self.assertRaises(ValueError):
                executor.execute(
                    payload, activation, expert_map, None, apply_input, None
                )


if __name__ == "__main__":
    unittest.main()
