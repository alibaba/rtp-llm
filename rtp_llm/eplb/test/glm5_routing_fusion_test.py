import unittest
from unittest.mock import patch

import torch

from rtp_llm.eplb.test.test_support import make_layer
from rtp_llm.models_py.modules.glm5_mega_moe import cmp_router_triton
from rtp_llm.models_py.modules.glm5_mega_moe.input_packer_triton import (
    fused_pack_mega_moe_inputs,
)


class Glm5RoutingFusionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7413)
        self.plan = make_layer(
            tuple(range(256)) + tuple(range(32)) * 2, device="cuda", replica_offset=7
        )

    def test_cmp_matches_native_gate_and_clears_workspace(self):
        import rtp_kernel.glm5 as ops

        bias = torch.randn(256, device="cuda") * 0.1
        for rows in (1, 16, 17, 64, 257):
            for pdl in (False, True):
                with self.subTest(rows=rows, pdl=pdl):
                    logits = torch.randn(rows, 256, device="cuda") * 5
                    logits[0].zero_()
                    source = logits.clone()
                    native_ids, native_weights = ops.router_topk(source, bias)
                    expected_ids = self.plan.map_ids(native_ids)
                    ids, weights = torch.empty_like(native_ids), torch.empty_like(
                        native_weights
                    )
                    cmp_router_triton.router_topk(
                        logits, bias, ids, weights, self.plan, pdl=pdl
                    )
                    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
                    torch.testing.assert_close(weights, native_weights, rtol=0, atol=0)
                    self.assertEqual(torch.count_nonzero(logits).item(), 0)

    def test_cmp_ties_nonfinite_and_graph_map_updates(self):
        import rtp_kernel.glm5 as ops

        logits = torch.zeros(32, 256, device="cuda")
        logits[1, :16] = float("nan")
        logits[2, :16] = float("inf")
        logits[3].fill_(-float("inf"))
        bias = torch.zeros(256, device="cuda")
        native_ids, native_weights = ops.router_topk(logits.clone(), bias)
        ids, weights = torch.empty_like(native_ids), torch.empty_like(native_weights)
        source = logits.clone()

        def call():
            source.copy_(logits)
            cmp_router_triton.router_topk(
                source, bias, ids, weights, self.plan, pdl=True
            )

        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        for _ in range(2):
            self.plan.logical_to_physical.add_(3)
            graph.replay()
            torch.testing.assert_close(
                ids, self.plan.map_ids(native_ids), rtol=0, atol=0
            )
            torch.testing.assert_close(
                weights, native_weights, rtol=0, atol=0, equal_nan=True
            )
            self.assertEqual(torch.count_nonzero(source).item(), 0)

    def pack(self, x, weights, ids, *, plan=None, offset=0, shared=False):
        rows, dim = x.shape
        q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        sf = torch.empty((rows, dim // 128), device="cuda", dtype=torch.int32)
        output_ids, output_weights = torch.empty_like(ids), torch.empty_like(weights)
        shared_sf = (
            torch.empty(
                ((rows + 63) // 64 * 128, dim // 128), device="cuda", dtype=torch.int32
            )
            if shared
            else None
        )
        fused_pack_mega_moe_inputs(
            x,
            weights,
            ids,
            q,
            sf,
            output_ids,
            output_weights,
            out_shared_sf=shared_sf,
            shared_block_m=64 if shared else 0,
            eplb=plan,
            token_offset=offset,
        )
        return q.view(torch.uint8), sf, output_ids, output_weights, shared_sf

    def test_pack_preserves_quant_weights_padding_and_chunk_rotation(self):
        for rows in (0, 1, 17, 2053):
            for shared in (False, True):
                with self.subTest(rows=rows, shared=shared):
                    x = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
                    weights = torch.rand(rows, 8, device="cuda")
                    ids = torch.randint(0, 256, (rows, 8), device="cuda")
                    ids[:, -1] = -1
                    for offset in (0, 200000):
                        reference_ids = self.plan.map_ids(
                            torch.cat(
                                (
                                    torch.zeros(
                                        offset, 8, device="cuda", dtype=torch.int64
                                    ),
                                    ids,
                                )
                            )
                        )[offset:]
                        expected = self.pack(x, weights, reference_ids, shared=shared)
                        actual = self.pack(
                            x,
                            weights,
                            ids,
                            plan=self.plan,
                            offset=offset,
                            shared=shared,
                        )
                        for a, b in zip(actual, expected):
                            if a is not None:
                                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_pack_graph_borrows_updated_maps(self):
        x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
        ids = torch.zeros(64, 8, device="cuda", dtype=torch.int64)
        weights = torch.ones(64, 8, device="cuda")
        self.pack(x, weights, ids, plan=self.plan)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.pack(x, weights, ids, plan=self.plan)
        for _ in range(2):
            self.plan.logical_to_physical.add_(1)
            graph.replay()
            torch.testing.assert_close(
                output[2], self.plan.map_ids(ids), rtol=0, atol=0
            )

    def test_cmp_router_projection_graph_chain(self):
        import rtp_kernel.glm5 as ops

        x = torch.randn(64, 6144, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(256, 6144, device="cuda", dtype=torch.bfloat16) * 0.02
        bias = torch.randn(256, device="cuda") * 0.1
        logits = torch.zeros(64, 256, device="cuda")
        ids = torch.empty(64, 8, device="cuda", dtype=torch.int64)
        weights = torch.empty(64, 8, device="cuda")
        original_pdl = ops.get_pdl()
        try:
            ops.set_pdl(True)
            reference_ids, reference_weights = ops.router_topk(
                ops.router_proj(x, weight), bias
            )

            def call():
                ops.router_proj(x, weight, out=logits)
                cmp_router_triton.router_topk(
                    logits, bias, ids, weights, self.plan, pdl=True
                )

            call()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            for _ in range(3):
                graph.replay()
                torch.testing.assert_close(
                    ids, self.plan.map_ids(reference_ids), rtol=0, atol=0
                )
                torch.testing.assert_close(weights, reference_weights, rtol=0, atol=0)
                self.assertEqual(torch.count_nonzero(logits).item(), 0)
        finally:
            ops.set_pdl(original_pdl)

    def test_switch_and_layout_support_gate(self):
        logits = torch.zeros(16, 256, device="cuda")
        bias = torch.zeros(256, device="cuda")
        ids = torch.empty(16, 8, device="cuda", dtype=torch.int64)
        weights = torch.empty(16, 8, device="cuda")
        self.assertTrue(
            cmp_router_triton.is_supported(logits, bias, ids, weights, self.plan)
        )
        with patch.dict("os.environ", {"RTP_LLM_GLM5_EPLB_FUSED_ROUTING": "0"}):
            self.assertFalse(
                cmp_router_triton.is_supported(logits, bias, ids, weights, self.plan)
            )

    def test_fusion_launch_counts(self):
        import rtp_kernel.glm5 as ops
        from torch.profiler import ProfilerActivity, profile

        logits = torch.randn(64, 256, device="cuda")
        bias = torch.zeros(256, device="cuda")
        ids = torch.empty(64, 8, device="cuda", dtype=torch.int64)
        weights = torch.empty(64, 8, device="cuda")
        x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)

        def fused_gate():
            cmp_router_triton.router_topk(logits, bias, ids, weights, self.plan)

        def old_gate():
            ops.router_topk(logits, bias, topk_indices=ids, topk_weights=weights)
            ids.copy_(self.plan.map_ids(ids))

        def fused_pack():
            self.pack(x, weights, ids, plan=self.plan)

        def old_pack():
            self.pack(x, weights, self.plan.map_ids(ids))

        counts = {}
        for name, fn in (
            ("gate_old", old_gate),
            ("gate_fused", fused_gate),
            ("pack_old", old_pack),
            ("pack_fused", fused_pack),
        ):
            ids.zero_()
            fn()
            ids.zero_()
            torch.cuda.synchronize()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                fn()
                torch.cuda.synchronize()
            kernels = [
                event.name
                for event in prof.events()
                if event.device_type == torch.autograd.DeviceType.CUDA
            ]
            counts[name] = len(kernels)
            print(name, kernels, flush=True)
        self.assertEqual(counts["gate_fused"], 1)
        self.assertEqual(counts["pack_fused"], 1)
        self.assertGreater(counts["gate_old"], counts["gate_fused"])
        self.assertGreater(counts["pack_old"], counts["pack_fused"])


if __name__ == "__main__":
    unittest.main()
