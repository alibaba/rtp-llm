"""Real CPU allocations cover gathered decode rows and retain graph generations."""

import ast
import os
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4.moe.strategies.sm120_decode_experts import (
    Sm120DecodeExperts,
)


class DecodeWorkspaceCapacityTest(unittest.TestCase):
    def setUp(self):
        self.old_cache = Sm120DecodeExperts._sm120_masked_ws_cache
        Sm120DecodeExperts._sm120_masked_ws_cache = []
        self.addCleanup(
            setattr, Sm120DecodeExperts, "_sm120_masked_ws_cache", self.old_cache
        )
        self.experts = Sm120DecodeExperts(SimpleNamespace())

    def allocate(self, rows, experts=2):
        return self.experts._ensure_sm120_masked_workspace(
            rows, 2, experts, 128, 512, ((rows + 127) // 128) * 128, torch.device("cpu")
        )

    def test_post_gather_shapes_cover_all_capture_batches(self):
        for world in (1, 4, 8):
            for batch in (1, 2, 4, 16, 128):
                rows = world * max(4, batch * 4)
                ws = self.allocate(rows)
                self.assertGreaterEqual(ws["n"], rows)
                self.assertGreaterEqual(ws["alignment"], rows)
                self.assertGreaterEqual(ws["gather"].shape[0], rows)
                self.assertEqual(ws["expert_x"].shape, (2, ws["alignment"], 128))
                self.assertEqual(ws["expert_x_scale"].shape, (2, ws["alignment"], 1))
                self.assertTrue(torch.all(ws["expert_x_scale"] == 0))
                self.assertEqual(ws["down_in_scale"].shape, (2, ws["alignment"], 1))
                self.assertEqual(ws["down_in_scale"].dtype, torch.int32)

    def test_incomplete_packed_scale_group_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "divisible by 4"):
            self.experts._ensure_sm120_masked_workspace(
                32, 2, 2, 128, 128, 128, torch.device("cpu")
            )
        self.assertEqual(Sm120DecodeExperts._sm120_masked_ws_cache, [])

    def test_smaller_capture_reuses_storage(self):
        large = self.allocate(256)
        small = self.allocate(32)
        self.assertIs(large, small)
        self.assertEqual(large["gather"].data_ptr(), small["gather"].data_ptr())

    def test_growth_retains_existing_graph_storage(self):
        small = self.allocate(32)
        pointer = small["gather"].data_ptr()
        large = self.allocate(512)
        self.assertIsNot(small, large)
        self.assertIs(Sm120DecodeExperts._sm120_masked_ws_cache[0], small)
        self.assertEqual(small["gather"].data_ptr(), pointer)
        self.assertEqual(small["gather"].shape[0], 32)

    def test_expert_partitions_do_not_alias(self):
        a, b = self.allocate(32, 2), self.allocate(32, 4)
        self.assertNotEqual(a["expert_x"].data_ptr(), b["expert_x"].data_ptr())
        self.assertEqual(b["expert_x"].shape[0], 4)


class WoAProjectionShapeTest(unittest.TestCase):
    def setUp(self):
        path = pathlib.Path(__file__).parents[1] / "fp8/attention.py"
        tree = ast.parse(path.read_text())
        methods = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_wo_a_einsum_from_fp8"
        ]
        self.assertEqual(len(methods), 1)
        self.deep_gemm = SimpleNamespace(fp8_einsum=mock.Mock())
        namespace = {
            "torch": torch,
            "os": os,
            "is_sm120": lambda device: True,
            "deep_gemm": self.deep_gemm,
        }
        exec(
            compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"),
            namespace,
        )
        self.project = namespace["_wo_a_einsum_from_fp8"]
        self.owner = SimpleNamespace(
            o_lora_rank=128,
            wo_a_s=torch.ones((2, 1, 4)),
            _wo_a_stk_w=torch.ones((2, 128, 512)).to(torch.float8_e4m3fn),
            _wo_a_stk_s=object(),
        )
        self.x = torch.ones((3, 2, 512)).to(torch.float8_e4m3fn)
        self.scales = torch.full((3, 2, 4), 127, dtype=torch.uint8).view(torch.int32)

    def test_sm120_fallback_retains_hidden_width_and_trims_padding(self):
        calls = []

        def gemm(a, weight, a_scale, w_scale, **kwargs):
            self.assertEqual(a.shape, (4, 512))
            self.assertEqual(a_scale.shape, (4, 4))
            self.assertEqual(w_scale.shape, (1, 4))
            self.assertTrue(torch.all(a_scale == 1))
            self.assertTrue(torch.all(a[3].float() == 0))
            self.assertEqual(kwargs["scale_granularity_mnk"], (1, 128, 128))
            calls.append(True)
            return (a.float() @ weight.float().T).to(torch.bfloat16)

        with mock.patch.dict(
            os.environ, {"DSV4_SM120_WOA_EINSUM": "0"}
        ), mock.patch.dict(
            sys.modules,
            {"flashinfer.gemm": SimpleNamespace(gemm_fp8_nt_groupwise=gemm)},
        ):
            result = self.project(self.owner, self.x, self.scales, 1, 3)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.shape, (1, 3, 2, 128))
        self.assertTrue(torch.equal(result, torch.full_like(result, 512)))
        self.deep_gemm.fp8_einsum.assert_not_called()

    def test_prefill_opt_in_keeps_the_einsum_path(self):
        def einsum(equation, activation, weight, out, **kwargs):
            self.assertEqual(equation, "bhr,hdr->bhd")
            self.assertIs(activation[0], self.x)
            self.assertIs(activation[1], self.scales)
            self.assertEqual(kwargs["recipe"], (1, 1, 128))
            out.fill_(7)

        self.deep_gemm.fp8_einsum.side_effect = einsum
        with mock.patch.dict(os.environ, {"DSV4_SM120_WOA_EINSUM": "1"}):
            result = self.project(self.owner, self.x, self.scales, 1, 3)
        self.assertEqual(result.shape, (1, 3, 2, 128))
        self.assertTrue(torch.equal(result, torch.full_like(result, 7)))
        self.deep_gemm.fp8_einsum.assert_called_once()


if __name__ == "__main__":
    unittest.main()
