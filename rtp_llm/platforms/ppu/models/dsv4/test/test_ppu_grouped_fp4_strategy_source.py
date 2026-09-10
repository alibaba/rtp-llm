"""Source-only contract tests for the TP4/EP1 PPU grouped-MXFP4 strategy."""

import ast
import math
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Sequence, Tuple
from unittest.mock import patch

_SOURCE_PATH = Path(__file__).resolve().parents[1] / "ppu_grouped_fp4.py"


def _load_helpers():
    tree = ast.parse(_SOURCE_PATH.read_text())
    wanted = {
        "_supports_topology",
        "_runtime_eligible",
        "_derive_inter_local_and_tp",
        "_select_capacity",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {
        "math": math,
        "Sequence": Sequence,
        "Tuple": Tuple,
        "MoeCfg": object,
        "torch": SimpleNamespace(),
        "_GROUPED_M_ALIGNMENT": 128,
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(_SOURCE_PATH), "exec"),
        namespace,
    )
    return tree, namespace


def _cfg(*, tp_size=4, ep_size=1, inter=2048, experts=256, dim=7168):
    return SimpleNamespace(
        tp_size=tp_size,
        ep_size=ep_size,
        dim=dim,
        moe_inter_dim=inter,
        n_local_experts=experts,
        n_routed_experts=experts,
    )


def _shapes(inter_local: int, experts: int = 256, dim: int = 7168):
    w1 = (experts, inter_local, dim // 2)
    w2 = (experts, dim, inter_local // 2)
    s1 = (experts, inter_local, dim // 32)
    s2 = (experts, dim, inter_local // 32)
    return w1, w2, w1, s1, s2, s1


class PpuGroupedFP4SourceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = _SOURCE_PATH.read_text()
        cls.tree, cls.helpers = _load_helpers()

    def test_strategy_identity_and_exact_topology(self):
        supports = self.helpers["_supports_topology"]
        self.assertTrue(supports(_cfg(tp_size=4, ep_size=1)))
        self.assertFalse(supports(_cfg(tp_size=1, ep_size=1)))
        self.assertFalse(supports(_cfg(tp_size=4, ep_size=8)))
        self.assertFalse(supports(_cfg(tp_size=8, ep_size=1)))

        strategy = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "PpuGroupedFP4Strategy"
        )
        assignments = {
            target.id: ast.literal_eval(stmt.value)
            for stmt in strategy.body
            if isinstance(stmt, ast.Assign)
            for target in stmt.targets
            if isinstance(target, ast.Name) and isinstance(stmt.value, ast.Constant)
        }
        self.assertEqual(assignments["name"], "ppu_grouped_fp4")
        can_handle = next(
            node
            for node in strategy.body
            if isinstance(node, ast.FunctionDef) and node.name == "can_handle"
        )
        self.assertIn("_supports_topology(cfg)", ast.unparse(can_handle))
        self.assertIn("_runtime_eligible()", ast.unparse(can_handle))

    def test_runtime_eligibility_rejects_generic_gpu_and_missing_symbol(self):
        runtime_eligible = self.helpers["_runtime_eligible"]

        class FakeCuda:
            def __init__(self, *, available, name):
                self.available = available
                self.name = name

            def is_available(self):
                return self.available

            def current_device(self):
                return 0

            def get_device_name(self, device):
                self.test_case.assertEqual(device, 0)
                return self.name

        def set_cuda(*, available, name):
            cuda = FakeCuda(available=available, name=name)
            cuda.test_case = self
            self.helpers["torch"] = SimpleNamespace(cuda=cuda)

        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad = lambda *args: None

        set_cuda(available=False, name="ZW-M890P")
        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            self.assertFalse(runtime_eligible())

        set_cuda(available=True, name="NVIDIA H100")
        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            self.assertFalse(runtime_eligible())

        set_cuda(available=True, name="ZW-M890P")
        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            self.assertTrue(runtime_eligible())

        missing_symbol = ModuleType("deep_gemm")
        with patch.dict(sys.modules, {"deep_gemm": missing_symbol}):
            self.assertFalse(runtime_eligible())

        with patch.dict(sys.modules, {"deep_gemm": None}):
            self.assertFalse(runtime_eligible())

    def test_packed_geometry_detects_pure_tp_preshard(self):
        derive = self.helpers["_derive_inter_local_and_tp"]
        inter_local, routed_tp = derive(_cfg(), *_shapes(512))
        self.assertEqual((inter_local, routed_tp), (512, 4))

        inter_local, routed_tp = derive(_cfg(), *_shapes(2048))
        self.assertEqual((inter_local, routed_tp), (2048, 1))

        with self.assertRaises(ValueError):
            derive(_cfg(), *_shapes(1024))
        bad = list(_shapes(512))
        bad[1] = (256, 7168, 255)
        with self.assertRaises(ValueError):
            derive(_cfg(), *bad)
        with self.assertRaises(RuntimeError):
            derive(_cfg(tp_size=1), *_shapes(2048))

    def test_hidden_dim_rejects_exact_gather_incompatible_alignment(self):
        derive = self.helpers["_derive_inter_local_and_tp"]

        for dim in (64, 128, 384):
            with (
                self.subTest(dim=dim),
                self.assertRaisesRegex(ValueError, "dim aligned to 512"),
            ):
                derive(_cfg(dim=dim, inter=256), *_shapes(64, dim=dim))

    def test_real_hidden_dim_accepts_full_route_alignment(self):
        derive = self.helpers["_derive_inter_local_and_tp"]
        inter_local, routed_tp = derive(_cfg(dim=7168), *_shapes(512, dim=7168))
        self.assertEqual((inter_local, routed_tp), (512, 4))

    def test_compact_nopad_has_no_host_capacity_path(self):
        self.assertNotIn("_select_capacity", self.helpers)
        self.assertIn("compact_mxfp4_routes_nopad", self.source)
        self.assertNotIn(".cpu()", self.source)
        self.assertNotIn(".tolist()", self.source)

    def test_no_count_clamp_or_cuda_grouped_fallback(self):
        calls = [node for node in ast.walk(self.tree) if isinstance(node, ast.Call)]
        called_attributes = {
            node.func.attr for node in calls if isinstance(node.func, ast.Attribute)
        }
        referenced_names = {
            node.id for node in ast.walk(self.tree) if isinstance(node, ast.Name)
        }
        self.assertNotIn("clamp", called_attributes)
        self.assertNotIn("safe_counts", self.source)
        self.assertNotIn("GroupedFP4Strategy", referenced_names)
        self.assertNotIn("LocalLoopStrategy", self.source)
        self.assertNotIn("from .grouped_fp4", self.source)

    def test_quant_device_and_symbol_contracts_fail_closed(self):
        required_fragments = (
            "packed int8/uint8 MXFP4 weights",
            "float8_e8m0fnu checkpoint scales",
            "ZW-M890P",
            "m_grouped_gemm_fp4_fp4_bf16_nt_nopad",
            "compact_mxfp4_routes_nopad",
            "self.routed_tp_size = routed_tp_size",
            "ppu_grouped_fp4 grouped GEMM was not bound by setup",
            "gather_local_loop_compatible",
        )
        for fragment in required_fragments:
            self.assertIn(fragment, self.source)
        self.assertNotIn("except Exception", self.source)

    def test_setup_logs_bound_operator_path_once_and_forward_does_not_sync(self):
        strategy = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "PpuGroupedFP4Strategy"
        )
        setup = next(
            node
            for node in strategy.body
            if isinstance(node, ast.FunctionDef) and node.name == "setup_weights"
        )
        forward = next(
            node
            for node in strategy.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        log_once = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_log_operator_path_once"
        )

        self.assertEqual(len(log_once.decorator_list), 1)
        self.assertEqual(
            ast.unparse(log_once.decorator_list[0]), "lru_cache(maxsize=1)"
        )
        log_source = ast.unparse(log_once)
        for field in (
            "DSV4_PPU_GROUPED_OPERATOR_PATH",
            "strategy=ppu_grouped_fp4",
            "operator_module=%s",
            "activation_mode=%s",
            "exact_gather=%s",
        ):
            self.assertIn(field, log_source)

        setup_source = ast.unparse(setup)
        self.assertIn("self._grouped_gemm = grouped_gemm", setup_source)
        self.assertNotIn("_fused_swiglu_quant", setup_source)
        self.assertNotIn("rtp_llm_ops", setup_source)
        self.assertIn("_log_operator_path_once", setup_source)

        forward_source = ast.unparse(forward)
        self.assertNotIn("_log_operator_path_once", forward_source)
        self.assertNotIn("import deep_gemm", forward_source)
        # The opt-in SG activation consumes the already-loaded compute_ops
        # binding; importing that module does not synchronize the device.
        self.assertIn("ppu_silu_and_mul_post_quant_mxfp4", forward_source)
        for host_sync in (".cpu()", ".tolist()", ".item()", "cuda.synchronize"):
            self.assertNotIn(host_sync, forward_source)

    def test_forward_uses_local_compatible_grouped_chain_without_fallback(self) -> None:
        tree = ast.parse(self.source)
        strategy = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "PpuGroupedFP4Strategy"
        )
        forward = next(
            node
            for node in strategy.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        calls = [node for node in ast.walk(forward) if isinstance(node, ast.Call)]

        split_require_calls = [
            node
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "require_silu_mul_split"
        ]
        self.assertEqual(len(split_require_calls), 1)
        forward_source = ast.unparse(forward)
        self.assertIn(
            "gate_up[:, :self.inter_local].float().contiguous()", forward_source
        )
        self.assertIn(
            "gate_up[:, self.inter_local:].float().contiguous()", forward_source
        )
        self.assertIn(".to(torch.bfloat16).contiguous()", forward_source)
        self.assertNotIn("_fused_swiglu_quant", forward_source)

        downcast_calls = [
            node
            for node in calls
            if isinstance(node.func, ast.Name) and node.func.id == "downcast_to_mxfp4"
        ]
        self.assertEqual(len(downcast_calls), 2)
        self.assertEqual(ast.unparse(downcast_calls[0].args[0]), "x.contiguous()")
        self.assertEqual(ast.unparse(downcast_calls[1].args[0]), "hidden")

        grouped_calls = [
            node
            for node in calls
            if isinstance(node.func, ast.Attribute)
            and node.func.attr == "_grouped_gemm"
        ]
        self.assertEqual(len(grouped_calls), 2)
        self.assertIn("self._ppu_w13", ast.unparse(grouped_calls[0]))
        self.assertIn("self._ppu_w2", ast.unparse(grouped_calls[1]))
        self.assertIn("(hidden_fp4, hidden_scale)", ast.unparse(grouped_calls[1]))

        exact_gather_calls = [
            node
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "gather_local_loop_compatible"
        ]
        self.assertEqual(len(exact_gather_calls), 1)
        self.assertEqual(
            [ast.unparse(arg) for arg in exact_gather_calls[0].args],
            [
                "down",
                "adjusted_ids",
                "weights.contiguous()",
                "output_index",
                "gathered",
            ],
        )
        self.assertNotIn("ep_gather", forward_source)
        self.assertNotIn("LocalLoopStrategy", forward_source)
        self.assertNotIn("except", forward_source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
