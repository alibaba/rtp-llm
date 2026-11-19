"""CPU-only routing contract tests; GPU correctness still needs TP2 replay."""
import ast
import pathlib
import types
import unittest
import os
from unittest.mock import patch


def load_routed_method():
    path = pathlib.Path(__file__).with_name("shared_overlap.py")
    tree = ast.parse(path.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "Qwen4MoeLayer")
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "_forward_routed")
    module = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(
        name="Candidate", bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[], body=[method], decorator_list=[])], type_ignores=[]))

    class Base:
        def _forward_routed(self, x):
            return "baseline"

    namespace = {"Base": Base, "torch": types.SimpleNamespace(bfloat16="bf16"),
                 "logger": types.SimpleNamespace(info=lambda *a: None)}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["Candidate"]


class TboDispatchTest(unittest.TestCase):
    def test_puretp_v2_qualification_without_deepep_tbo(self):
        path = pathlib.Path(__file__).with_name("shared_overlap.py")
        tree = ast.parse(path.read_text())
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "Qwen4MoeLayer")
        method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "_prepare_ep_shared_overlap")
        method.body = [x for x in method.body if not isinstance(x, ast.ImportFrom)]
        module = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name="Candidate", bases=[], keywords=[], body=[method], decorator_list=[])], type_ignores=[]))
        types_by_name = {name: type(name, (), {}) for name in ("Fp8Linear", "FusedMoe", "DeepGemmMaskedExecutor", "DeepGemmMaskedExecutorV2", "DeepEpLowLatencyRouter", "PureTpRouterFp8PerBlock", "DenseMLP")}
        namespace = dict(types_by_name)
        namespace.update({"os": os, "logger": types.SimpleNamespace(info=lambda *a: None, warning=lambda *a: None), "torch": types.SimpleNamespace(version=types.SimpleNamespace(hip=None), cuda=types.SimpleNamespace(get_device_capability=lambda _: (9, 0), Event=object)), "_shared_stream_for_device": lambda _: "stream"})
        exec(compile(module, str(path), "exec"), namespace)
        T = types_by_name
        layer = namespace["Candidate"]()
        layer.parallelism_config = types.SimpleNamespace(tp_size=2, ep_size=2, dp_size=1)
        layer.config = types.SimpleNamespace(eplb_config=types.SimpleNamespace(enable_eplb=lambda: False))
        layer.shared_expert = T["DenseMLP"]()
        def linear(shape):
            obj = T["Fp8Linear"]()
            obj.weight = types.SimpleNamespace(shape=shape, is_cuda=True, device=types.SimpleNamespace(index=0))
            return obj
        layer.shared_expert.up_proj = linear((640, 2560))
        layer.shared_expert.down_proj = linear((2560, 320))
        layer.shared_expert_gate = linear((1, 2560))
        layer.fused_moe = T["FusedMoe"]()
        layer.fused_moe.router = T["PureTpRouterFp8PerBlock"]()
        layer.fused_moe.router.tp_collective_size = 2
        layer.fused_moe.router.do_recompute_topk = True
        layer.fused_moe.fused_experts = T["DeepGemmMaskedExecutorV2"]()
        layer.use_ep_shared_allreduce = True
        layer.use_unified_tp_allreduce = False
        layer._sm90_mega_moe = None
        layer._ep_shared_stream = None
        layer._ep_two_batch = None
        layer.correction_bias = None
        layer.fake_balance_expert = None
        with patch.dict(os.environ, {"QWEN_MOE_EP_SHARED_OVERLAP": "1", "QWEN_MOE_EP_TBO": "1"}):
            layer._prepare_ep_shared_overlap()
        self.assertEqual(layer._ep_shared_stream, "stream")
        self.assertIsNone(layer._ep_two_batch)
        layer._ep_shared_stream = None
        layer.fused_moe.router.do_recompute_topk = False
        with patch.dict(os.environ, {"QWEN_MOE_EP_SHARED_OVERLAP": "1", "QWEN_MOE_EP_TBO": "0"}):
            layer._prepare_ep_shared_overlap()
        self.assertIsNone(layer._ep_shared_stream)
        layer.fused_moe.router.do_recompute_topk = True
        with patch.dict(os.environ, {"QWEN_MOE_EP_SHARED_OVERLAP": "0", "QWEN_MOE_EP_TBO": "0"}):
            layer._prepare_ep_shared_overlap()
        self.assertIsNone(layer._ep_shared_stream)

    def test_independent_tbo_and_shape_fallback(self):
        cls = load_routed_method()
        layer = cls()
        layer._sm90_mega_moe = None
        layer._ep_two_batch = types.SimpleNamespace(
            supports=lambda x: x.shape[0] >= 4 and x.shape[0] % 4 == 0,
            forward=lambda layer, x: "tbo")
        x = types.SimpleNamespace(is_cuda=True, ndim=2, shape=(8, 2560),
                                  dtype="bf16", is_contiguous=lambda: True)
        self.assertEqual(layer._forward_routed(x), "tbo")
        x.shape = (3, 2560)
        self.assertEqual(layer._forward_routed(x), "baseline")
        x.shape = (8, 2560)
        layer._ep_two_batch = None
        self.assertEqual(layer._forward_routed(x), "baseline")

    def test_non_cuda_and_incorrect_dtype_fall_back(self):
        cls = load_routed_method()
        layer = cls()
        layer._sm90_mega_moe = None
        layer._ep_two_batch = types.SimpleNamespace(supports=lambda x: True,
                                                   forward=lambda *a: "tbo")
        x = types.SimpleNamespace(is_cuda=False, ndim=2, shape=(8, 2560),
                                  dtype="bf16", is_contiguous=lambda: True)
        self.assertEqual(layer._forward_routed(x), "baseline")
        x.is_cuda = True
        x.dtype = "fp16"
        self.assertEqual(layer._forward_routed(x), "baseline")


if __name__ == "__main__":
    unittest.main()
