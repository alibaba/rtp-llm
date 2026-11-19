"""CPU-only routing contract tests; GPU correctness still needs TP2 replay."""
import ast
import pathlib
import types
import unittest


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
