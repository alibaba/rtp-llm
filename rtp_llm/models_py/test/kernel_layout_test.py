"""Keep tensor backends independent of model assembly and cache ownership."""

import ast
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]


class KernelLayoutTest(unittest.TestCase):
    def test_input_quantization_dispatch_import_keeps_backends_lazy(self):
        # Isolate this helper's import from the application's package startup.
        program = """
import importlib, pathlib, sys, types
root = pathlib.Path(sys.argv[1])
for name, path in [('rtp_llm', root.parent), ('rtp_llm.models_py', root)]:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
module = importlib.import_module('rtp_llm.models_py.kernels.cuda.fp8_input_quant')
assert callable(module._quantize_projection_input)
backends = ('rtp_llm.ops', 'rtp_llm.models_py.kernels.cuda.fp8_kernel', 'triton')
assert not any(name.startswith(backends) for name in sys.modules)
"""
        subprocess.run([sys.executable, "-c", program, str(ROOT)], check=True)

    def test_tensor_backends_do_not_import_model_layers(self):
        families = (
            "kernels/cuda/gdn_sm120",
            "kernels/cuda/qsa",
            "kernels/cuda/qwen4_ple",
            "triton_kernels/gated_residual",
        )
        forbidden = (
            "rtp_llm.models_py.modules",
            "rtp_llm.models_py.model_desc",
        )
        for family in families:
            sources = sorted((ROOT / family).glob("*.py"))
            self.assertTrue(sources, family)
            for path in sources:
                tree = ast.parse(path.read_text(), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        modules = [node.module or ""]
                    elif isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    else:
                        continue
                    for module in modules:
                        with self.subTest(source=str(path), module=module):
                            self.assertFalse(module.startswith(forbidden))

    def test_model_adapters_do_not_define_device_programs(self):
        sources = [
            path
            for path in (ROOT / "modules/qwen4_exp").rglob("*.py")
            if "test" not in path.relative_to(ROOT / "modules/qwen4_exp").parts
        ]
        self.assertTrue(sources)
        for path in sources:
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                for decorator in getattr(node, "decorator_list", ()):
                    target = (
                        decorator.func if isinstance(decorator, ast.Call) else decorator
                    )
                    if isinstance(target, ast.Attribute):
                        with self.subTest(source=str(path), line=node.lineno):
                            self.assertNotIn(target.attr, ("jit", "kernel"))


if __name__ == "__main__":
    unittest.main()
