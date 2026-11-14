"""Check that input quantization can be imported without loading GPU backends."""

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


if __name__ == "__main__":
    unittest.main()
