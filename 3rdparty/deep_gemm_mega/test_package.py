"""Validate the prepared wheel source before native compilation."""

import ast
from pathlib import Path
import sys
import unittest


ROOT = Path(sys.argv.pop(1)).resolve()
PACKAGE = ROOT / "deep_gemm_mega"


class PackageContractTest(unittest.TestCase):
    def test_python_imports_are_isolated(self):
        for path in PACKAGE.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    self.assertFalse(any(n.name.split(".")[0] == "deep_gemm" for n in node.names), path)
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    self.assertNotEqual((node.module or "").split(".")[0], "deep_gemm", path)
        self.assertFalse((ROOT / "deep_gemm").exists())
        init = (PACKAGE / "__init__.py").read_text()
        self.assertIn("from . import deep_gemm_mega_cpp", init)
        self.assertLess(init.index("import torch"), init.index("from . import deep_gemm_mega_cpp"))

    def test_package_data_paths_exist(self):
        setup = ast.parse((ROOT / "setup.py").read_text())
        call = next(n for n in ast.walk(setup) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == "setup")
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        self.assertEqual(ast.literal_eval(kwargs["name"]), "deep-gemm-mega")
        data = ast.literal_eval(kwargs["package_data"])
        self.assertEqual(list(data), ["deep_gemm_mega"])
        for pattern in data["deep_gemm_mega"]:
            if pattern.startswith("include/actlize_"):
                continue  # CustomBuildPy installs pinned dependency headers.
            self.assertTrue(any(PACKAGE.glob(pattern)), pattern)
        self.assertTrue((PACKAGE / "configs/dsv4_flash_bm64/ZW-M890P_fp4_megamoe.lut").is_file())
        self.assertIn("deep_gemm_mega.deep_gemm_mega_cpp", (ROOT / "setup.py").read_text())

    def test_runtime_isolation(self):
        compiler = (ROOT / "csrc/jit/compiler.hpp").read_text()
        self.assertIn('".deep_gemm_mega"', compiler)
        self.assertIn('"DG_MEGA_JIT_CACHE_DIR"', compiler)
        self.assertNotIn('"DG_JIT_CACHE_DIR"', compiler)
        lut = (ROOT / "csrc/jit_kernels/heuristics/runtime_lut.hpp").read_text()
        self.assertIn('"RTP_PPU_MEGA_LUT_PATH"', lut)
        activation = (ROOT / "csrc/apis/activation.hpp").read_text()
        self.assertIn('(m, "ActivationQuantMode", py::module_local())', activation)


if __name__ == "__main__":
    unittest.main()
