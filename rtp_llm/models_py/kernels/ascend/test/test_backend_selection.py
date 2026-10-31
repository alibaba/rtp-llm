import ast
import unittest
from pathlib import Path

ASCEND_DIR = Path(__file__).resolve().parents[1]
MODELS_PY_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = MODELS_PY_DIR / "model_desc" / "qwen3_next.py"


def _imported_modules(nodes):
    modules = []
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, ast.ImportFrom):
                modules.append(child.module or "")
            elif isinstance(child, ast.Import):
                modules.extend(alias.name for alias in child.names)
    return modules


class TestBackendSelectionWithoutRuntimeDependencies(unittest.TestCase):
    def test_shared_model_selects_ascend_or_existing_triton_backend(self):
        tree = ast.parse(
            MODEL_PATH.read_text(encoding="utf-8"), filename=str(MODEL_PATH)
        )
        backend_branches = [
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and "get_device_type() == DeviceType.Ascend" in ast.unparse(node.test)
        ]
        self.assertEqual(len(backend_branches), 1)

        branch = backend_branches[0]
        ascend_imports = _imported_modules(branch.body)
        default_imports = _imported_modules(branch.orelse)
        self.assertIn("rtp_llm.models_py.kernels.ascend", ascend_imports)
        self.assertFalse(any("triton_kernels" in name for name in ascend_imports))
        self.assertTrue(any("triton_kernels" in name for name in default_imports))
        self.assertFalse(any("kernels.ascend" in name for name in default_imports))

    def test_ascend_modules_do_not_dispatch_back_to_other_backends(self):
        for module_path in ASCEND_DIR.glob("*.py"):
            with self.subTest(module=module_path.name):
                source = module_path.read_text(encoding="utf-8")
                self.assertNotIn("_is_npu_", source)
                self.assertNotIn("rtp_llm.models_py.triton_kernels", source)


if __name__ == "__main__":
    unittest.main()
