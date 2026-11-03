"""CPU-only contract tests for TP-local DeepSeek-V4 prefill-Q sizing."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

_PATH = Path(__file__).resolve().parents[1] / "models/dsv4/impl/prefill_workspace.py"
_SPEC = importlib.util.spec_from_file_location("dsv4_prefill_workspace_under_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
tp_local_prefill_q_dim = _MODULE.tp_local_prefill_q_dim


class TpPrefillWorkspaceDimsTest(unittest.TestCase):
    def test_cuda_retains_original_buffer_capacities(self):
        path = Path(__file__).resolve().parents[3] / "models_py/model_desc/deepseek_v4_model.py"
        tree = ast.parse(path.read_text())
        names = {"_resolve_prefill_q_token_capacity",
                 "_resolve_mtp_hidden_token_capacity"}
        methods = [n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name in names]
        scope = {"tp_local_prefill_q_dim": tp_local_prefill_q_dim}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), scope)
        model = SimpleNamespace(
            _platform_provider=SimpleNamespace(),
            _v4_args=SimpleNamespace(n_heads=128, head_dim=256, tp_size=4),
            _resolve_shared_token_capacity=lambda: 777,
        )
        self.assertEqual(scope["_resolve_prefill_q_token_capacity"](model), 777)
        self.assertEqual(scope["_resolve_mtp_hidden_token_capacity"](model), 777)

    def test_prefill_q_dim_is_tp_local(self):
        self.assertEqual(tp_local_prefill_q_dim(128, 256, 1), 32768)
        self.assertEqual(tp_local_prefill_q_dim(128, 256, 8), 4096)

    def test_prefill_q_dim_rejects_invalid_head_partition(self):
        with self.assertRaisesRegex(AssertionError, "not divisible"):
            tp_local_prefill_q_dim(10, 256, 8)


if __name__ == "__main__":
    unittest.main()
