"""CPU-only contract tests for TP-local DeepSeek-V4 prefill-Q sizing."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

_PATH = Path(__file__).resolve().parents[6] / "rtp_llm/models_py/model_desc/deepseek_v4_ppu/model.py"
_helper = next(n for n in ast.parse(_PATH.read_text()).body
               if isinstance(n, ast.FunctionDef) and n.name == "tp_local_prefill_q_dim")
_scope = {}
exec(compile(ast.Module(body=[_helper], type_ignores=[]), str(_PATH), "exec"), _scope)
tp_local_prefill_q_dim = _scope["tp_local_prefill_q_dim"]


class TpPrefillWorkspaceDimsTest(unittest.TestCase):
    def test_oversized_prefill_hidden_owns_storage_and_restores_graph_buffer(self):
        import torch

        path = Path(__file__).resolve().parents[6] / "rtp_llm/models_py/modules/dsv4/ppu/prefill/forward.py"
        method = next(n for n in ast.parse(path.read_text()).body
                      if isinstance(n, ast.FunctionDef) and n.name == "forward_prefill")
        start = next(i for i, n in enumerate(method.body)
                     if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                     and n.targets[0].id == "fixed_mtp_buffer")
        method.body = method.body[start:]
        method.args = ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
        method.returns = None
        fixed = torch.full((2, 4), -1.0)
        model = SimpleNamespace(_mtp_hidden_buffer=fixed, _mtp_hidden_valid_tokens=1)
        scope = dict(torch=torch, v4=model, input_ids=torch.zeros(5, dtype=torch.long),
                     PyModelOutputs=lambda hidden, mtp=None: (hidden, mtp))
        scope.update(dict.fromkeys(("kv_cache", "positions", "cu_seqlens", "block_tables_by_type",
                                    "attn", "attn_inputs", "prepare_hidden_fn", "capture_context")))

        def write(*args, **kwargs):
            model._mtp_hidden_buffer.fill_(7)
            model._mtp_hidden_valid_tokens = scope["input_ids"].numel()
            return torch.ones(1)

        scope["forward_layers"] = write
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
        _, output = scope["forward_prefill"]()
        self.assertEqual(tuple(output.shape), (5, 4))
        self.assertIs(model._mtp_hidden_buffer, fixed)
        self.assertEqual(model._mtp_hidden_valid_tokens, 1)
        self.assertTrue(torch.all(fixed == -1))
        scope["input_ids"] = torch.zeros(2, dtype=torch.long)
        self.assertIsNone(scope["forward_prefill"]()[1])
        fixed.zero_()
        self.assertTrue(torch.all(output == 7))

        def fail(*args, **kwargs):
            raise RuntimeError("forward failed")

        scope["input_ids"] = torch.zeros(5, dtype=torch.long)
        scope["forward_layers"] = fail
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            scope["forward_prefill"]()
        self.assertIs(model._mtp_hidden_buffer, fixed)
        self.assertEqual(model._mtp_hidden_valid_tokens, 2)

    def test_workspace_covers_scheduler_bypass_and_fanout(self):
        path = Path(__file__).resolve().parents[6] / "rtp_llm/models_py/modules/dsv4/ppu/prefill/forward.py"
        method = next(n for n in ast.parse(path.read_text()).body
                      if isinstance(n, ast.FunctionDef) and n.name == "forward_layers")
        allocation = next(n for n in method.body if isinstance(n, ast.If))
        code = compile(ast.Module(body=[allocation], type_ignores=[]), str(path), "exec")
        for configured, actual, cp_size in (
            (4096, 128, 1),       # ordinary small batch retains the allocator floor
            (4096, 32768, 1),     # unchunked system prompt
            (32768, 65536, 1),    # singleton with two return sequences
            (4096, 8192, 4),      # CP gather must grow with local Q
        ):
            with self.subTest(configured=configured, actual=actual, cp_size=cp_size):
                scope = {
                    "v4": SimpleNamespace(
                        fp8_kv_cache=True, _cp_info=object() if cp_size > 1 else None,
                        _cp_size=cp_size, _prefill_ws_q_rows=configured,
                        _prefill_ws_q_dim=4096,
                        _prefill_ws_full_rows=configured * cp_size if cp_size > 1 else 0,
                        _prefill_ws_main_w=512, _prefill_ws_idx_w=256,
                    ),
                    "input_ids": SimpleNamespace(device="cpu", size=lambda axis: actual),
                    "PrefillWorkspace": lambda device, **kwargs: SimpleNamespace(**kwargs),
                }
                exec(code, scope)
                workspace = scope["ws"]
                self.assertGreaterEqual(workspace.q_rows, actual)
                self.assertEqual(workspace.q_rows, max(configured, actual))
                self.assertEqual(workspace.cp_rows,
                                 max(configured, actual) * cp_size if cp_size > 1 else 0)

    def test_cuda_retains_original_buffer_capacities(self):
        path = Path(__file__).resolve().parents[5] / "models_py/model_desc/deepseek_v4_model.py"
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
