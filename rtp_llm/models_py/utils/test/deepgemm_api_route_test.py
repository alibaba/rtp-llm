"""CPU routing checks for the CUDA wrapper's new and original wheel APIs.

Execute the production resolver and selected wrapper definitions without loading
torch, Triton, or a native wheel. Backend doubles represent exported names only;
these checks do not establish CUDA12/PPU native ABI or numerical compatibility.
"""

import __future__

import ast
import contextlib
import io
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

RTP_ROOT = Path(__file__).resolve().parents[3]
WRAPPER_PATH = RTP_ROOT / "models_py/kernels/cuda/deepgemm_wrapper.py"
MODULE_UTIL_PATH = RTP_ROOT / "utils/module_util.py"
SYMBOL = "fp8_fp4_paged_mqa_logits"
NEW_SYMBOL = "fp8_fp4_paged_mqa_logits_main_compat"
IMPL = "_fp8_fp4_paged_mqa_logits_impl"


def _load_real_module_util():
    module = ModuleType("deepgemm_route_module_util")
    # Postpone annotations so this stdlib-only test also runs on Python 3.9.
    # The complete production module executes unchanged, including resolve_symbol.
    code = compile(
        MODULE_UTIL_PATH.read_text(),
        str(MODULE_UTIL_PATH),
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    exec(code, module.__dict__)
    return module


MODULE_UTIL = _load_real_module_util()


def _load_wrapper():
    tree = ast.parse(WRAPPER_PATH.read_text(), filename=str(WRAPPER_PATH))
    names = {
        "_deep_gemm_impl_new_map",
        "_deep_gemm_impl_old_map",
        IMPL,
        "_lazy_init_deep_gemm",
        "_missing_deep_gemm",
        SYMBOL,
    }
    nodes = []
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            node_names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            node_names = {node.target.id}
        elif isinstance(node, ast.FunctionDef):
            node_names = {node.name}
        else:
            continue
        if node_names & names:
            nodes.append(node)
            found.update(node_names & names)
    if found != names:
        raise AssertionError(f"Production wrapper definitions missing: {names - found}")
    namespace = {
        "__name__": "deepgemm_route_wrapper",
        "resolve_symbol": MODULE_UTIL.resolve_symbol,
        "has_deep_gemm": lambda: True,
        # Only evaluate the public function's torch.float32 default. No tensor
        # operation or torch import is part of this routing test.
        "torch": SimpleNamespace(float32=object()),
    }
    code = compile(
        ast.Module(body=nodes, type_ignores=[]),
        str(WRAPPER_PATH),
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    exec(code, namespace)
    return namespace


class DeepGemmApiRouteTest(unittest.TestCase):
    def setUp(self):
        self.wrapper = _load_wrapper()
        self.arguments = tuple(object() for _ in range(9))

    def initialize(self, backend):
        with patch.dict("sys.modules", {"deep_gemm": backend}):
            # The real resolver prints an upgrade hint for an old-only wheel.
            with contextlib.redirect_stdout(io.StringIO()):
                self.wrapper["_lazy_init_deep_gemm"]([SYMBOL])

    def invoke(self):
        return self.wrapper[SYMBOL](*self.arguments)

    def test_actual_maps_keep_explicit_main_and_original_names(self):
        self.assertEqual(self.wrapper["_deep_gemm_impl_new_map"][SYMBOL], NEW_SYMBOL)
        self.assertEqual(self.wrapper["_deep_gemm_impl_old_map"][SYMBOL], SYMBOL)
        self.assertIs(self.wrapper["resolve_symbol"], MODULE_UTIL.resolve_symbol)

    def test_new_wheel_selects_main_compat_over_original_v2_name(self):
        main_result, v2_result = object(), object()
        main = Mock(return_value=main_result)
        original = Mock(return_value=v2_result)
        self.initialize(SimpleNamespace(**{NEW_SYMBOL: main, SYMBOL: original}))
        self.assertIs(self.wrapper[IMPL], main)
        self.assertIs(self.invoke(), main_result)
        main.assert_called_once_with(*self.arguments)
        original.assert_not_called()

    def test_new_wheel_does_not_require_original_name(self):
        result = object()
        main = Mock(return_value=result)
        self.initialize(SimpleNamespace(**{NEW_SYMBOL: main}))
        self.assertIs(self.wrapper[IMPL], main)
        self.assertIs(self.invoke(), result)
        main.assert_called_once_with(*self.arguments)

    def test_old_cuda12_and_ppu_export_shape_selects_original_name(self):
        for backend_name in ("CUDA12", "PPU"):
            with self.subTest(backend=backend_name):
                self.wrapper = _load_wrapper()
                result = object()
                original = Mock(return_value=result)
                self.initialize(SimpleNamespace(**{SYMBOL: original}))
                self.assertIs(self.wrapper[IMPL], original)
                self.assertIs(self.invoke(), result)
                original.assert_called_once_with(*self.arguments)

    def test_missing_both_symbols_raises_when_public_wrapper_is_called(self):
        self.initialize(SimpleNamespace())
        # resolve_symbol intentionally returns None for two absent names. The
        # actual public wrapper turns this into the normal availability error.
        self.assertIsNone(self.wrapper[IMPL])
        with self.assertRaisesRegex(RuntimeError, "DeepGEMM is not available"):
            self.invoke()

    def test_symbol_lookup_internal_errors_do_not_select_old_backend(self):
        for error in (
            ImportError("native dependency missing"),
            ModuleNotFoundError("native dependency missing", name="deep_gemm_cuda"),
            RuntimeError("backend initialization failed"),
        ):
            with self.subTest(error=type(error).__name__):
                self.wrapper = _load_wrapper()
                original = Mock()
                old_lookups = []

                class BrokenBackend(ModuleType):
                    def __getattr__(self, name):
                        if name == NEW_SYMBOL:
                            raise error
                        if name == SYMBOL:
                            old_lookups.append(name)
                            return original
                        raise AttributeError(name)

                with self.assertRaises(type(error)) as raised:
                    self.initialize(BrokenBackend("deep_gemm"))
                self.assertIs(raised.exception, error)
                self.assertIsNone(self.wrapper[IMPL])
                self.assertEqual(old_lookups, [])
                original.assert_not_called()

    def test_selected_callable_internal_errors_do_not_retry_original(self):
        for error in (
            TypeError("native API argument error"),
            ValueError("native shape error"),
            RuntimeError("native launch error"),
        ):
            with self.subTest(error=type(error).__name__):
                self.wrapper = _load_wrapper()
                main = Mock(side_effect=error)
                original = Mock()
                self.initialize(SimpleNamespace(**{NEW_SYMBOL: main, SYMBOL: original}))
                with self.assertRaises(type(error)) as raised:
                    self.invoke()
                self.assertIs(raised.exception, error)
                main.assert_called_once_with(*self.arguments)
                original.assert_not_called()
                self.assertIs(self.wrapper[IMPL], main)

    def test_backend_import_error_is_propagated(self):
        error = ModuleNotFoundError("native dependency missing", name="deep_gemm_cuda")
        import_module = __import__

        def fail_backend_import(name, *args, **kwargs):
            if name == "deep_gemm":
                raise error
            return import_module(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fail_backend_import):
            with self.assertRaises(ModuleNotFoundError) as raised:
                self.wrapper["_lazy_init_deep_gemm"]([SYMBOL])
        self.assertIs(raised.exception, error)
        self.assertIsNone(self.wrapper[IMPL])


if __name__ == "__main__":
    unittest.main()
