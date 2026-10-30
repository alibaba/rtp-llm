"""CPU checks for explicit legacy/current routing with both wheels installed."""

import ast
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "flash_mla_legacy_route", ROOT / "utils/flash_mla_legacy.py"
)
LEGACY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LEGACY)
spec = importlib.util.spec_from_file_location(
    "flash_mla_current_route", ROOT / "utils/flash_mla_current.py"
)
CURRENT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CURRENT)


class FlashMlaRouteTest(unittest.TestCase):
    def test_explicit_current_wins_over_new_legacy_default(self):
        current = object()
        with patch.object(CURRENT, "import_module", return_value=current) as load:
            self.assertIs(CURRENT.get_current_flash_mla(), current)
        load.assert_called_once_with("flash_mla.current")

    def test_missing_current_namespace_keeps_old_top_level(self):
        # CUDA12 fcaae18 and older bdd wheels have no DEFAULT_API marker.
        old = SimpleNamespace(
            flash_mla_with_kvcache=Mock(),
            _build_info=SimpleNamespace(BACKEND="cuda"),
        )
        with patch.object(
            CURRENT,
            "import_module",
            side_effect=[ModuleNotFoundError(name="flash_mla.current"), old],
        ) as load:
            self.assertIs(CURRENT.get_current_flash_mla(), old)
        self.assertEqual(
            [call.args[0] for call in load.call_args_list],
            ["flash_mla.current", "flash_mla"],
        )

    def test_missing_current_in_marked_new_wheel_is_rejected(self):
        for default in ("legacy", "current"):
            for backend in (
                SimpleNamespace(DEFAULT_API=default),
                SimpleNamespace(_build_info=SimpleNamespace(DEFAULT_API=default)),
            ):
                with self.subTest(default=default), patch.object(
                    CURRENT,
                    "import_module",
                    side_effect=[
                        ModuleNotFoundError(name="flash_mla.current"),
                        backend,
                    ],
                ):
                    with self.assertRaisesRegex(ImportError, "lacks flash_mla.current"):
                        CURRENT.get_current_flash_mla()

    def test_current_internal_errors_do_not_select_top_level(self):
        for error in (
            ModuleNotFoundError(name="flash_mla"),
            ModuleNotFoundError(name="flash_mla_cuda"),
            ImportError("native symbol missing"),
        ):
            with self.subTest(error=error), patch.object(
                CURRENT, "import_module", side_effect=error
            ) as load:
                with self.assertRaises(type(error)) as raised:
                    CURRENT.get_current_flash_mla()
                self.assertIs(raised.exception, error)
            load.assert_called_once_with("flash_mla.current")

    def test_old_top_level_internal_error_is_propagated(self):
        error = ImportError("native symbol missing")
        with patch.object(
            CURRENT,
            "import_module",
            side_effect=[ModuleNotFoundError(name="flash_mla.current"), error],
        ):
            with self.assertRaises(ImportError) as raised:
                CURRENT.get_current_flash_mla()
            self.assertIs(raised.exception, error)

    def test_independent_legacy_wins_when_both_packages_exist(self):
        old, integrated = object(), object()
        with patch.object(
            LEGACY,
            "import_module",
            side_effect=lambda name: {
                "flash_mla_legacy": old,
                "flash_mla.legacy": integrated,
            }[name],
        ) as load:
            self.assertIs(LEGACY.get_legacy_flash_mla(), old)
        load.assert_called_once_with("flash_mla_legacy")

    def test_missing_independent_package_uses_integrated_legacy(self):
        integrated = object()
        missing = ModuleNotFoundError(name="flash_mla_legacy")
        with patch.object(
            LEGACY, "import_module", side_effect=[missing, integrated]
        ) as load:
            self.assertIs(LEGACY.get_legacy_flash_mla(), integrated)
        self.assertEqual(
            [call.args[0] for call in load.call_args_list],
            ["flash_mla_legacy", "flash_mla.legacy"],
        )

    def test_broken_independent_package_does_not_change_backend(self):
        for error in (
            ModuleNotFoundError(name="flash_mla_legacy_cuda"),
            ImportError("native symbol missing"),
        ):
            with self.subTest(error=error), patch.object(
                LEGACY, "import_module", side_effect=error
            ) as load:
                with self.assertRaises(type(error)) as raised:
                    LEGACY.get_legacy_flash_mla()
                self.assertIs(raised.exception, error)
            load.assert_called_once_with("flash_mla_legacy")

    @staticmethod
    def _missing_namespaces(backend):
        return [
            ModuleNotFoundError(name="flash_mla_legacy"),
            ModuleNotFoundError(name="flash_mla.legacy"),
            backend,
        ]

    def test_old_cuda12_top_level_native_capability_is_preserved(self):
        backend = SimpleNamespace(cuda=SimpleNamespace(dense_decode_fwd=Mock()))
        with patch.object(
            LEGACY, "import_module", side_effect=self._missing_namespaces(backend)
        ) as load:
            self.assertIs(LEGACY.get_legacy_flash_mla(), backend)
        self.assertEqual(load.call_args.args, ("flash_mla",))

    def test_unmarked_top_level_without_legacy_native_is_rejected(self):
        for native in (None, SimpleNamespace(dense_decode_fwd=None)):
            with self.subTest(native=native), patch.object(
                LEGACY,
                "import_module",
                side_effect=self._missing_namespaces(SimpleNamespace(cuda=native)),
            ):
                with self.assertRaisesRegex(ImportError, "ambiguous/current"):
                    LEGACY.get_legacy_flash_mla()

    def test_old_ppu_top_level_native_capability_is_preserved(self):
        backend = SimpleNamespace(
            flash_mla_interface=SimpleNamespace(
                flash_mla_cuda=SimpleNamespace(fwd_kvcache_mla=Mock())
            )
        )
        with patch.object(
            LEGACY, "import_module", side_effect=self._missing_namespaces(backend)
        ):
            self.assertIs(LEGACY.get_legacy_flash_mla(), backend)

    def test_ppu_marker_does_not_override_explicit_current_default(self):
        backend = SimpleNamespace(
            DEFAULT_API="current",
            flash_mla_interface=SimpleNamespace(
                flash_mla_cuda=SimpleNamespace(fwd_kvcache_mla=Mock())
            ),
        )
        with patch.object(
            LEGACY, "import_module", side_effect=self._missing_namespaces(backend)
        ):
            with self.assertRaisesRegex(ImportError, "ambiguous/current"):
                LEGACY.get_legacy_flash_mla()

    def test_explicit_current_default_cannot_pass_native_probe(self):
        backend = SimpleNamespace(
            DEFAULT_API="current",
            cuda=SimpleNamespace(dense_decode_fwd=Mock()),
        )
        with patch.object(
            LEGACY,
            "import_module",
            side_effect=self._missing_namespaces(backend),
        ):
            with self.assertRaisesRegex(ImportError, "ambiguous/current"):
                LEGACY.get_legacy_flash_mla()

    def test_explicit_legacy_default_is_accepted(self):
        backend = SimpleNamespace(DEFAULT_API="legacy")
        with patch.object(
            LEGACY, "import_module", side_effect=self._missing_namespaces(backend)
        ):
            self.assertIs(LEGACY.get_legacy_flash_mla(), backend)

    def test_integrated_package_internal_import_error_does_not_fall_back(self):
        broken = ModuleNotFoundError(name="flash_mla_legacy_cuda")
        with patch.object(
            LEGACY,
            "import_module",
            side_effect=[ModuleNotFoundError(name="flash_mla_legacy"), broken],
        ) as load:
            with self.assertRaises(ModuleNotFoundError) as raised:
                LEGACY.get_legacy_flash_mla()
            self.assertIs(raised.exception, broken)
        self.assertEqual(load.call_count, 2)

    def test_top_level_internal_import_error_is_propagated(self):
        broken = ImportError("native symbol missing")
        with patch.object(
            LEGACY, "import_module", side_effect=self._missing_namespaces(broken)
        ):
            with self.assertRaises(ImportError) as raised:
                LEGACY.get_legacy_flash_mla()
            self.assertIs(raised.exception, broken)

    @staticmethod
    def _bindings(path, owner, names):
        """Execute actual import statements without importing native modules."""
        tree = ast.parse(path.read_text())
        if owner is not None:
            tree = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == owner
            )
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        return [
            node
            for function in functions
            for node in ast.walk(function)
            if isinstance(node, ast.ImportFrom)
            and any(
                alias.name.startswith(("flash_mla", "get_mla")) for alias in node.names
            )
        ]

    def test_page_rr_metadata_and_three_kernels_share_integrated_backend(self):
        path = (
            ROOT
            / "modules/factory/attention/cuda_mla_impl/flashmla_sparse_page_rr_impl.py"
        )
        imports = self._bindings(path, "PageRRSparseMlaOp", {"plan", "_attention"})
        self.assertEqual(len(imports), 4)
        expected = {
            "get_mla_metadata",
            "flash_mla_sparse_fwd",
            "flash_mla_with_kvcache",
            "flash_mla_sparse_decode_fp8_kv",
        }
        native = SimpleNamespace(**{name: Mock() for name in expected})
        scope = {}
        with patch.dict("sys.modules", {"flash_mla.legacy": native}):
            for node in imports:
                self.assertEqual(node.module, "flash_mla.legacy")
                exec(
                    compile(
                        ast.Module(body=[node], type_ignores=[]), str(path), "exec"
                    ),
                    scope,
                )
        for name in expected:
            self.assertIs(scope[name], getattr(native, name))

    def test_v41_uses_current_even_with_independent_legacy_installed(self):
        path = ROOT / "modules/dsv41/flash_mla.py"
        imports = self._bindings(
            path, None, {"flash_mla_sparse_fwd", "flash_mla_with_kvcache"}
        )
        self.assertEqual(len(imports), 2)
        native = SimpleNamespace(
            flash_mla_sparse_fwd=Mock(), flash_mla_with_kvcache=Mock()
        )
        with patch.dict(
            "sys.modules",
            {
                "rtp_llm.models_py.utils.flash_mla_current": CURRENT,
                "flash_mla_legacy": object(),
            },
        ), patch.object(CURRENT, "import_module", return_value=native):
            for node in imports:
                self.assertEqual(
                    node.module, "rtp_llm.models_py.utils.flash_mla_current"
                )
                scope = {}
                exec(
                    compile(
                        ast.Module(body=[node], type_ignores=[]), str(path), "exec"
                    ),
                    scope,
                )
                self.assertIs(scope["native"], getattr(native, node.names[0].name))


if __name__ == "__main__":
    unittest.main()
