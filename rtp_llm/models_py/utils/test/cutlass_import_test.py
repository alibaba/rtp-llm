import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CutlassImportTest(unittest.TestCase):
    def _wheel(self, root, name, files, record=True, version="4.7.1"):
        site = root / name
        for relative, content in files.items():
            path = site / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        info = site / (name.replace("-", "_") + f"-{version}.dist-info")
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
        if record:
            with (info / "RECORD").open("w") as manifest:
                writer = csv.writer(manifest)
                for path in files:
                    writer.writerow((path, "", ""))
        # Reproduce rules_python's namespace shims, absent from wheel RECORD.
        for path in site.rglob("*"):
            if path.is_dir() and "dist-info" not in str(path):
                init = path / "__init__.py"
                if not init.exists():
                    init.write_text(
                        "# __path__ manipulation added by bazelbuild/rules_python "
                        "to support namespace pkgs.\n"
                        "__path__ = __import__('pkgutil').extend_path(__path__, __name__)\n"
                    )
        return site

    def _run(self, sites, script):
        helper = Path(__file__).parent.parent / "cutlass.py"
        code = "\n".join(
            [
                "import importlib.util, sys",
                "from pathlib import Path",
                # Isolate fixture wheels from the test runner's installed DSL.
                "sys.path[:] = [p for p in sys.path if not "
                "(Path(p) / 'nvidia_cutlass_dsl').is_dir()]",
                f"sys.path[:0] = {[str(site) for site in sites]!r}",
                f"spec = importlib.util.spec_from_file_location('helper', {str(helper)!r})",
                "helper = importlib.util.module_from_spec(spec)",
                "spec.loader.exec_module(helper)",
                script,
            ]
        )
        environment = os.environ.copy()
        environment.pop("KIMI_K3_CUTLASS_DSL_ROOT", None)
        return subprocess.run(
            [sys.executable, "-c", code],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def _check_split_wheels(self, record, legacy=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core = self._wheel(
                root,
                "nvidia-cutlass-dsl-libs-core",
                {
                    "nvidia_cutlass_dsl/dsl_packages/cutlass/__init__.py": "from ._mlir.loader import FLAVOR\n",
                },
                record=record,
            )
            base = self._wheel(
                root,
                "nvidia-cutlass-dsl-libs-base",
                {
                    "nvidia_cutlass_dsl/dsl_packages/cutlass/_mlir/loader.py": "from pathlib import Path\n"
                    "FLAVOR = Path(__file__).with_name('native.cu13').read_text()\n",
                },
                record=record,
            )
            cuda = self._wheel(
                root,
                "nvidia-cutlass-dsl-libs-cu13",
                {"nvidia_cutlass_dsl/dsl_packages/cutlass/_mlir/native.cu13": "cu13"},
                record=record,
            )
            shim = self._wheel(root, "nvidia-cutlass-dsl", {})
            older = []
            if legacy:
                older.append(
                    self._wheel(
                        root / "legacy",
                        "nvidia-cutlass-dsl",
                        {
                            "nvidia_cutlass_dsl/python_packages/cutlass/__init__.py": "FLAVOR = 'old'\n",
                            "nvidia_cutlass_dsl/lib/libcute_dsl_runtime.so": "old runtime",
                        },
                        version="4.4.2",
                    )
                )
            for sites in ((core, base, cuda), (base, cuda, core)):
                with self.subTest(order=sites):
                    result = self._run(
                        [*sites, shim, *older],
                        "helper.setup_cutlass_import_path()\n"
                        "import cutlass\n"
                        "first = cutlass.__file__\n"
                        "helper.setup_cutlass_import_path()\n"
                        "assert cutlass.__file__ == first\n"
                        "assert cutlass.FLAVOR == 'cu13'\n",
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    if legacy:
                        result = self._run(
                            [*sites, shim, *older],
                            "helper.setup_cutlass_import_path()\n"
                            "assert not (Path(helper._install_view.name) / "
                            "'nvidia_cutlass_dsl/lib/libcute_dsl_runtime.so').exists()\n",
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)

    def test_split_wheels_join_native_loader_siblings(self):
        self._check_split_wheels(record=True)

    def test_split_runfiles_without_record_ignore_generated_namespace_init(self):
        self._check_split_wheels(record=False)

    def test_split_wheels_exclude_older_installed_runtime_libraries(self):
        self._check_split_wheels(record=False, legacy=True)

    def test_single_wheel_keeps_both_supported_layouts(self):
        for layout in ("dsl_packages", "python_packages"):
            with self.subTest(
                layout=layout
            ), tempfile.TemporaryDirectory() as directory:
                site = self._wheel(
                    Path(directory),
                    "nvidia-cutlass-dsl",
                    {
                        f"nvidia_cutlass_dsl/{layout}/cutlass/__init__.py": "VALUE = 42\n"
                    },
                )
                result = self._run(
                    [site],
                    "helper.setup_cutlass_import_path()\n"
                    "import cutlass\n"
                    "assert cutlass.VALUE == 42\n"
                    "assert helper._install_view is None\n",
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_single_install_deduplicates_path_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            site = self._wheel(
                Path(directory),
                "nvidia-cutlass-dsl",
                {"nvidia_cutlass_dsl/dsl_packages/cutlass/__init__.py": "VALUE = 42\n"},
            )
            (site / "rtp_llm").mkdir()
            symlink = Path(directory) / "alias"
            symlink.symlink_to(site, target_is_directory=True)
            for alias in (site / "rtp_llm" / "..", symlink):
                with self.subTest(alias=alias):
                    result = self._run(
                        [site, alias],
                        "helper.setup_cutlass_import_path()\n"
                        "import cutlass\n"
                        "assert cutlass.VALUE == 42\n"
                        "assert helper._install_view is None\n",
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_identical_split_payloads_are_complete(self):
        for record in (True, False):
            with self.subTest(
                record=record
            ), tempfile.TemporaryDirectory() as directory:
                sites = [
                    self._wheel(
                        Path(directory),
                        name,
                        {
                            "nvidia_cutlass_dsl/dsl_packages/cutlass/__init__.py": "VALUE = 42\n"
                        },
                        record=record,
                    )
                    for name in (
                        "nvidia-cutlass-dsl-libs-core",
                        "nvidia-cutlass-dsl-libs-base",
                    )
                ]
                result = self._run(
                    sites,
                    "helper.setup_cutlass_import_path()\n"
                    "import cutlass\n"
                    "assert cutlass.VALUE == 42\n",
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_conflicting_payloads_fail_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            sites = [
                self._wheel(
                    Path(directory),
                    name,
                    {"nvidia_cutlass_dsl/dsl_packages/cutlass/__init__.py": content},
                )
                for name, content in (
                    ("nvidia-cutlass-dsl-libs-core", "VALUE = 1\n"),
                    ("nvidia-cutlass-dsl-libs-base", "VALUE = 2\n"),
                )
            ]
            result = self._run(sites, "helper.setup_cutlass_import_path()")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Conflicting CUTLASS wheel payload", result.stderr)

    def test_default_package_survives_later_dependency_path_insertion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site = self._wheel(
                root,
                "nvidia-cutlass-dsl",
                {"nvidia_cutlass_dsl/dsl_packages/cutlass/__init__.py": "VALUE = 42\n"},
            )
            rival = root / "vendored" / "cutlass"
            rival.mkdir(parents=True)
            (rival / "__init__.py").write_text("VALUE = 0\n")
            result = self._run(
                [site],
                "helper.setup_cutlass_import_path()\n"
                f"sys.path.insert(0, {str(rival.parent)!r})\n"
                "import cutlass\n"
                "assert cutlass.VALUE == 42\n",
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
