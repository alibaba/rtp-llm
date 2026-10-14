import csv
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
                "((Path(p) / 'nvidia_cutlass_dsl').is_dir() or "
                "(Path(p) / 'cutlass').is_dir())]",
                f"sys.path[:0] = {[str(site) for site in sites]!r}",
                f"spec = importlib.util.spec_from_file_location('helper', {str(helper)!r})",
                "helper = importlib.util.module_from_spec(spec)",
                "spec.loader.exec_module(helper)",
                script,
            ]
        )
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_prefers_build_install_to_partial_pip_repositories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = self._wheel(
                root / "raw",
                "nvidia-cutlass-dsl-libs-core",
                {"nvidia_cutlass_dsl/dsl_packages/cutlass/__init__.py": "VALUE = 0\n"},
            )
            complete = self._wheel(
                root / "build",
                "nvidia-cutlass-dsl",
                {
                    "nvidia_cutlass_dsl/dsl_packages/cutlass/__init__.py": "from ._mlir.loader import VALUE\n",
                    "nvidia_cutlass_dsl/dsl_packages/cutlass/_mlir/loader.py": "from pathlib import Path\n"
                    "VALUE = Path(__file__).with_name('runtime').read_text()\n",
                    "nvidia_cutlass_dsl/dsl_packages/cutlass/_mlir/runtime": "complete",
                },
            )
            packages = complete / "nvidia_cutlass_dsl/dsl_packages"
            # Compare paths in the child, where the selected import is observable.
            result = self._run(
                [partial, complete, packages],
                "helper.setup_cutlass_import_path()\n"
                "import cutlass\n"
                f"assert Path(cutlass.__file__).parent == Path({str(packages / 'cutlass')!r})\n"
                "assert cutlass.VALUE == 'complete'\n"
                "helper.setup_cutlass_import_path()\n",
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_wrong_version_at_direct_import_path(self):
        with tempfile.TemporaryDirectory() as directory:
            site = self._wheel(
                Path(directory),
                "nvidia-cutlass-dsl",
                {
                    "nvidia_cutlass_dsl/dsl_packages/cutlass/__init__.py": "__version__ = '4.4.2'\n"
                },
            )
            result = self._run(
                [site, site / "nvidia_cutlass_dsl/dsl_packages"],
                "helper.setup_cutlass_import_path()",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("differs from installed 4.7.1", result.stderr)

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
                    "assert cutlass.VALUE == 42\n",
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
                        "assert cutlass.VALUE == 42\n",
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)

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
