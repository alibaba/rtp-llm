import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INIT_PATH = REPO_ROOT / "rtp_llm" / "__init__.py"
PKG_DIR = REPO_ROOT / "rtp_llm"


def _run_bootstrap_probe(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )


def test_bootstrap_error_is_suppressed_only_during_pytest_plugin_discovery():
    code = textwrap.dedent(
        f"""
        import importlib.abc
        import importlib.machinery
        import importlib.util
        import sys
        import types
        import warnings

        init_path = {str(INIT_PATH)!r}
        pkg_dir = {str(PKG_DIR)!r}

        spec = importlib.util.spec_from_file_location(
            "rtp_llm", init_path, submodule_search_locations=[pkg_dir]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["rtp_llm"] = module

        utils_pkg = types.ModuleType("rtp_llm.utils")
        utils_pkg.__path__ = []
        sys.modules["rtp_llm.utils"] = utils_pkg
        import_util = types.ModuleType("rtp_llm.utils.import_util")
        import_util.has_internal_source = lambda: False
        sys.modules["rtp_llm.utils.import_util"] = import_util
        triton_compile_patch = types.ModuleType("rtp_llm.utils.triton_compile_patch")
        triton_compile_patch.enable_compile_monitor = lambda: None
        sys.modules["rtp_llm.utils.triton_compile_patch"] = triton_compile_patch
        sys.modules["pytest"] = types.ModuleType("pytest")
        sys.modules.pop("triton", None)

        class BoomLoader(importlib.abc.Loader):
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                raise RuntimeError("triton boom")

        class BoomFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "triton":
                    return importlib.machinery.ModuleSpec(fullname, BoomLoader())
                return None

        sys.meta_path.insert(0, BoomFinder())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spec.loader.exec_module(module)

        assert module._bootstrap_error is not None
        assert any("Skipping heavy rtp_llm bootstrap" in str(w.message) for w in caught)
        """
    )

    result = _run_bootstrap_probe(code)

    assert result.returncode == 0, result.stderr + result.stdout


def test_bootstrap_error_raises_after_pytest_conftest_has_run():
    code = textwrap.dedent(
        f"""
        import importlib.abc
        import importlib.machinery
        import importlib.util
        import sys
        import types

        init_path = {str(INIT_PATH)!r}
        pkg_dir = {str(PKG_DIR)!r}

        spec = importlib.util.spec_from_file_location(
            "rtp_llm", init_path, submodule_search_locations=[pkg_dir]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["rtp_llm"] = module

        utils_pkg = types.ModuleType("rtp_llm.utils")
        utils_pkg.__path__ = []
        sys.modules["rtp_llm.utils"] = utils_pkg
        import_util = types.ModuleType("rtp_llm.utils.import_util")
        import_util.has_internal_source = lambda: False
        sys.modules["rtp_llm.utils.import_util"] = import_util
        triton_compile_patch = types.ModuleType("rtp_llm.utils.triton_compile_patch")
        triton_compile_patch.enable_compile_monitor = lambda: None
        sys.modules["rtp_llm.utils.triton_compile_patch"] = triton_compile_patch
        sys.modules["pytest"] = types.ModuleType("pytest")
        sys._RTP_CONFTEST_DONE = True
        sys.modules["triton"] = types.ModuleType("triton")

        class BoomLoader(importlib.abc.Loader):
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                raise RuntimeError("ops boom")

        class BoomFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "rtp_llm.ops":
                    return importlib.machinery.ModuleSpec(fullname, BoomLoader())
                return None

        sys.meta_path.insert(0, BoomFinder())

        try:
            spec.loader.exec_module(module)
        except RuntimeError as exc:
            assert "ops boom" in str(exc)
        else:
            raise AssertionError("bootstrap error was unexpectedly suppressed")
        """
    )

    result = _run_bootstrap_probe(code)

    assert result.returncode == 0, result.stderr + result.stdout
