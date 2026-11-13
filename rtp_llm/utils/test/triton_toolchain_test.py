import contextlib
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rtp_llm.utils import triton_toolchain


class TritonToolchainTest(unittest.TestCase):
    def setUp(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        stack.enter_context(mock.patch.dict(os.environ))
        for name in ("TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH", "CUDA_HOME"):
            os.environ.pop(name, None)
        stack.enter_context(
            mock.patch.object(triton_toolchain, "_DEFAULT_PTXAS_PATHS", ())
        )

    def make_ptxas(self, name, version="13.2", *, status=0, stderr=False):
        path = self.root / name / "bin" / "ptxas"
        path.parent.mkdir(parents=True)
        output = f"ptxas: NVIDIA (R) Ptx optimizing assembler\nCuda compilation tools, release {version}, V{version}.51"
        path.write_text(
            '#!/bin/sh\n[ "$1" = "--version" ] || exit 2\n'
            f"printf '%s\\n' {shlex.quote(output)}{' >&2' if stderr else ''}\n"
            f"exit {status}\n"
        )
        path.chmod(0o755)
        return str(path)

    def test_generic_ptxas_is_selected_for_blackwell(self):
        generic = self.make_ptxas("generic")
        os.environ["TRITON_PTXAS_PATH"] = generic
        os.environ["CUDA_HOME"] = str(Path(self.make_ptxas("cuda-home")).parents[1])

        with self.assertLogs(level="INFO") as logs:
            triton_toolchain.configure_sm120_triton_ptxas()

        self.assertEqual(os.environ["TRITON_PTXAS_BLACKWELL_PATH"], generic)
        self.assertIn(generic, "\n".join(logs.output))
        self.assertIn("13.2", "\n".join(logs.output))

    def test_explicit_blackwell_selection_is_never_overridden(self):
        os.environ["TRITON_PTXAS_PATH"] = self.make_ptxas("generic")
        for explicit in (self.make_ptxas("old", "12.9"), "/missing/ptxas", ""):
            with self.subTest(explicit=explicit):
                os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = explicit

                triton_toolchain.configure_sm120_triton_ptxas()

                self.assertEqual(os.environ["TRITON_PTXAS_BLACKWELL_PATH"], explicit)

    def test_old_generic_compiler_falls_back_to_cuda_home(self):
        generic = self.make_ptxas("generic", "12.9")
        cuda_home = self.make_ptxas("cuda-home", "13.2")
        os.environ["TRITON_PTXAS_PATH"] = generic
        os.environ["CUDA_HOME"] = str(Path(cuda_home).parents[1])

        triton_toolchain.configure_sm120_triton_ptxas()

        self.assertEqual(os.environ["TRITON_PTXAS_BLACKWELL_PATH"], cuda_home)
        self.assertEqual(os.environ["TRITON_PTXAS_PATH"], generic)

    def test_unset_paths_use_an_installed_default_compiler(self):
        missing = str(self.root / "missing" / "ptxas")
        installed = self.make_ptxas("installed", "13.2")
        with mock.patch.object(
            triton_toolchain, "_DEFAULT_PTXAS_PATHS", (missing, installed)
        ):
            triton_toolchain.configure_sm120_triton_ptxas()

        self.assertEqual(os.environ["TRITON_PTXAS_BLACKWELL_PATH"], installed)

    def test_version_probe_rejects_old_and_malformed_versions(self):
        for version in ("12.9", "13.1", "unknown"):
            with self.subTest(version=version):
                os.environ["TRITON_PTXAS_PATH"] = self.make_ptxas(version, version)

                triton_toolchain.configure_sm120_triton_ptxas()

                self.assertNotIn("TRITON_PTXAS_BLACKWELL_PATH", os.environ)

    def test_version_probe_accepts_newer_versions_and_stderr_output(self):
        for version in ("13.2", "13.10", "14.0"):
            with self.subTest(version=version):
                compiler = self.make_ptxas(version, version, stderr=True)
                os.environ["TRITON_PTXAS_PATH"] = compiler
                os.environ.pop("TRITON_PTXAS_BLACKWELL_PATH", None)

                triton_toolchain.configure_sm120_triton_ptxas()

                self.assertEqual(os.environ["TRITON_PTXAS_BLACKWELL_PATH"], compiler)

    def test_unusable_compilers_leave_existing_environment_unchanged(self):
        not_executable = self.make_ptxas("not-executable")
        Path(not_executable).chmod(0o644)
        for compiler in (
            str(self.root / "missing" / "ptxas"),
            not_executable,
            self.make_ptxas("failed", status=1),
        ):
            with self.subTest(compiler=compiler):
                os.environ["TRITON_PTXAS_PATH"] = compiler
                before = dict(os.environ)

                triton_toolchain.configure_sm120_triton_ptxas()

                self.assertEqual(dict(os.environ), before)

    def test_timed_out_probe_does_not_prevent_server_startup(self):
        os.environ["TRITON_PTXAS_PATH"] = "/stuck/ptxas"
        with mock.patch.object(
            triton_toolchain.subprocess,
            "check_output",
            side_effect=subprocess.TimeoutExpired("/stuck/ptxas", 5),
        ):
            triton_toolchain.configure_sm120_triton_ptxas()

        self.assertNotIn("TRITON_PTXAS_BLACKWELL_PATH", os.environ)

    def test_repeated_setup_preserves_the_initial_selection(self):
        first = self.make_ptxas("first")
        os.environ["TRITON_PTXAS_PATH"] = first
        triton_toolchain.configure_sm120_triton_ptxas()
        os.environ["TRITON_PTXAS_PATH"] = self.make_ptxas("second", "14.0")

        triton_toolchain.configure_sm120_triton_ptxas()

        self.assertEqual(os.environ["TRITON_PTXAS_BLACKWELL_PATH"], first)


if __name__ == "__main__":
    unittest.main()
