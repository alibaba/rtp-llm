import os
import sys
import tempfile
import unittest
from unittest import mock

from rtp_llm.utils import aiter_jit_patch


class AiterJitPatchTest(unittest.TestCase):
    def test_matches_only_codegen_scripts_inside_aiter_packages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            aiter_root = os.path.join(tmpdir, "site-packages", "aiter_meta")
            os.makedirs(aiter_root)
            aiter_script = os.path.join(aiter_root, "generate.py")
            unrelated_script = os.path.join(tmpdir, "generate.py")
            with mock.patch.object(
                aiter_jit_patch,
                "_aiter_package_roots",
                return_value=(aiter_root,),
            ):
                self.assertTrue(
                    aiter_jit_patch._is_aiter_codegen_command(
                        f"{sys.executable} {aiter_script} --output_dir /tmp/out"
                    )
                )
                self.assertFalse(
                    aiter_jit_patch._is_aiter_codegen_command(
                        f"{sys.executable} {unrelated_script} --output_dir /tmp/out"
                    )
                )

    def test_matches_bazel_runfiles_script_symlink(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            aiter_root = os.path.join(tmpdir, "runfiles", "aiter_meta")
            real_script = os.path.join(tmpdir, "external", "codegen.py")
            logical_script = os.path.join(aiter_root, "hsa", "codegen.py")
            os.makedirs(os.path.dirname(real_script))
            os.makedirs(os.path.dirname(logical_script))
            with open(real_script, "w"):
                pass
            os.symlink(real_script, logical_script)
            with mock.patch.object(
                aiter_jit_patch,
                "_aiter_package_roots",
                return_value=(aiter_root,),
            ):
                self.assertTrue(
                    aiter_jit_patch._is_aiter_codegen_command(
                        f"{sys.executable} {logical_script} --output_dir /tmp/out"
                    )
                )

    def test_codegen_command_does_not_expand_filter_globs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "generate.py")
            args_path = os.path.join(tmpdir, "args.txt")
            with open(script_path, "w") as f:
                f.write(
                    "import pathlib, sys\n"
                    f"pathlib.Path({args_path!r}).write_text('\\n'.join(sys.argv[1:]))\n"
                )

            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                open("fmha_fwd_d256_bf16_kernel.o", "w").close()
                command = (
                    f"{sys.executable} {script_path} "
                    "--filter *bf16*_nlogits* --output_dir /tmp/out"
                )
                ret = aiter_jit_patch._run_aiter_codegen_command(command)
            finally:
                os.chdir(cwd)

            self.assertEqual(ret, 0)
            with open(args_path) as f:
                args = f.read().splitlines()
            self.assertIn("*bf16*_nlogits*", args)
            self.assertNotIn("fmha_fwd_d256_bf16_kernel.o", args)

    def test_codegen_command_preserves_exit_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "generate.py")
            with open(script_path, "w") as f:
                f.write("import sys\nsys.exit(7)\n")

            ret = aiter_jit_patch._run_aiter_codegen_command(
                f"{sys.executable} {script_path}"
            )

            self.assertEqual(ret, 7)

    def test_remove_stale_aiter_jit_module(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jit_dir = os.path.join(tmpdir, "aiter", "jit")
            build_dir = os.path.join(jit_dir, "build", "mha_batch_prefill_bad")
            os.makedirs(build_dir)
            so_path = os.path.join(jit_dir, "mha_batch_prefill_bad.so")
            with open(so_path, "w"):
                pass

            removed = aiter_jit_patch._maybe_remove_stale_aiter_jit_module(
                "aiter.jit.mha_batch_prefill_bad",
                ImportError(f"{so_path}: undefined symbol: fmha_batch_prefill"),
            )

            self.assertTrue(removed)
            self.assertFalse(os.path.exists(so_path))
            self.assertFalse(os.path.exists(build_dir))


if __name__ == "__main__":
    unittest.main()
