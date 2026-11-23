import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "prepare_context", Path(__file__).with_name("prepare_context.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BuildContextTest(unittest.TestCase):
    def test_source_bundle_and_checkout_credentials_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "vllm"
            source.mkdir()
            subprocess.run(["git", "init", "-q", str(source)], check=True)

            def git(*args):
                return subprocess.check_output(
                    ["git", "-C", str(source), *args], text=True
                ).strip()

            git(
                "remote",
                "add",
                "origin",
                "https://fixture-user:fixture-password@code.example/vllm.git",
            )
            git("config", "http.extraheader", "Authorization: fixture-only")
            (source / "vllm").mkdir()
            (source / "vllm/code.py").write_text("value = 1\n")
            (source / ".dockerignore").write_text("/.venv\nvllm/*.so\n__pycache__/\n")
            git("add", ".")
            git(
                "-c",
                "user.name=fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "fixture",
            )
            bundle = source / "navi_sched_dist"
            bundle.mkdir()
            (bundle / "navi_sched.whl").write_bytes(b"compiled install inputs")
            (source / ".navi_sources").mkdir()
            (source / ".navi_sources/private-checkout").write_text("not shipped")
            target = source / "navi_build_context"
            module.prepare_context(source, bundle, target)
            code = target / "vllm_source"
            self.assertEqual("value = 1\n", (code / "vllm/code.py").read_text())
            self.assertFalse((code / "navi_sched_dist").exists())
            self.assertFalse((code / ".navi_sources").exists())
            self.assertFalse((code / "navi_build_context").exists())
            self.assertEqual(
                b"compiled install inputs",
                (target / "navi_sched_dist/navi_sched.whl").read_bytes(),
            )
            self.assertIn("/vllm_source/.venv", (target / ".dockerignore").read_text())
            self.assertIn(
                "vllm_source/vllm/*.so", (target / ".dockerignore").read_text()
            )
            copied_config = (code / ".git/config").read_text()
            self.assertNotIn("fixture-password", copied_config)
            self.assertNotIn("Authorization", copied_config)
            self.assertEqual(
                git("rev-parse", "HEAD"),
                subprocess.check_output(
                    ["git", "-C", str(code), "rev-parse", "HEAD"], text=True
                ).strip(),
            )
            self.assertIn("fixture-password", git("remote", "get-url", "origin"))
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                module.prepare_context(source, bundle, target)


if __name__ == "__main__":
    unittest.main()
