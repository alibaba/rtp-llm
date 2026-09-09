"""Exercise the pytest launcher as a subprocess, including its result files."""

import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


class PytestLauncherTest(unittest.TestCase):
    def _run(self, body, env_overrides=None):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "test_fixture.py"
            source.write_text(body)
            report = Path(root) / "results.xml"
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(sys.path)
            env["XML_OUTPUT_FILE"] = str(report)
            env["TEST_TMPDIR"] = root
            for name, value in (env_overrides or {}).items():
                if value is None:
                    env.pop(name, None)
                else:
                    env[name] = value
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("pytest_main.py")),
                    str(source),
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            xml = ET.parse(report).getroot() if report.exists() else None
            return process, xml

    def test_collects_functions_missing_from_handwritten_main(self):
        process, report = self._run(
            "def test_first():\n    assert True\n"
            "def test_second():\n    assert False, 'new regression'\n"
            "if __name__ == '__main__':\n    test_first()\n"
        )
        self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn("new regression", process.stdout)
        self.assertIsNotNone(report)
        self.assertEqual(len(report.findall(".//testcase")), 2)

    def test_reports_successful_test_bodies(self):
        process, report = self._run("def test_pass():\n    assert 3 == 3\n")
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn("collected=1 passed=1 failed=0", process.stdout)
        self.assertIsNotNone(report)
        self.assertEqual(len(report.findall(".//testcase")), 1)

    def test_all_skipped_is_not_success(self):
        process, _ = self._run(
            "import pytest\n"
            "@pytest.mark.skip(reason='fixture unavailable')\n"
            "def test_missing():\n    assert True\n"
        )
        self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn("no test body passed", process.stdout)

    def test_empty_collection_is_not_success(self):
        process, _ = self._run("value = 3\n")
        self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_collection_error_is_not_success(self):
        process, _ = self._run("raise RuntimeError('missing runtime dependency')\n")
        self.assertNotEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn("missing runtime dependency", process.stdout)

    def test_missing_home_gets_absolute_cache_before_collection(self):
        process, _ = self._run(
            "import os\n"
            "from pathlib import Path\n"
            "cache_at_import = Path(os.environ['DG_JIT_CACHE_DIR'])\n"
            "def test_cache():\n"
            "    assert 'HOME' not in os.environ\n"
            "    assert cache_at_import.is_absolute()\n"
            "    assert cache_at_import == Path(os.environ['TEST_TMPDIR']) / 'deep_gemm'\n"
            "    cache_at_import.mkdir()\n"
            "    source = cache_at_import / 'kernel.cu'\n"
            "    source.write_text('kernel source')\n"
            "    os.chdir(cache_at_import.parent)\n"
            "    assert source.read_text() == 'kernel source'\n",
            env_overrides={"HOME": None, "DG_JIT_CACHE_DIR": None},
        )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_explicit_deep_gemm_cache_is_preserved(self):
        process, _ = self._run(
            "import os\n"
            "cache_at_import = os.environ['DG_JIT_CACHE_DIR']\n"
            "def test_cache():\n"
            "    assert cache_at_import == '/configured/deep-gemm-cache'\n",
            env_overrides={
                "HOME": None,
                "DG_JIT_CACHE_DIR": "/configured/deep-gemm-cache",
            },
        )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_missing_bazel_temp_directory_gets_absolute_cache(self):
        process, _ = self._run(
            "import os\n"
            "from pathlib import Path\n"
            "cache_at_import = Path(os.environ['DG_JIT_CACHE_DIR'])\n"
            "def test_cache():\n"
            "    assert cache_at_import.is_absolute()\n"
            "    assert cache_at_import.name == 'deep_gemm'\n",
            env_overrides={"HOME": None, "TEST_TMPDIR": None, "DG_JIT_CACHE_DIR": None},
        )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_relative_deep_gemm_cache_keeps_its_directory_after_chdir(self):
        with tempfile.TemporaryDirectory() as root:
            cache = Path(root) / "configured-cache"
            process, _ = self._run(
                "import os\n"
                "from pathlib import Path\n"
                "cache_at_import = Path(os.environ['DG_JIT_CACHE_DIR'])\n"
                "def test_cache():\n"
                f"    assert cache_at_import == Path({str(cache)!r})\n"
                "    assert cache_at_import.is_absolute()\n"
                "    cache_at_import.mkdir()\n"
                "    source = cache_at_import / 'kernel.cu'\n"
                "    source.write_text('kernel source')\n"
                "    os.chdir(os.environ['TEST_TMPDIR'])\n"
                "    assert source.read_text() == 'kernel source'\n",
                env_overrides={"DG_JIT_CACHE_DIR": os.path.relpath(cache)},
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)


if __name__ == "__main__":
    unittest.main()
