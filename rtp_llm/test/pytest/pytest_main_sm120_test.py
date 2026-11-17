"""SM120 cases split out of pytest_main_test.py.

``pytest_main_test.py`` and ``pytest_main.py`` are left unchanged; the
SM120-capable launcher lives in the sibling ``pytest_main_sm120.py`` and
these cases exercise it as a real subprocess with normal XML result files
(no mocks).
"""

import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


class PytestLauncherSm120Test(unittest.TestCase):
    """Sibling of ``PytestLauncherTest`` carrying the SM120 cases."""

    def _run(self, body, cuda_args=(), capability=None):
        # Mirrors the main test's helper verbatim except for the launcher
        # path: the sibling pytest_main_sm120.py carries the SM120 gate so
        # the main launcher is never touched.
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "test_fixture.py"
            source.write_text(body)
            report = Path(root) / "results.xml"
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(sys.path)
            if capability is not None:
                (Path(root) / "torch.py").write_text(
                    "from types import SimpleNamespace\n"
                    "version = SimpleNamespace(cuda='13.2')\n"
                    "cuda = SimpleNamespace(device_count=lambda: 1, "
                    f"get_device_capability=lambda index: {capability!r})\n"
                )
                env["PYTHONPATH"] = root + os.pathsep + env["PYTHONPATH"]
            env["XML_OUTPUT_FILE"] = str(report)
            env["TEST_TMPDIR"] = root
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("pytest_main_sm120.py")),
                    *cuda_args,
                    str(source),
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            xml = ET.parse(report).getroot() if report.exists() else None
            return process, xml

    def test_exact_sm120_gate_rejects_other_minor_architectures(self):
        for capability, expected_success in [((12, 0), True), ((12, 1), False)]:
            with self.subTest(capability=capability):
                process, report = self._run(
                    "def test_pass():\n    assert True\n",
                    cuda_args=(
                        "--cuda-devices",
                        "1",
                        "--cuda-major",
                        "12",
                        "--cuda-minor",
                        "0",
                    ),
                    capability=capability,
                )
                self.assertEqual(
                    process.returncode == 0, expected_success, process.stderr
                )
                if expected_success:
                    self.assertIsNotNone(report)
                else:
                    self.assertIn("require SM120 devices, found SM121", process.stderr)
                    self.assertIsNone(report)

    def test_major_only_gate_preserves_minor_compatibility(self):
        process, report = self._run(
            "def test_pass():\n    assert True\n",
            cuda_args=("--cuda-devices", "1", "--cuda-major", "12"),
            capability=(12, 1),
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIsNotNone(report)


if __name__ == "__main__":
    unittest.main()
