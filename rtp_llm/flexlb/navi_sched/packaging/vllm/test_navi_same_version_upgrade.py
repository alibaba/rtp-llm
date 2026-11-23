"""Check that fast images really replace a wheel with the same package version."""

import base64
import csv
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import venv
import zipfile


class SameVersionUpgradeTest(unittest.TestCase):
    def test_same_version_replaces_installed_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = dict(
                {
                    key: value
                    for key, value in os.environ.items()
                    if key not in ("PYTHONPATH", "PYTHONHOME")
                },
                PIP_NO_INDEX="1",
                PIP_DISABLE_PIP_VERSION_CHECK="1",
                OMEGA_MODELSCOPE_RUNTIME="0",
            )
            venv.EnvBuilder(with_pip=True).create(root / "venv")
            python = root / "venv/bin/python"
            tag = "cp%d%d" % sys.version_info[:2]

            def wheel(folder, value):
                folder.mkdir()
                path = folder / f"navi_sched-0.1.0-{tag}-{tag}-linux_x86_64.whl"
                dist = "navi_sched-0.1.0.dist-info/"
                files = {
                    "navi_upgrade_fixture.py": f"VALUE = {value!r}\n".encode(),
                    dist
                    + "METADATA": b"Metadata-Version: 2.1\nName: navi-sched\nVersion: 0.1.0\n",
                    dist
                    + "WHEEL": f"Wheel-Version: 1.0\nGenerator: regression-test\nRoot-Is-Purelib: true\nTag: {tag}-{tag}-linux_x86_64\n".encode(),
                }
                record = io.StringIO()
                writer = csv.writer(record)
                for name, data in files.items():
                    digest = (
                        base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                        .rstrip(b"=")
                        .decode()
                    )
                    writer.writerow([name, "sha256=" + digest, len(data)])
                writer.writerow([dist + "RECORD", "", ""])
                files[dist + "RECORD"] = record.getvalue().encode()
                with zipfile.ZipFile(path, "w") as archive:
                    for name, data in files.items():
                        archive.writestr(name, data)
                return path

            old = wheel(root / "old", "old")
            new = wheel(root / "new", "new-adapter-code")
            subprocess.run(
                [str(python), "-m", "pip", "install", "--no-deps", str(old)],
                env=environment,
                check=True,
                capture_output=True,
            )

            def value():
                return subprocess.check_output(
                    [
                        str(python),
                        "-c",
                        "from navi_upgrade_fixture import VALUE; print(VALUE)",
                    ],
                    env=environment,
                    text=True,
                ).strip()

            self.assertEqual("old", value())
            source = root / "source"
            source.mkdir()
            fields = {
                "NAVI_SCHED_SOURCE_BRANCH": "flexlb_adapt",
                "NAVI_SCHED_SOURCE_REPOSITORY": "foundation_models/RTP-LLM",
                "NAVI_SCHED_RTP_INTERNAL_COMMIT": "b" * 40,
                "NAVI_SCHED_COMMIT": "a" * 40,
                "NAVI_SCHED_WHEEL_VERSION": "0.1.0",
                "NAVI_SCHED_WHEEL_ARCHITECTURE": "amd64",
                "NAVI_SCHED_WHEEL_PYTHON_TAG": tag,
                "NAVI_SCHED_WHEEL_SHA256": hashlib.sha256(new.read_bytes()).hexdigest(),
            }
            for name, data in fields.items():
                (source / name).write_text(data + "\n")
            command = [
                "sh",
                str(Path(__file__).with_name("install_navi_sched_wheel.sh")),
                str(python),
                str(new),
                str(source),
                str(root / "info"),
                "a" * 40,
                "0.1.0",
                (
                    "ppu-deploy"
                    if environment.get("NAVI_SCHED_VLLM_PLATFORM") == "ppu"
                    else "v0.17.1_omega"
                ),
                "c" * 40,
                "amd64",
                "flexlb_adapt",
                "b" * 40,
            ]
            result = subprocess.run(
                command, env=environment, capture_output=True, text=True
            )
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            self.assertEqual("new-adapter-code", value())


if __name__ == "__main__":
    unittest.main()
