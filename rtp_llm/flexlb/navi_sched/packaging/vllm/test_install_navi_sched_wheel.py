import os
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest


class RTPWheelInstallTest(unittest.TestCase):
    def install(
        self,
        architecture,
        mismatch=False,
        ppu=False,
        abi="cp312",
        reject=False,
        branch=None,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            suffix = "linux_x86_64" if architecture == "amd64" else "linux_aarch64"
            tag = abi if ppu else "cp310"
            wheel = root / (
                "navi_sched-0.1.0-" + tag + "-" + tag + "-" + suffix + ".whl"
            )
            wheel.write_bytes(b"wheel fixture")
            fields = {
                "NAVI_SCHED_SOURCE_BRANCH": "flexlb_adapt",
                "NAVI_SCHED_SOURCE_REPOSITORY": "foundation_models/RTP-LLM",
                "NAVI_SCHED_COMMIT": "a" * 40,
                "NAVI_SCHED_RTP_INTERNAL_COMMIT": ("c" if mismatch else "b") * 40,
                "NAVI_SCHED_WHEEL_VERSION": "0.1.0",
                "NAVI_SCHED_WHEEL_ARCHITECTURE": architecture,
                "NAVI_SCHED_WHEEL_SHA256": hashlib.sha256(
                    wheel.read_bytes()
                ).hexdigest(),
                "NAVI_SCHED_WHEEL_PYTHON_TAG": tag,
            }
            for name, value in fields.items():
                (source / name).write_text(value + "\n")
            python = root / "python"
            marker = root / "installed"
            python.write_text(
                '#!/bin/sh\nif [ "$1" = -m ]; then touch "'
                + str(marker)
                + '"; elif echo "$2" | grep -q sys.version_info; '
                "then echo cp312; else echo 0.1.0; fi\n"
            )
            python.chmod(0o755)
            result = subprocess.run(
                [
                    "sh",
                    str(Path(__file__).with_name("install_navi_sched_wheel.sh")),
                    str(python),
                    str(wheel),
                    str(source),
                    str(root / "info"),
                    "a" * 40,
                    "0.1.0",
                    branch or ("ppu-deploy-navi" if ppu else "navi_sched"),
                    "d" * 40,
                    architecture,
                    "flexlb_adapt",
                    "b" * 40,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env={
                    **os.environ,
                    "NAVI_SCHED_VLLM_PLATFORM": "ppu" if ppu else "cuda",
                },
            )
            if mismatch or reject:
                self.assertNotEqual(0, result.returncode)
                self.assertFalse(marker.exists())
            else:
                self.assertEqual(0, result.returncode, result.stdout)
                self.assertTrue(marker.exists())
                values = dict(
                    line.split("=", 1)
                    for line in (root / "info/build-info.env").read_text().splitlines()
                )
                self.assertEqual("flexlb_adapt", values["NAVI_SCHED_SOURCE_BRANCH"])
                self.assertEqual("b" * 40, values["NAVI_SCHED_RTP_INTERNAL_COMMIT"])
                self.assertEqual(architecture, values["IMAGE_ARCHITECTURE"])

    def test_x86(self):
        self.install("amd64")

    def test_arm(self):
        self.install("arm64")

    def test_reject_internal_revision_before_install(self):
        self.install("amd64", mismatch=True)

    def test_versioned_ppu_cp312(self):
        self.install("amd64", ppu=True, branch="navi_sched_0.20.1_ppu")

    def test_versioned_ppu_rejects_cp310_before_install(self):
        self.install(
            "amd64", ppu=True, abi="cp310", reject=True, branch="navi_sched_0.20.1_ppu"
        )

    def test_target_ppu_cp312(self):
        self.install("amd64", ppu=True, branch="ppu-deploy")

    def test_target_ppu_rejects_cp310(self):
        self.install("amd64", ppu=True, branch="ppu-deploy", abi="cp310", reject=True)

    def test_ppu_cp312(self):
        self.install("amd64", ppu=True)

    def test_ppu_rejects_cp310_before_install(self):
        self.install("amd64", ppu=True, abi="cp310", reject=True)

    def test_ppu_rejects_arm_before_install(self):
        self.install("arm64", ppu=True, reject=True)

    def test_cuda_source_branch(self):
        self.install("amd64", branch="navi_sched_0.17.1_cuda")

    def test_cuda_target_branch(self):
        self.install("amd64", branch="v0.17.1_omega")


if __name__ == "__main__":
    unittest.main()
