# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import zipfile
from email.parser import Parser
from pathlib import Path

WHEEL_PATH = Path(sys.argv.pop(1))

EXPECTED_CUDA12_9_REQUIREMENTS = {
    "torch": (
        "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/cu129/"
        "torch-2.8.0%2Bcu129-cp310-cp310-manylinux_2_28_x86_64.whl"
        "#sha256=54d240b5d3b1f9075d4ee6179675a22c1974f7bef1885d134c582678d5180cd3"
    ),
    "torchvision": (
        "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/cu129/"
        "torchvision-0.23.0%2Bcu129-cp310-cp310-manylinux_2_28_x86_64.whl"
        "#sha256=5690810877f2d7d1a2b432e31d68d4a9ccbb695a9a8fa0e27bbad44c6a90a181"
    ),
    "fast-safetensors": (
        "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/cu129/"
        "fast_safetensors-0.7.3%2Btorch2.1.2.cu121-cp310-cp310-linux_x86_64.whl"
        "#sha256=dd760931feb6dd585cc0b14b1bacf39c1e31a39bb9f8e12a8c62720caf1ccc1f"
    ),
    "fastsafetensors": (
        "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/cu129/"
        "fastsafetensors-0.3.4.dev20260901%2Bali.fuseshm.g78ac75c8.aone67880226-"
        "cp310-cp310-linux_x86_64.whl"
        "#sha256=bb084a01e6b3d97a8790e583cb5c0fcadcef1960e64b069bb2a97fff0079fe40"
    ),
}


class RtpLlmWheelMetadataTest(unittest.TestCase):
    def test_cuda12_9_direct_requirements_match_lock_file(self) -> None:
        with zipfile.ZipFile(WHEEL_PATH) as wheel:
            metadata_files = [
                name
                for name in wheel.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            self.assertEqual(len(metadata_files), 1)
            metadata = Parser().parsestr(wheel.read(metadata_files[0]).decode())

        requirements = metadata.get_all("Requires-Dist", [])
        for package, expected_url in EXPECTED_CUDA12_9_REQUIREMENTS.items():
            matching = [
                requirement
                for requirement in requirements
                if requirement.partition("@")[0].strip().lower() == package
            ]
            self.assertEqual(len(matching), 1)
            self.assertEqual(matching[0].partition("@")[2].strip(), expected_url)

        self.assertNotIn("torch==2.6.0+cu126", requirements)


if __name__ == "__main__":
    unittest.main()
