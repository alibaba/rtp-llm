# SPDX-License-Identifier: Apache-2.0

import re
import sys
import unittest
import zipfile
from email.parser import Parser
from pathlib import Path

PLATFORM = sys.argv.pop(1)
WHEEL_PATH = Path(sys.argv.pop(1))


def _pinned(url: str, sha256: str) -> str:
    return f"{url}#sha256={sha256}"


EXPECTED_REQUIREMENTS = {
    "cuda12_9_x86": {
        "torch": _pinned(
            "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/cu129/"
            "torch-2.8.0%2Bcu129-cp310-cp310-manylinux_2_28_x86_64.whl",
            "54d240b5d3b1f9075d4ee6179675a22c1974f7bef1885d134c582678d5180cd3",
        ),
        "torchvision": _pinned(
            "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/cu129/"
            "torchvision-0.23.0%2Bcu129-cp310-cp310-manylinux_2_28_x86_64.whl",
            "5690810877f2d7d1a2b432e31d68d4a9ccbb695a9a8fa0e27bbad44c6a90a181",
        ),
        "fast-safetensors": _pinned(
            "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/cu129/"
            "fast_safetensors-0.7.3%2Btorch2.1.2.cu121-cp310-cp310-linux_x86_64.whl",
            "dd760931feb6dd585cc0b14b1bacf39c1e31a39bb9f8e12a8c62720caf1ccc1f",
        ),
        "fastsafetensors": _pinned(
            "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/cu129/"
            "fastsafetensors-0.3.4.dev20260901%2Bali.fuseshm.g78ac75c8.aone67880226-"
            "cp310-cp310-linux_x86_64.whl",
            "bb084a01e6b3d97a8790e583cb5c0fcadcef1960e64b069bb2a97fff0079fe40",
        ),
    },
    "cuda13_x86": {
        "torch": _pinned(
            "https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/miji/0430/"
            "torch-2.11.0%2Bcu130-cp310-cp310-manylinux_2_28_x86_64.whl",
            "4c5be01584b7fee22d3c0d04062fd28026044acd07ffd0ee64cbd54b60e62d39",
        ),
        "torchvision": _pinned(
            "https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/miji/0430/"
            "torchvision-0.26.0%2Bcu130-cp310-cp310-manylinux_2_28_x86_64.whl",
            "7d820708732d2467caf2ef16c3ad60c8ff402bd05bd246b108ee080f3fbfdc6e",
        ),
        "deep-gemm": _pinned(
            "https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/rtp_llm/deep_gemm/"
            "cuda13_b200/deep_gemm-2.5.0%2B8a4dfba-cp310-cp310-linux_x86_64.whl",
            "5844fee49160525128ca49b0e8ae73aac1eb4db44da108ede222cc85e9aac50a",
        ),
        "flash-mla": _pinned(
            "https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/miji/0430/"
            "flash_mla-1.0.0%2B9241ae3-cp310-cp310-linux_x86_64.whl",
            "67bba01c854fe3b06397c297d49724e16029984a37124d05047a78ff48e7c44f",
        ),
        "rtp-kernel": _pinned(
            "https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/miji/0430/"
            "rtp_kernel-0.1.0%2Bcu13.4a1a7e3-cp310-cp310-linux_x86_64.whl",
            "69992ef98016b1f176a092741841b79c9eba540c819fcaec663ad7550eb6aa52",
        ),
        "fast-safetensors": _pinned(
            "https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/0507/"
            "fast_safetensors-0.7.3%2Btorch2.11.cu130-cp310-cp310-linux_x86_64.whl",
            "dd6aa7fe17acf6bfb3ea00dbfb934146b78d852ef3e0b1979dd0edb8bfaaf19a",
        ),
        "fastsafetensors": _pinned(
            "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/cu130/"
            "fastsafetensors-0.3.4.dev20260901%2Bali.fuseshm.g78ac75c8.aone67880226-"
            "cp310-cp310-linux_x86_64.whl",
            "bb084a01e6b3d97a8790e583cb5c0fcadcef1960e64b069bb2a97fff0079fe40",
        ),
    },
    "cuda13_arm": {
        "torch": _pinned(
            "https://rtp-maga.cn-zhangjiakou.oss.aliyuncs.com/rtp_llm/arm_pkg/"
            "torch-2.11.0%2Bcu130-cp310-cp310-manylinux_2_28_aarch64.whl",
            "4af01fad0822353e766770ff2c7d6bdc2cbcc2ac7fcd6da93a9e3c6f3f932b21",
        ),
        "torchvision": _pinned(
            "https://rtp-maga.cn-zhangjiakou.oss.aliyuncs.com/rtp_llm/arm_pkg/"
            "torchvision-0.26.0%2Bcu130-cp310-cp310-manylinux_2_28_aarch64.whl",
            "3094bed175eee817f9fc61d1c8bdecb1d807c25f49517fbfe60f15d24135fcde",
        ),
        "deep-gemm": _pinned(
            "https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/rtp_llm/deep_gemm/"
            "cuda13_gb300/deep_gemm-2.5.0%2B6053f00-cp310-cp310-linux_aarch64.whl",
            "0b2b85be56f2f2f4401025f3d113dc1af59ac745aeac638be06a277c82f2abe9",
        ),
        "flash-mla": _pinned(
            "https://rtp-maga.cn-zhangjiakou.oss.aliyuncs.com/0530/arm_pkg/sglang/"
            "flash_mla-1.0.0%2B92fd68b-cp310-cp310-linux_aarch64.whl",
            "6b03d0663596016ca1d2832e402ef42450be88d1f9586e1bf8f1c3626481320d",
        ),
        "rtp-kernel": _pinned(
            "https://rtp-maga.cn-zhangjiakou.oss.aliyuncs.com/0608/arm_pkg/"
            "rtp_kernel-0.1.0%2Bcu13.fb4b4ab-cp310-cp310-linux_aarch64.whl",
            "2df85dae2bba17de32f38546454344e590142fef18272425e0e050a86d658b73",
        ),
        "fast-safetensors": _pinned(
            "https://rtp-maga.cn-zhangjiakou.oss.aliyuncs.com/0513/arm_pkg/"
            "fast_safetensors-0.7.3%2Btorch2.11.cu130-cp310-cp310-linux_aarch64.whl",
            "46355b2e6b138248677f5306060a3eee5d05d9baf1d9c026ef069c84b2ea5a99",
        ),
        "fastsafetensors": _pinned(
            "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/cu130_arm/"
            "fastsafetensors-0.3.4.dev20260901%2Bali.fuseshm.g78ac75c8.aone67880226-"
            "cp310-cp310-linux_aarch64.whl",
            "95a89d0b641b7dd57f8e137a04a82d3dd9d09cf36378764d81662cfd0185f8c9",
        ),
        "tilelang": _pinned(
            "https://rtp-maga.cn-zhangjiakou.oss.aliyuncs.com/rtp_llm/arm_pkg/"
            "tilelang-0.1.9%2Bcuda.git441c3b06-cp38-abi3-linux_aarch64.whl",
            "44c4e53b75919d97b5af467f5667bd9bfcfc30b14cec1a76e8ba1db32ac4a763",
        ),
        "z3-solver": _pinned(
            "https://rtp-maga.cn-zhangjiakou.oss.aliyuncs.com/rtp_llm/arm_pkg/"
            "z3_solver-4.13.0.0%2Blocal.ali-py2.py3-none-manylinux2014_aarch64.whl",
            "e94170200c64f5d67295529e0568b15ba59a5f3b29c496402f8c941aec4aaa9b",
        ),
    },
}


def _normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


class RtpLlmWheelMetadataTest(unittest.TestCase):
    def test_direct_requirements_match_platform_lock_file(self) -> None:
        with zipfile.ZipFile(WHEEL_PATH) as wheel:
            metadata_files = [
                name
                for name in wheel.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            self.assertEqual(len(metadata_files), 1)
            metadata = Parser().parsestr(wheel.read(metadata_files[0]).decode())

        requirements = metadata.get_all("Requires-Dist", [])
        for package, expected_url in EXPECTED_REQUIREMENTS[PLATFORM].items():
            matching = [
                requirement
                for requirement in requirements
                if _normalize_package_name(requirement.partition("@")[0].strip())
                == package
            ]
            self.assertEqual(len(matching), 1, package)
            self.assertEqual(
                matching[0].partition("@")[2].strip(), expected_url, package
            )

        self.assertNotIn("torch==2.6.0+cu126", requirements)


if __name__ == "__main__":
    unittest.main()
