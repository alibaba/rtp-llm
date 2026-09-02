# SPDX-License-Identifier: Apache-2.0

import re
import sys
import unittest
import zipfile
from email.parser import Parser
from pathlib import Path
from typing import Dict, FrozenSet, Sequence

PLATFORM_DIRECT_URL_PACKAGES: Dict[str, FrozenSet[str]] = {
    "cuda12_9_x86": frozenset(
        {"torch", "torchvision", "fast-safetensors", "fastsafetensors"}
    ),
    "cuda13_x86": frozenset(
        {
            "torch",
            "torchvision",
            "deep-gemm",
            "flash-mla",
            "rtp-kernel",
            "fast-safetensors",
            "fastsafetensors",
        }
    ),
    "cuda13_arm": frozenset(
        {
            "torch",
            "torchvision",
            "deep-gemm",
            "flash-mla",
            "rtp-kernel",
            "fast-safetensors",
            "fastsafetensors",
            "tilelang",
            "z3-solver",
        }
    ),
}


def _normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _locked_direct_urls(lock_path: Path) -> Dict[str, str]:
    lines = lock_path.read_text().splitlines()
    result: Dict[str, str] = {}
    for index, line in enumerate(lines[:-1]):
        requirement = re.match(r"^([A-Za-z0-9_.-]+) @ (\S+) \\$", line)
        if requirement is None:
            continue
        hash_line = re.match(
            r"^\s+--hash=sha256:([0-9a-f]{64})(?: \\)?$", lines[index + 1]
        )
        if hash_line is None:
            raise AssertionError(
                f"direct requirement lacks one adjacent SHA256: {line!r}"
            )
        package = _normalize_package_name(requirement.group(1))
        result[package] = f"{requirement.group(2)}#sha256={hash_line.group(1)}"
    return result


def _wheel_direct_urls(wheel_path: Path) -> Dict[str, str]:
    with zipfile.ZipFile(wheel_path) as wheel:
        metadata_files = [
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise AssertionError(
                f"expected one wheel METADATA file, got {metadata_files!r}"
            )
        metadata = Parser().parsestr(wheel.read(metadata_files[0]).decode())

    result: Dict[str, str] = {}
    for requirement in metadata.get_all("Requires-Dist", []):
        package, separator, url = requirement.partition("@")
        if not separator:
            continue
        normalized = _normalize_package_name(package.strip())
        if normalized in result:
            raise AssertionError(f"duplicate direct requirement: {normalized}")
        result[normalized] = url.strip()
    return result


class RtpLlmWheelMetadataTest(unittest.TestCase):
    platform = ""
    wheel_path = Path()
    lock_path = Path()

    def test_direct_requirements_match_platform_lock_file(self) -> None:
        expected_names = PLATFORM_DIRECT_URL_PACKAGES[self.platform]
        locked_urls = _locked_direct_urls(self.lock_path)
        missing_from_lock = expected_names - locked_urls.keys()
        self.assertFalse(missing_from_lock, sorted(missing_from_lock))

        expected = {name: locked_urls[name] for name in expected_names}
        self.assertEqual(_wheel_direct_urls(self.wheel_path), expected)


def main(argv: Sequence[str]) -> None:
    if len(argv) != 4:
        raise SystemExit(f"usage: {argv[0]} <platform> <lock-file> <rtp-llm-wheel>")
    RtpLlmWheelMetadataTest.platform = argv[1]
    RtpLlmWheelMetadataTest.lock_path = Path(argv[2])
    RtpLlmWheelMetadataTest.wheel_path = Path(argv[3])
    unittest.main(argv=[argv[0]])


if __name__ == "__main__":
    main(sys.argv)
