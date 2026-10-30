"""Keep CUDA wheel inputs aligned with their hash-pinned locks."""

import re
import unittest
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

CUDA13_REQUIREMENTS = (
    ("requirements_torch_gpu_cuda13.txt", "requirements_lock_torch_gpu_cuda13.txt"),
    ("requirements_cuda13_arm.txt", "requirements_lock_cuda13_arm.txt"),
)


class RequirementsSourcesTest(unittest.TestCase):
    def test_indexed_compute_wheels_preserve_locked_integrity(self):
        directory = Path(__file__).resolve().parent
        for source, lock in CUDA13_REQUIREMENTS:
            requirements = (directory / source).read_text()
            locked = (directory / lock).read_text()
            for package in (
                "deep-gemm",
                "flash-mla",
                "cuda-linear-attention",
                "flash-linear-attention",
            ):
                with self.subTest(source=source, package=package):
                    pins = re.findall(
                        rf"^{re.escape(package)}==\S+$", requirements, re.MULTILINE
                    )
                    self.assertEqual(len(pins), 1)
                    self.assertRegex(
                        locked,
                        rf"(?m)^{re.escape(pins[0])} \\\n    --hash=sha256:[0-9a-f]{{64}}",
                    )

    def test_direct_compute_wheels_preserve_locked_integrity(self):
        directory = Path(__file__).resolve().parent
        for source, lock in CUDA13_REQUIREMENTS:
            locked = (directory / lock).read_text()
            wheels = [
                line.strip()
                for line in (directory / source).read_text().splitlines()
                if line.startswith(("http://", "https://"))
                and urlsplit(line.strip()).path.endswith(".whl")
            ]
            for wheel in wheels:
                with self.subTest(
                    source=source,
                    wheel=unquote(urlsplit(wheel).path).rsplit("/", 1)[-1],
                ):
                    parsed = urlsplit(wheel)
                    self.assertEqual(parsed.scheme, "https")
                    hashes = parse_qs(parsed.fragment).get("sha256", [])
                    self.assertEqual(len(hashes), 1)
                    self.assertRegex(hashes[0], r"^[0-9a-f]{64}$")
                    filename = unquote(parsed.path).rsplit("/", 1)[-1]
                    entry = next(
                        entry
                        for entry in re.split(r"\n(?=\S)", locked)
                        if filename in unquote(entry)
                    )
                    self.assertIn("--hash=sha256:" + hashes[0], entry)


if __name__ == "__main__":
    unittest.main()
