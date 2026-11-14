"""Keep direct CUDA wheel inputs authenticated and aligned with their locks."""

import re
import unittest
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit


class RequirementsSourcesTest(unittest.TestCase):
    def test_direct_compute_wheels_preserve_locked_integrity(self):
        directory = Path(__file__).resolve().parent
        for source, lock in (
            (
                "requirements_torch_gpu_cuda13.txt",
                "requirements_lock_torch_gpu_cuda13.txt",
            ),
            ("requirements_cuda13_arm.txt", "requirements_lock_cuda13_arm.txt"),
        ):
            locked = (directory / lock).read_text()
            wheels = [
                line.strip()
                for line in (directory / source).read_text().splitlines()
                if line.startswith(("http://", "https://"))
                and urlsplit(line.strip()).path.endswith(".whl")
            ]
            self.assertTrue(wheels, source)
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
