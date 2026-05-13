"""Cross-platform GPU detection test.

Self-contained (no pip deps), designed to run under pytest on remote workers.

NOTE: Detection logic is intentionally duplicated from device_resource.py so
this test stays fully self-contained using only stdlib + subprocess.
"""

import subprocess
import unittest

_NVIDIA_SMI_PATHS = [
    "nvidia-smi",
    "/usr/local/cuda/bin/nvidia-smi",
    "/usr/local/nvidia/bin/nvidia-smi",
    "/usr/bin/nvidia-smi",
]


class TestDeviceDetection(unittest.TestCase):

    def _detect_nvidia(self):
        """Detect NVIDIA-compatible devices via nvidia-smi.

        Tries multiple known paths because Bazel sandbox workers may not
        have nvidia-smi on PATH.  Rejects output containing "error" to avoid
        false positives from driver wrappers when the driver is absent.
        """
        for smi in _NVIDIA_SMI_PATHS:
            try:
                r = subprocess.run(
                    [smi, "--query-gpu=name", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if r.returncode == 0 and r.stdout.strip():
                    lines = [
                        l.strip() for l in r.stdout.strip().splitlines() if l.strip()
                    ]
                    if lines and "error" not in lines[0].lower():
                        return lines[0], len(lines)
            except FileNotFoundError:
                continue
        return None

    def _detect_rocm(self):
        """Detect AMD GPUs via rocm-smi / rocminfo."""
        try:
            r = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if r.returncode == 0:
                names = []
                for line in r.stdout.strip().splitlines():
                    if "GPU[" in line and ":" in line:
                        parts = line.split(":")
                        if len(parts) >= 3 and parts[-1].strip():
                            names.append(parts[-1].strip())
                if names:
                    return names[0], len(names)
        except FileNotFoundError:
            pass
        try:
            r = subprocess.run(
                ["rocminfo"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if r.returncode == 0:
                names = [
                    line.split(":")[-1].strip()
                    for line in r.stdout.splitlines()
                    if "Marketing Name" in line
                    and line.split(":")[-1].strip() != "Host"
                ]
                if names:
                    return names[0], len(names)
        except FileNotFoundError:
            pass
        return None

    def test_at_least_one_platform_detected(self):
        nvidia = self._detect_nvidia()
        rocm = self._detect_rocm()

        print(f"NVIDIA-compatible: {nvidia}")
        print(f"ROCm:       {rocm}")

        detected = nvidia or rocm
        self.assertIsNotNone(detected, "No GPU/accelerator detected on this worker")

        name, count = detected
        self.assertIsInstance(name, str)
        self.assertGreater(len(name), 0)
        self.assertIsInstance(count, int)
        self.assertGreater(count, 0)
        print(f"\nDetected: {name} x{count}")


if __name__ == "__main__":
    unittest.main()
