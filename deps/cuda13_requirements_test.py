"""Offline checks for the SM120 dependency contract and generated lock."""

import shlex
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename

ROOT = Path(__file__).resolve().parent


def requirements(path):
    result = {}
    for raw in path.read_text().splitlines():
        line = raw.strip().removesuffix("\\").strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            result.update(requirements(path.parent / shlex.split(line)[1]))
            continue
        if line.startswith("--"):
            continue
        if line.startswith(("http://", "https://")):
            name, _, _, _ = parse_wheel_filename(
                Path(unquote(urlsplit(line).path)).name
            )
            line = f"{name} @ {line}"
        requirement = Requirement(line)
        result[canonicalize_name(requirement.name)] = requirement
    return result


def identity(requirement):
    return (
        unquote(requirement.url) if requirement.url else None,
        str(requirement.specifier),
        str(requirement.marker),
        sorted(requirement.extras),
    )


# Provider and compiler packages whose pins deliberately differ.
_PROFILE_PINS = {"deep-gemm", "flashinfer-python", "nvidia-cutlass-dsl"}
# Requested or locked by one profile only.
_REGULAR_ONLY = {"flashinfer-cubin", "nvidia-cutlass-dsl-libs-cu13"}
_DSV4_ONLY_INPUTS = {
    "backports-strenum",
    "cuda-core",
    "cuda-pathfinder",
    "cuda-tile",
    "nccl4py",
    "nvidia-cudnn-frontend",
    "nvidia-cutlass-dsl-libs-base",
}
_DSV4_ONLY_LOCKED = {"backports-strenum", "cuda-core", "nccl4py"}


class Cuda13RequirementsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generic = requirements(ROOT / "requirements_torch_gpu_cuda13.txt")
        cls.dsv4 = requirements(ROOT / "requirements_torch_gpu_cuda13_sm120_dsv4.txt")
        cls.generic_lock = requirements(ROOT / "requirements_lock_torch_gpu_cuda13.txt")
        cls.dsv4_lock = requirements(
            ROOT / "requirements_lock_torch_gpu_cuda13_sm120_dsv4.txt"
        )
        cls.overrides = requirements(ROOT / "requirements_cuda13_overrides.txt")

    def test_shared_inputs_keep_the_same_pins(self):
        # The sparse-attention provider uses JIT kernels instead of the generic
        # cubin package; the two provider versions must remain separate.
        self.assertEqual(self.generic.keys() - self.dsv4.keys(), _REGULAR_ONLY)
        self.assertEqual(self.dsv4.keys() - self.generic.keys(), _DSV4_ONLY_INPUTS)
        for name in self.generic.keys() & self.dsv4.keys():
            if name in _PROFILE_PINS:
                continue
            with self.subTest(package=name):
                self.assertEqual(
                    identity(self.generic[name]), identity(self.dsv4[name])
                )

    def test_lock_keeps_the_generic_dependency_closure(self):
        self.assertEqual(
            self.generic_lock.keys() - self.dsv4_lock.keys(), _REGULAR_ONLY
        )
        self.assertEqual(
            self.dsv4_lock.keys() - self.generic_lock.keys(), _DSV4_ONLY_LOCKED
        )
        # Provider-specific runtime dependencies are deliberate. Other common
        # packages must not drift merely because a second lock is generated.
        overrides = {
            "cuda-pathfinder",
            "cuda-tile",
            "deep-gemm",
            "flashinfer-python",
            "nvidia-cudnn-frontend",
            "nvidia-cutlass-dsl",
            "nvidia-cutlass-dsl-libs-base",
        }
        for name in self.generic_lock.keys() & self.dsv4_lock.keys():
            if name in overrides:
                continue
            with self.subTest(package=name):
                self.assertEqual(
                    identity(self.generic_lock[name]), identity(self.dsv4_lock[name])
                )

    def test_sm120_lock_satisfies_inputs_and_resolution_overrides(self):
        for name, requested in (self.dsv4 | self.overrides).items():
            with self.subTest(package=name):
                self.assertIn(name, self.dsv4_lock)
                locked = self.dsv4_lock[name]
                if requested.url:
                    self.assertEqual(unquote(requested.url), unquote(locked.url or ""))
                elif requested.specifier:
                    if locked.url:
                        _, version, _, _ = parse_wheel_filename(
                            Path(unquote(urlsplit(locked.url).path)).name
                        )
                    else:
                        pins = list(locked.specifier)
                        self.assertEqual(len(pins), 1)
                        self.assertEqual(pins[0].operator, "==")
                        version = pins[0].version
                    self.assertIn(version, requested.specifier)


if __name__ == "__main__":
    unittest.main()
