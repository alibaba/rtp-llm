"""Run a Bazel pytest target and reject a successful run with no test execution."""

import argparse
import os
import sys
import tempfile
from typing import Optional

from rtp_llm.utils.jit_cache_env import configure_cache_env


class ExecutionCount:
    def __init__(self):
        self.passed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call" and report.passed:
            self.passed += 1

    def pytest_sessionfinish(self, session, exitstatus):
        print(
            f"[bazel-pytest] collected={session.testscollected} "
            f"passed={self.passed} failed={session.testsfailed}"
        )
        if exitstatus == 0 and self.passed == 0:
            print("ERROR: no test body passed (empty or entirely skipped target)")
            session.exitstatus = 1


def _parse_cuda_capability(spec: str) -> tuple[int, Optional[int]]:
    parts = spec.strip().split(".")
    if (
        len(parts) not in (1, 2)
        or not all(part.isascii() and part.isdigit() for part in parts)
    ):
        raise ValueError(f"Invalid --cuda-capability specification: {spec!r}")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) == 2 else None
    if major >= 100:
        raise ValueError(f"Invalid --cuda-capability specification: {spec!r}")
    return major, minor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-devices", type=int, default=0)
    parser.add_argument("--cuda-capability", default=None)
    parser.add_argument("test_file")
    options, pytest_args = parser.parse_known_args()

    if options.cuda_devices < 0:
        raise ValueError(
            f"--cuda-devices must be non-negative, got {options.cuda_devices}"
        )
    if options.cuda_capability is not None and options.cuda_devices <= 0:
        raise ValueError("--cuda-capability requires --cuda-devices > 0")

    # DeepGEMM may import before the model's HOME fallback. NVCC needs an
    # absolute cache path because its compiler subprocess can change directory.
    cache_root = os.environ.get("TEST_TMPDIR") or tempfile.gettempdir()
    configure_cache_env(
        "DG_JIT_CACHE_DIR",
        os.path.abspath(
            os.environ.get("DG_JIT_CACHE_DIR") or os.path.join(cache_root, "deep_gemm")
        ),
    )

    if options.cuda_devices > 0:
        import torch

        cap_spec = options.cuda_capability if options.cuda_capability is not None else "10"
        req_major, req_minor = _parse_cuda_capability(cap_spec)
        if not torch.version.cuda or int(torch.version.cuda.split(".")[0]) != 13:
            raise RuntimeError(f"CUDA 13 runtime required, got {torch.version.cuda}")
        if torch.cuda.device_count() < options.cuda_devices:
            raise RuntimeError(
                f"requires {options.cuda_devices} CUDA devices, "
                f"found {torch.cuda.device_count()}"
            )
        for index in range(options.cuda_devices):
            capability = torch.cuda.get_device_capability(index)
            if capability[0] != req_major or (
                req_minor is not None and capability[1] != req_minor
            ):
                required_minor = "x" if req_minor is None else str(req_minor)
                raise RuntimeError(
                    f"DSV4 CUDA13 tests require SM{req_major}{required_minor} devices, "
                    f"found SM{capability[0]}{capability[1]} on device {index}"
                )

    # Machine-installed plugins must not change test collection or dependencies.
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    import pytest

    args = [options.test_file, "-ra", "-p", "no:cacheprovider"] + pytest_args
    xml_path = os.environ.get("XML_OUTPUT_FILE")
    if xml_path:
        args += ["--junitxml", xml_path]
    return pytest.main(args, plugins=[ExecutionCount()])


if __name__ == "__main__":
    sys.exit(main())
