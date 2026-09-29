"""Run a Bazel pytest target and reject a successful run with no test execution."""

import argparse
import os
import sys
import tempfile


def _expected_compute_capability():
    value = os.environ.get("EXPECTED_CUDA_COMPUTE_CAPABILITY")
    if not value:
        return None
    parts = value.split(".")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise RuntimeError(
            "EXPECTED_CUDA_COMPUTE_CAPABILITY must be '<major>.<minor>', "
            f"got {value!r}"
        )
    return int(parts[0]), int(parts[1])


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-devices", type=int, default=0)
    parser.add_argument("test_file")
    options, pytest_args = parser.parse_known_args()

    # DeepGEMM may import before the model's HOME fallback. NVCC needs an
    # absolute cache path because its compiler subprocess can change directory.
    cache_root = os.environ.get("TEST_TMPDIR") or tempfile.gettempdir()
    os.environ["DG_JIT_CACHE_DIR"] = os.path.abspath(
        os.environ.get("DG_JIT_CACHE_DIR") or os.path.join(cache_root, "deep_gemm")
    )

    if options.cuda_devices:
        import torch

        expected_cuda_major = int(os.environ.get("EXPECTED_CUDA_MAJOR") or "13")
        expected_capability = _expected_compute_capability()
        actual_cuda = torch.version.cuda
        actual_cuda_major = int(actual_cuda.split(".")[0]) if actual_cuda else None
        if actual_cuda_major != expected_cuda_major:
            raise RuntimeError(
                f"CUDA {expected_cuda_major} runtime required, got {actual_cuda}"
            )
        if torch.cuda.device_count() < options.cuda_devices:
            raise RuntimeError(
                f"requires {options.cuda_devices} CUDA devices, "
                f"found {torch.cuda.device_count()}"
            )
        for index in range(options.cuda_devices):
            actual_capability = torch.cuda.get_device_capability(index)
            if expected_capability is not None:
                if actual_capability != expected_capability:
                    raise RuntimeError(
                        f"CUDA device {index} requires compute capability "
                        f"{expected_capability[0]}.{expected_capability[1]}, got "
                        f"{actual_capability[0]}.{actual_capability[1]}"
                    )
            elif actual_capability[0] != 10:
                raise RuntimeError("DSV4 CUDA13 tests require Blackwell devices")

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
