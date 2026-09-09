"""Run a Bazel pytest target and reject a successful run with no test execution."""

import argparse
import os
import sys
import tempfile


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

        if not torch.version.cuda or int(torch.version.cuda.split(".")[0]) != 13:
            raise RuntimeError(f"CUDA 13 runtime required, got {torch.version.cuda}")
        if torch.cuda.device_count() < options.cuda_devices:
            raise RuntimeError(
                f"requires {options.cuda_devices} CUDA devices, "
                f"found {torch.cuda.device_count()}"
            )
        for index in range(options.cuda_devices):
            if torch.cuda.get_device_capability(index)[0] != 10:
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
