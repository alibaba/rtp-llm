"""Strict candidate acceptance on two Blackwell GPUs with CUDA 13.

Run with the RTP test environment and its usual GPU reservation:
  python validate_wo_a_quant_wheel.py --wheel /path/to/deep_gemm.whl \
      --sha256 <expected-sha256> --output /path/to/new-results

Installs only the hash-verified wheel into an isolated directory; other runtime
dependencies come from the invoking Python environment. Does not update shared
dependency pins. A passing ordinary test target is not candidate acceptance.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

GPU_CLASSES = {"WoAQuantCudaTest", "ExternalQuantCudaTest"}


def verify_wheel(wheel, expected):
    digest = hashlib.sha256()
    with wheel.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected.lower():
        raise ValueError(f"Wheel SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def summarize_results(path):
    counts = dict(cpu_passed=0, gpu_passed=0, gpu_skipped=0, failed=0)
    passed_classes = set()
    for case in ET.parse(path).iter("testcase"):
        cls = case.get("classname", "").rsplit(".", 1)[-1]
        gpu = cls in GPU_CLASSES
        if case.find("failure") is not None or case.find("error") is not None:
            counts["failed"] += 1
        elif case.find("skipped") is not None:
            counts["gpu_skipped"] += int(gpu)
        else:
            counts["gpu_passed" if gpu else "cpu_passed"] += 1
            if gpu:
                passed_classes.add(cls)
    counts["accepted"] = (
        passed_classes == GPU_CLASSES
        and counts["gpu_skipped"] == 0
        and counts["failed"] == 0
    )
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    digest = verify_wheel(wheel, args.sha256)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[6]
    report = dict(wheel=str(wheel), sha256=digest, accepted=False)
    with tempfile.TemporaryDirectory(prefix="wo-a-candidate-") as directory:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                "--target",
                directory,
                str(wheel),
            ],
            check=True,
        )
        env = dict(os.environ)
        env.pop("PYTHONOPTIMIZE", None)
        env.update(
            PYTHONPATH=os.pathsep.join(
                [directory, str(repo), env.get("PYTHONPATH", "")]
            ),
            RTP_V41_REQUIRE_EXTERNAL_QUANT="1",
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            PYTEST_ADDOPTS="",
            XML_OUTPUT_FILE=str(output / "junit.xml"),
        )
        # Check the actual import before pytest: an already installed package or
        # a source checkout must not accidentally stand in for this wheel.
        entry = """
import pathlib, runpy, sys
import deep_gemm
expected = pathlib.Path(sys.argv.pop(1)).resolve()
actual = pathlib.Path(deep_gemm.__file__).resolve()
if not actual.is_relative_to(expected):
    raise RuntimeError(f'Imported {actual} instead of candidate at {expected}')
if not hasattr(deep_gemm.epilogue, 'FP8Group32'):
    raise RuntimeError('Candidate missing FP8Group32')
if not hasattr(deep_gemm, 'get_jit_config_fingerprint'):
    raise RuntimeError('Candidate missing JIT fingerprint')
print('Candidate import:', actual, flush=True)
runpy.run_path('rtp_llm/test/pytest/pytest_main.py', run_name='__main__')
"""
        with (output / "tests.log").open("w") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    entry,
                    directory,
                    "--cuda-devices",
                    "2",
                    str(Path(__file__).with_name("test_v41_wo_a_quant.py")),
                    "-o",
                    "addopts=",
                ],
                cwd=repo,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if (output / "junit.xml").exists():
            report.update(summarize_results(output / "junit.xml"))
        report["exit_code"] = result.returncode
        report["accepted"] = report["accepted"] and result.returncode == 0
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
