"""Validate the installed adapter and prepare its image regression tests."""

import argparse
import importlib
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--test-root", type=Path)
    args = parser.parse_args()
    for name in (
        "navi_sched",
        "navi_sched.adapters.vllm.VLLMCoreCallbacks",
        "navi_sched.adapters.vllm.VLLMEngineTransport",
        "navi_sched.adapters.vllm.VLLMWorkerDiagnostics",
    ):
        importlib.import_module(name)
    actual = Path("/opt/navi_sched/build-info/NAVI_SCHED_COMMIT").read_text().strip()
    if actual != args.expected_commit:
        raise RuntimeError("installed adapter source commit does not match this CI run")
    if args.test_root is not None:
        target = args.test_root / "navi_integration"
        target.mkdir(parents=True, exist_ok=True)
        for name in (
            "test_navi_same_version_upgrade.py",
            "test_install_navi_sched_wheel.py",
            "install_navi_sched_wheel.sh",
        ):
            shutil.copy2(Path(__file__).with_name(name), target / name)
    print("Verified installed NaviSched source:", actual, flush=True)


if __name__ == "__main__":
    main()
