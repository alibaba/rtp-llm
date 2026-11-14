"""Bazel entry for read-only perf-test golden validation."""

import argparse
from typing import Optional, Sequence

from rtp_llm.test.perf_test.regression.golden import load_result, validate_golden


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a perf-test result golden")
    parser.add_argument("--result", required=True)
    parser.add_argument("--golden", required=True)
    args = parser.parse_args(argv)
    validate_golden(load_result(args.result), args.golden)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
