"""Bazel entry for explicit perf-test golden generation."""

import argparse
from typing import Optional, Sequence

from rtp_llm.test.perf_test.regression.golden import load_result, write_golden


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a perf-test result golden")
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    write_golden(load_result(args.result), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
