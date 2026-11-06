"""Bazel executable for the unified perf test."""

from rtp_llm.test.perf_test.entry import main

if __name__ == "__main__":
    raise SystemExit(main())
