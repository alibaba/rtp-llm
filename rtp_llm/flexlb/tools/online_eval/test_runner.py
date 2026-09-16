#!/usr/bin/env python3
"""Select a test class; functional contracts are the default developer loop.

Use --suite workload for sustained scenarios, or --suite all for full regression.
Port leasing and lane planning remain in the existing parallel runner.
"""
import sys
import parallel_runner

if __name__ == "__main__":
    if not any(arg == "--suite" or arg.startswith("--suite=") for arg in sys.argv[1:]):
        sys.argv.extend(["--suite", "functional"])
    raise SystemExit(parallel_runner.main())
