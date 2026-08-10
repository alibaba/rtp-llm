"""Bazel entry point for the K3 Prefill performance and PD services."""

import os
import sys

ops_overlay = os.environ.get("OPS_OVERLAY")
if ops_overlay:
    sys.path.insert(0, os.path.realpath(ops_overlay))

from rtp_llm.start_server import main

if __name__ == "__main__":
    main()
