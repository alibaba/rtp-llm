"""Bazel entry point for the K3 Prefill performance and PD services."""

import importlib.util
import os
import sys

ops_overlay = os.environ.get("OPS_OVERLAY")
if ops_overlay:
    sys.path.insert(0, os.path.realpath(ops_overlay))

if os.environ.get("KIMI_K3_REQUIRE_DEEP_EP") == "1":
    # Do not import DeepEP in this top-level entry point. multiprocessing
    # re-executes it for Frontend/Dash workers, which do not use DeepEP, and
    # importing the native extension installs a fatal SIGPIPE handler in those
    # processes. Resolve both the package and extension without executing them;
    # Backend workers import DeepEP normally when they initialize MoE.
    deep_ep_spec = importlib.util.find_spec("deep_ep")
    deep_ep_cpp_spec = importlib.util.find_spec("deep_ep_cpp")
    if deep_ep_spec is None or deep_ep_cpp_spec is None:
        raise ModuleNotFoundError(
            "KIMI_K3_REQUIRE_DEEP_EP=1 but DeepEP package/extension is unavailable"
        )
    print(
        "DeepEP runtime: "
        f"{os.path.realpath(deep_ep_spec.origin or '')}; "
        f"{os.path.realpath(deep_ep_cpp_spec.origin or '')}",
        flush=True,
    )

from rtp_llm.start_server import main

if __name__ == "__main__":
    main()
