"""Keep FP8 native JIT artifacts isolated from previous Bazel test actions."""

import os
import runpy

if __name__ == "__main__":
    # Set this before imports initialize FlashInfer's JIT paths. Shared worker
    # caches can contain native modules incompatible with the current TVM FFI.
    if test_tmpdir := os.environ.get("TEST_TMPDIR"):
        os.environ["FLASHINFER_WORKSPACE_BASE"] = os.path.join(
            test_tmpdir, "flashinfer"
        )
    runpy.run_module(
        "rtp_llm.models_py.modules.factory.linear.impl.cuda.test.fp8_linear_test",
        run_name="__main__",
        alter_sys=True,
    )
