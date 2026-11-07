"""Bazel entrypoint for resource-aware Qwen4 component and integration tests."""

import os


def _run():
    # Hide devices before models_py package initialization can probe CUDA.
    if os.environ.get("GPU_COUNT") == "0":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from rtp_llm.models_py.modules.qwen4_exp.test.support.test_runner import main

    main()


if __name__ == "__main__":
    _run()
