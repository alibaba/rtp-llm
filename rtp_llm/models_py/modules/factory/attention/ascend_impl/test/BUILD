# Pure-torch parity tests for the Ascend RoPE helper: CPU-only, no NPU needed.

py_test(
    name = "test_ascend_rope",
    srcs = ["test_ascend_rope.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm/models_py:modules",
    ],
)
