load("//rtp_llm/test/utils:cuda13_sm100_test.bzl", "cuda13_sm100_py_test")

cuda13_sm100_py_test(
    name = "attention_residual_native_test",
    srcs = ["attention_residual_native_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)
