load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "test_decode",
    srcs = ["test_decode.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
)
