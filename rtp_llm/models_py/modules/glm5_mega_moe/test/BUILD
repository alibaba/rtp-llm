load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "mega_moe_unit_test",
    srcs = ["mega_moe_unit_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    timeout = "short",
)
