py_library(
    name = "kernel_tuning",
    srcs = glob([
        "*.py",
        "aiter/*.py",
    ]),
    data = glob([
        "aiter/configs/*.csv",
    ]),
    deps = [
        "//rtp_llm:torch",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "aiter_fmoe_test",
    srcs = ["test/aiter_fmoe_test.py"],
    deps = [":kernel_tuning"],
)

py_test(
    name = "registry_test",
    srcs = ["test/registry_test.py"],
    deps = [":kernel_tuning"],
)
