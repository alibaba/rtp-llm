load("@arch_config//:arch_select.bzl", "requirement")

requirement([
    "triton",
    "torch",
    "numpy",
])

py_test(
    name = "test_casual_conv1d_prefill",
    srcs = [
        "test_casual_conv1d_prefill.py",
    ],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:causal_conv1d",
        "//rtp_llm/test/utils:test_util",
    ] + [
        ":numpy",
        ":torch",
        ":triton",
    ],
)

py_test(
    name = "test_casual_conv1d_decode",
    srcs = [
        "test_casual_conv1d_decode.py",
    ],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:causal_conv1d",
        "//rtp_llm/test/utils:test_util",
    ] + [
        ":numpy",
        ":torch",
        ":triton",
    ],
)
