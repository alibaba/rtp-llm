load("@arch_config//:arch_select.bzl", "requirement")

requirement([
    "triton",
    "torch",
    "numpy",
])

py_test(
    name = "gated_conv_test",
    srcs = [
        "gated_conv_test.py",
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:gated_causal_conv1d",
        "//rtp_llm/models_py/triton_kernels:causal_conv1d",
        "//rtp_llm/test/utils:test_util",
    ] + [
        ":numpy",
        ":torch",
        ":triton",
    ],
    visibility = ["//visibility:public"],
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)
