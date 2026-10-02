load("@arch_config//:arch_select.bzl", "requirement")

requirement([
    "triton",
    "torch",
    "numpy",
    "einops",
    "packaging",
])

py_test(
    name = "test_kda_ops",
    srcs = [
        "test_kda_ops.py",
    ],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
    ] + [
        ":einops",
        ":numpy",
        ":packaging",
        ":torch",
        ":triton",
    ],
)
