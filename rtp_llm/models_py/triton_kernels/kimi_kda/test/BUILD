load("@arch_config//:arch_select.bzl", "requirement")

requirement(["triton", "torch", "numpy", "einops", "packaging"])

py_test(
    name = "test_kda_ops",
    srcs = [
        "test_kda_ops.py",
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
    ] + [":triton", ":torch", ":numpy", ":einops", ":packaging"],
    visibility = ["//visibility:public"],
    exec_properties = {"gpu": "H20"},
)
