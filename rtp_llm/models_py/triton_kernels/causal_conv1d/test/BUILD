load("//bazel:arch_select.bzl", "requirement")

requirement(["triton", "torch", "numpy"])

py_test(
    name = "test_casual_conv1d_prefill",
    srcs = [
        "test_casual_conv1d_prefill.py"
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:causal_conv1d",
        "//rtp_llm/test/utils:test_util",

    ] + [":triton", ":torch", ":numpy"],
    visibility = ["//visibility:public"],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "test_casual_conv1d_decode",
    srcs = [
        "test_casual_conv1d_decode.py"
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:causal_conv1d",
        "//rtp_llm/test/utils:test_util",
    ] + [":triton", ":torch", ":numpy"],
    visibility = ["//visibility:public"],
    exec_properties = {'gpu':'H20'},
)
py_test(
    name = "test_causal_conv1d_int64",
    srcs = ["test_causal_conv1d_int64.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:causal_conv1d",
        ":numpy",
        ":torch",
        ":triton",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "L20A_DEV", "gpu_count": "1"},
    tags = ["manual"],
    timeout = "long",
)
