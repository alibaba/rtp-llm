load("//bazel:arch_select.bzl", "requirement")

requirement(["torch", "triton"])

py_test(
    name = "glm5_kda_ops_test",
    srcs = ["glm5_kda_ops_test.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:causal_conv1d",
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        ":torch",
        ":triton",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        'gpu': 'L20A_DEV',
        'gpu_count': '1',
    },
    tags = ["manual"],
    visibility = ["//visibility:public"],
)
