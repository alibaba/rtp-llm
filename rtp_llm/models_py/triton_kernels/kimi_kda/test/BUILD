load("//bazel:arch_select.bzl", "requirement")

requirement(["torch", "triton"])

py_test(
    name = "glm5_kda_ops_test",
    srcs = ["glm5_kda_ops_test.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        ":torch",
        ":triton",
    ],
    visibility = ["//visibility:public"],
)
