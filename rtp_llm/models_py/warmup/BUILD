load("//bazel:arch_select.bzl", "requirement")

deep_gemm = ["deep_gemm"]
requirement(deep_gemm)


py_library(
    name = "warmup",
    srcs = glob([
        "*.py",
    ]),
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py:kernels",
        "//rtp_llm:utils",
        "//rtp_llm:ops",
    ] + select({
        "@//:using_cuda12": deep_gemm,
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)
