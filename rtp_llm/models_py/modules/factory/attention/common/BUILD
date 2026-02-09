load("//bazel:arch_select.bzl", "requirement")

requirement(["flashinfer-python"])

package(default_visibility = ["//visibility:public"])

py_library(
    name = "common",
    srcs = glob(["*.py"]),
    deps = [
        "//rtp_llm:ops",
    ] + select({
        "@//:using_cuda12": ["flashinfer-python"],
        "//conditions:default": [],
    }),
)

