load("//bazel:arch_select.bzl", "deep_ep_py_deps")

deep_ep_py_deps()

py_library(
    name = "deepep_wrapper",
    srcs = ["deepep_wrapper.py", "deepep_initializer.py"],
    deps = [
        "//rtp_llm:torch",
    ] + select({
        "@//:using_cuda12":["deep_ep_py"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)
