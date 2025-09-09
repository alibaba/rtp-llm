package(default_visibility = ["//rtp_llm:__subpackages__"])

load("//bazel:arch_select.bzl", "deep_ep_py_deps")

deep_ep_py_deps()

py_library(
    name = "process_group_state",
    srcs = ["process_group_state.py"],
    deps = [
        "//rtp_llm:torch",
    ],
)

py_library(
    name = "deep_ep",
    srcs = ["deep_ep.py"],
    deps = [
        "//rtp_llm:torch",
    ] + select({
        "@//:using_cuda12":["deep_ep_py"],
        "//conditions:default": [],
    }),
)

py_library(
    name = "distribute",
    srcs = glob(["*.py"]),
    deps = [
        ":process_group_state",
        ":deep_ep",
    ],
)
