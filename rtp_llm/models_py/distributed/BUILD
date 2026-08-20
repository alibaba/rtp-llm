load("@arch_config//:arch_select.bzl", "deep_ep_py_deps")

deep_ep_py_deps()

py_library(
    name = "collective_torch",
    srcs = [
        "collective_torch.py",
        "rocm_rccl.py",
        "symm_mem.py",
    ],
    deps = [
        "//rtp_llm:torch",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "deepep_wrapper",
    srcs = ["deepep_wrapper.py"],
    deps = [
        "//rtp_llm:torch",
    ] + select({
        # CUDA13 inherits using_cuda12, so both cuda13 branches must be specialized
        # explicitly or Bazel would select the broader cuda12 alias and hand them the
        # cu12 wheel. x86 has its own deep-ep build; arm still defers it because the
        # available wheels link libcudart.so.12.
        "@//:using_cuda13_x86": ["deep_ep_py"],
        "@//:using_cuda13_arm": [],
        "@//:using_cuda12": ["deep_ep_py"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

py_library(
    name = "moriep_wrapper",
    srcs = ["moriep_wrapper.py"],
    deps = [
        "//rtp_llm:torch",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "user_buffers",
    srcs = ["user_buffers.py"],
    deps = [
        "//rtp_llm:torch",
    ],
    visibility = ["//visibility:public"],
)
