load("@arch_config//:arch_select.bzl", "deep_ep_py_deps")

deep_ep_py_deps()

py_library(
    name = "collective_torch",
    srcs = [
        "collective_torch.py",
        "flashinfer_all_reduce.py",
        "rocm_rccl.py",
        "symm_mem.py",
        "tp_fp8_all_reduce.py",
    ],
    deps = [
        "//rtp_llm:torch",
        "//rtp_llm/models_py/kernels/cuda/low_precision_all_reduce:tp_fp8_all_reduce",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "deepep_wrapper",
    srcs = ["deepep_wrapper.py"],
    deps = [
        "//rtp_llm:torch",
    ] + select({
        "@//:using_cuda12":["deep_ep_py"],
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
