py_library(
    name = "collective_torch",
    srcs = [
        "collective_torch.py",
        "fp8_collective_projection.py",
        "flashinfer_all_reduce.py",
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
        "@//:using_cuda13_x86": ["//rtp_llm/models_py:deep_ep"],
        "@//:using_cuda13_arm": ["//rtp_llm/models_py:deep_ep"],
        "@//:using_cuda12": ["//rtp_llm/models_py:deep_ep"],
        "@//:using_ppu": ["//rtp_llm/models_py:deep_ep"],
        "@//:using_rocm": [],
        "@//:using_cpu": [],
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
