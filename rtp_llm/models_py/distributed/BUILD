load("//bazel:arch_select.bzl", "deep_ep_py_deps")

deep_ep_py_deps()

py_library(
    name = "collective_torch",
    srcs = [
        "collective_torch.py",
        "glm53_collective_gemm.py",
        "pynccl_cp.py",
        "rocm_rccl.py",
        "symm_mem.py",
        "tp_token_shard.py",
        "sequence_parallel.py",
        "zigzag_token_layout.py",
    ],
    deps = [
        "//rtp_llm:torch",
        "//rtp_llm/models_py/triton_kernels:common",
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
    name = "user_buffers",
    srcs = ["user_buffers.py"],
    deps = [
        "//rtp_llm:torch",
    ],
    visibility = ["//visibility:public"],
)
