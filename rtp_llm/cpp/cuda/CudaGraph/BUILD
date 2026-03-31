load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "graph_base",
    hdrs = [
        "GraphBase.h",
    ],
    deps = [
        "//rtp_llm/models_py/bindings:op_defs",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "cuda_graph_srcs",
    srcs = [
        "CudaGraphUtils.cc",
        "CudaGraphRunner.cc",
        "CudaGraphPrefill.cc",
        "CudaGraphDecode.cc",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "cuda_graph_hdrs",
    srcs = [
        "GraphBase.h",
        "CudaGraphUtils.h",
        "CudaGraphRunner.h",
    ],
    visibility = ["//visibility:public"],
)
