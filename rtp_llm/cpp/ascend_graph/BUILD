load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

# Ascend ACL Graph core (decode-only).
#
# Mirrors //rtp_llm/cpp/cuda_graph:cuda_graph_impl but wraps
# c10_npu::NPUGraph instead of at::cuda::CUDAGraph. Built only on Ascend;
# the TU compiles to no-op stubs on other platforms so it is safe to depend
# on unconditionally.

cc_library(
    name = "ascend_graph_hdrs_lib",
    hdrs = [
        "ascend_graph_device_shims.h",
        "ascend_graph_runner.h",
        "ascend_graph_utils.h",
    ],
    deps = [
        "//rtp_llm/cpp/cuda_graph:cuda_graph_base",
        "//rtp_llm/cpp/cuda_graph:cuda_graph_hdrs_lib",
        "//rtp_llm/models_py/bindings:op_defs",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ascend_graph",
    hdrs = [
        "ascend_graph_device_shims.h",
        "ascend_graph_runner.h",
        "ascend_graph_utils.h",
    ],
    srcs = [
        "ascend_graph_device_shims.cc",
        "ascend_graph_runner.cc",
    ],
    deps = torch_deps() + [
        ":ascend_graph_hdrs_lib",
        "//rtp_llm/cpp/cuda_graph:cuda_graph_base",
        "//rtp_llm/cpp/cuda_graph:cuda_graph_hdrs_lib",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:profiling_scope",
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
    ] + select({
        "@//:using_ascend": [
            "@local_config_ascend//ascend:ascend_headers",
        ],
        "//conditions:default": [],
    }),
    # CANN's acl_sk.h contains a nameless "typedef struct {...};" that emits
    # a flagless gcc warning (cannot be disabled per-flag); keep warnings
    # non-fatal for this target.
    copts = copts() + select({
        "@//:using_ascend": ["-Wno-error"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
