load("//:def.bzl", "rocm_copts",)
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "rocm_types_hdr",
    hdrs = [
        "cuda_shims.h",
        "amd_bfloat16.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocm_custom_ar",
    srcs = [
        "custom_ar/custom_ar_comm.cc",
        "custom_ar/quick_ar_comm.cc",
    ],
    hdrs = [
        "custom_ar/custom_ar_comm.h",
        "custom_ar/quick_ar_comm.h",
    ],
    deps = [
        "//rtp_llm/cpp/cuda:nccl_util",
        "//rtp_llm/cpp/core:types",
        "@local_config_rocm//rocm:rocm_headers",
    ] + torch_deps() + select({
        "@//:using_aiter_src": [
            "@aiter_src//:module_custom_all_reduce",
            "@aiter_src//:module_quick_all_reduce",
        ],
        "//conditions:default": ["@aiter//:module_custom_all_reduce", "@aiter//:module_quick_all_reduce"],
    }),
    copts = rocm_copts() + ["-DUSE_ROCM=1"],
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocm_host_utils",
    srcs = glob([
        "*.cc",
        "int4_gemm_kernels/*.cc",
    ], exclude=["rocmMoeWrapper.cc"]),
    hdrs = glob([
        "*.h",
        "int4_gemm_kernels/*.h",
        "*.cuh",
    ], exclude=["rocmMoeWrapper.h"]),
    deps = [
        "@local_config_rocm//rocm:rocm",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:rocblas",
        "@local_config_rocm//rocm:hipblaslt",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/core:allocator",
        "//rtp_llm/cpp/core:types",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@havenask//aios/autil:scope",
    ] + select({
        "@//:using_aiter_src": [
            "@aiter_src//:module_mha_fwd",
        ],
        "//conditions:default": ["@aiter//:module_mha_fwd",],
    }),
    copts = rocm_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)
cc_library(
    name = "rocm",
    deps = [
        ":rocm_host_utils",
    ],
    copts=rocm_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)
