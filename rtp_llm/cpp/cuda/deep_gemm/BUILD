load("//:def.bzl", "cc_test_wrapper", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")
load("//rtp_llm/cpp/cuda/deep_gemm:def.bzl", "sm90_cuda_copts", "copy_filegroup")

cc_test = cc_test_wrapper

cc_library(
    name = "deep_gemm_utils",
    hdrs = ["utils.h", "JIT.h"],
    srcs = ["JIT.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "@boringssl//:ssl",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "deepgemm_hdrs",
    hdrs = select({
        "@//:using_cuda12": glob([
            "deep_gemm_template.h",
            "include/*.cuh"
        ]),
        "//conditions:default": [],
    }),
    deps = [
        ":deep_gemm_utils",
    ],
    copts = sm90_cuda_copts,
    visibility = ["//visibility:public"],
)

filegroup(
    name = "deep_gemm_jit_srcs",
    srcs = ["utils.h", "JIT.h", "JIT.cc"],
    visibility = ["//visibility:public"],
)

copy_filegroup(
    name = "cutlass_hdrs",
    srcs = ["@cutlass//:cutlass_origin"],
    out = "cutlass_hdr"
)

filegroup(
    name = "jit_includes",
    srcs = [
        "JITRuntime.h",
        "JITRuntime.cc",
        ":deep_gemm_jit_srcs",
        ":cutlass_hdrs",
        "interleave_ffma.py"
    ] + glob([
        "include/*.cuh"
    ]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "JIT",
    hdrs = ["JITRuntime.h"],
    srcs = ["JITRuntime.cc"],
    data = [":jit_includes"],
    copts = copts(),
    deps = [
        ":deep_gemm_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",

        "@bazel_tools//tools/cpp/runfiles",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "deep_gemm_plugins_impl",
    hdrs = ["DeepGemmPlugin.h", "ConfigUtils.h"],
    srcs = ["DeepGemmPlugin.cpp"],
    copts = copts(),
    deps = [
        ":deep_gemm_utils",
        ":JIT",
        "//rtp_llm/cpp/utils:math_utils",
        "//rtp_llm/cpp/config:static_config",
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
)

# deep_gemm_plugin_test and config_utils_test removed (require H20 hardware)
