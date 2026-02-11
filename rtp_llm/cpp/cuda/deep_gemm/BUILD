load("//:def.bzl", "cc_test_wrapper", "copts", "cuda_copts", "gen_cpp_code")
load("//bazel:arch_select.bzl", "torch_deps")
load("@rules_cc//examples:experimental_cc_shared_library.bzl", "cc_shared_library")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch", "if_cuda")
load("//rtp_llm/cpp/cuda/deep_gemm:def.bzl", "sm90_cuda_copts", "gen_cu_and_lib", "gen_dispatch_code", "copy_filegroup")
load("//rtp_llm/cpp/cuda/deep_gemm:template.bzl", "NORMAL_GEMM_CASES", "GROUPED_CONTIGUOUS_GEMM_CASES", "GROUPED_MASKED_GEMM_CASES",
                "QWEN_NORMAL_CASES", "QWEN_CONTIGUOUS_CASES", "QWEN_MASKED_CASES", "template_header", "template", "template_tail",
                "dispatch_template_header", "dispatch_template", "dispatch_template_tail")

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

gen_cu_and_lib("deepgemm_dpsk", NORMAL_GEMM_CASES + GROUPED_CONTIGUOUS_GEMM_CASES + GROUPED_MASKED_GEMM_CASES, 1, template_header, template, template_tail, 1)
gen_cu_and_lib("deepgemm_qwen", QWEN_NORMAL_CASES + QWEN_CONTIGUOUS_CASES + QWEN_MASKED_CASES, 1, template_header, template, template_tail, 1)

gen_dispatch_code(
    "deepgemm_dispatch",
    NORMAL_GEMM_CASES + GROUPED_CONTIGUOUS_GEMM_CASES + GROUPED_MASKED_GEMM_CASES +
    QWEN_NORMAL_CASES + QWEN_CONTIGUOUS_CASES + QWEN_MASKED_CASES,
    dispatch_template_header,
    dispatch_template,
    dispatch_template_tail
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
        "//rtp_llm/cpp/core:buffer_torch",
        "@bazel_tools//tools/cpp/runfiles",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "deep_gemm_plugins_impl",
    hdrs = ["DeepGemmPlugin.h", "ConfigUtils.h"],
    srcs = ["DeepGemmPlugin.cpp", ":deepgemm_dispatch"],
    copts = copts(),
    deps = [
        ":deep_gemm_utils",
        ":deepgemm_dpsk_inst",
        ":deepgemm_qwen_inst",
        ":JIT",
        "//rtp_llm/cpp/core:buffer_torch",
        "//rtp_llm/cpp/utils:math_utils",
        "//rtp_llm/cpp/config:static_config",
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
)

cc_test(
    name = "deep_gemm_plugin_test",
    srcs = [
        "test/deep_gemm_plugin_test.cpp",
    ],
    copts = copts(),
    deps = [
        ":deep_gemm_plugins_impl",
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/core:buffer_torch",
        "//rtp_llm/cpp/devices/testing:device_test_utils",
        "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
    ] + torch_deps(),
    timeout = "long",
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "config_utils_test",
    srcs = [
        "test/config_utils_test.cpp",
    ],
    copts = copts(),
    deps = [
        ":deep_gemm_plugins_impl",
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/core:buffer_torch",
        "//rtp_llm/cpp/devices/testing:device_test_utils",
        "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
    ] + torch_deps(),
    timeout = "long",
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)
