load("//:def.bzl", "copts", "cuda_copts", "gen_cpp_code")
load("//bazel:arch_select.bzl", "torch_deps")
load("@rules_cc//examples:experimental_cc_shared_library.bzl", "cc_shared_library")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch", "if_cuda")
load("//rtp_llm/cpp/deep_gemm:def.bzl", "sub_lib", "sm90_cuda_copts", "gen_cu_and_lib", "gen_dispatch_code", "copy_filegroup")
load("//rtp_llm/cpp/deep_gemm:template.bzl", "NORMAL_GEMM_CASES", "GROUPED_CONTIGUOUS_GEMM_CASES", "GROUPED_MASKED_GEMM_CASES",
                "QWEN_NORMAL_CASES", "QWEN_CONTIGUOUS_CASES", "QWEN_MASKED_CASES", "template_header", "template", "template_tail",
                "dispatch_template_header", "dispatch_template", "dispatch_template_tail")

cc_library(
    name = "deepgemm_hdrs",
    hdrs = [
        "utils.h",
    ] + select({
        "@//:using_cuda12": glob([
            "deep_gemm_template.h",
            "include/*.cuh"
        ]),
        "//conditions:default": [],
    }),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "@cutlass//:cutlass",
    ] + torch_deps(),
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
    name = "deep_gemm_jit_include",
    srcs = [
        "utils.h",
        "deep_gemm_template.h",
        "interleave_ffma.py",
        "JIT.h",
        "JIT.cc"
    ] + glob([
        "include/*.cuh"
    ]),
)

copy_filegroup(
    name = "cutlass_hdrs",
    srcs = ["@cutlass//:cutlass_origin"],
    out = "cutlass_hdr"
)

filegroup(
    name = "jit_includes",
    srcs = [
        ":deep_gemm_jit_include",
        ":cutlass_hdrs"
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "JIT",
    hdrs = ["JIT.h", "utils.h"],
    srcs = ["JIT.cc"],
    data = [":jit_includes"],
    copts = copts(),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "//rtp_llm/cpp/core:buffer_torch",
        "@bazel_tools//tools/cpp/runfiles",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "deep_gemm_plugins_impl",
    hdrs = ["DeepGemmPlugin.h", "utils.h"],
    srcs = ["DeepGemmPlugin.cpp", ":deepgemm_dispatch"],
    copts = copts(),
    deps = [
        ":deepgemm_dpsk_inst",
        ":deepgemm_qwen_inst",
        ":JIT",
        "//rtp_llm/cpp/core:buffer_torch",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "//rtp_llm/cpp/cuda:nvtx",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
)

cc_test(
    name = "deep_gemm_plugin_test",
    srcs = [
        "test/deep_gemm_plugin_test.cpp",
    ],
    copts = copts(),
    linkopts = [
        "-lpython3.10",
        "-ltorch",
        "-lc10",
        "-ltorch_cpu",
        "-ltorch_python",
        "-L/usr/local/cuda/lib64",
        "-lnvToolsExt",
        "-L/usr/local/cuda/extras/CUPTI/lib64",
        "-lcupti",
    ],
    deps = [
        ":deep_gemm_plugins_impl",
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/core:buffer_torch",
        "//rtp_llm/cpp/devices/testing:device_test_utils",
        "//rtp_llm/cpp/devices/cuda_impl:gpu_base",
        "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
    ],
    exec_properties = {'gpu':'H20'},
)
