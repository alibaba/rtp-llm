load("//:def.bzl", "copts", "cuda_copts", "gen_cpp_code")
load("//bazel:arch_select.bzl", "torch_deps")
load("@rules_cc//examples:experimental_cc_shared_library.bzl", "cc_shared_library")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch", "if_cuda")
load("//rtp_llm/cpp/deep_gemm:def.bzl", "sub_lib", "sm90_cuda_copts", "gen_cu_and_lib", "gen_dispatch_code")
load("//rtp_llm/cpp/deep_gemm:template.bzl", "NORMAL_GEMM_CASES", "GROUPED_CONTIGUOUS_GEMM_CASES", "GROUPED_MASKED_GEMM_CASES",
                "QWEN_NORMAL_CASES", "QWEN_CONTIGUOUS_CASES", "QWEN_MASKED_CASES", "template_header", "template", "template_tail",
                "dispatch_template_header", "dispatch_template", "dispatch_template_tail")

test_linkopts = [
    "-lpython3.10",
    "-ltorch",
    "-lc10",
    "-ltorch_cpu",
    "-ltorch_python",
    "-lnccl",
    "-lnvToolsExt",
    "-lcupti",
] + select({
    "@//:using_cuda": [
        "-ltorch_cuda",
        "-lc10_cuda",
        "-L/usr/local/cuda/lib64",
        "-lcudart",
    ],
    "@//:using_rocm": [
        "-ltorch_hip",
        "-lc10_hip",
    ],
    "//conditions:default": [
    ],
})

gen_cpp_code("deepgemm_qwen", [QWEN_NORMAL_CASES + QWEN_CONTIGUOUS_CASES + QWEN_MASKED_CASES],
             template_header, template, template_tail, element_per_file=1, suffix=".cu")

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

sub_lib("deepgemm_qwen_inst", ":deepgemm_qwen")

gen_cu_and_lib("deepgemm_dpsk", NORMAL_GEMM_CASES + GROUPED_CONTIGUOUS_GEMM_CASES + GROUPED_MASKED_GEMM_CASES, 2, template_header, template, template_tail, 1)

gen_dispatch_code(
    "deepgemm_dispatch", 
    NORMAL_GEMM_CASES + GROUPED_CONTIGUOUS_GEMM_CASES + GROUPED_MASKED_GEMM_CASES + 
    QWEN_NORMAL_CASES + QWEN_CONTIGUOUS_CASES + QWEN_MASKED_CASES,
    dispatch_template_header,
    dispatch_template,
    dispatch_template_tail
)

cc_library(
    name = "deep_gemm_plugins_impl",
    hdrs = ["DeepGemmPlugin.h", "utils.h"],
    srcs = ["DeepGemmPlugin.cpp", ":deepgemm_qwen_inst", ":deepgemm_dispatch"],
    copts = copts(),
    deps = [
        ":deepgemm_dpsk_inst",
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
    tags = ["H20"],
    exec_properties = {'gpu':'H20'},
)
