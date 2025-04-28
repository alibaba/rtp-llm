load("//:def.bzl", "copts", "cuda_copts", "gen_cpp_code")
load("//bazel:arch_select.bzl", "torch_deps")
load("@rules_cc//examples:experimental_cc_shared_library.bzl", "cc_shared_library")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch", "if_cuda")
load("//src/fastertransformer/deep_gemm:def.bzl", "sub_lib", "sm90_cuda_copts")

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



NORMAL_GEMM_CASES = [
    ("2112", "7168", "1", "DeepGemmType::Normal"),
    ("4096", "7168", "1", "DeepGemmType::Normal"),
    ("7168", "2048", "1", "DeepGemmType::Normal"),
    ("2048", "7168", "1", "DeepGemmType::Normal"),
    ("16384", "512", "1", "DeepGemmType::Normal"),
    ("24576", "1536", "1", "DeepGemmType::Normal"),
    ("7168", "16384", "1", "DeepGemmType::Normal"),
    ("18432", "7168", "1", "DeepGemmType::Normal"),
    ("36864", "7168", "1", "DeepGemmType::Normal"),
    ("7168", "18432", "1", "DeepGemmType::Normal"),

    # tp 8
    ("3072", "1536", "1", "DeepGemmType::Normal"),
    ("2048", "512", "1", "DeepGemmType::Normal"),
    ("2304", "7168", "1", "DeepGemmType::Normal"),
    ("7168", "2304", "1", "DeepGemmType::Normal")
]

GROUPED_CONTIGUOUS_GEMM_CASES = [
    ("4096", "7168", "256", "DeepGemmType::GroupedContiguous"),
    ("7168", "4096", "256", "DeepGemmType::GroupedContiguous"),
    ("7168", "2048", "256", "DeepGemmType::GroupedContiguous"),

    ("4096", "7168", "128", "DeepGemmType::GroupedContiguous"),
    ("7168", "4096", "128", "DeepGemmType::GroupedContiguous"),
    ("7168", "2048", "128", "DeepGemmType::GroupedContiguous"),

    ("4096", "7168", "8", "DeepGemmType::GroupedContiguous"),
    ("7168", "4096", "8", "DeepGemmType::GroupedContiguous"),
    ("7168", "2048", "8", "DeepGemmType::GroupedContiguous"),

    ("4096", "7168", "2", "DeepGemmType::GroupedContiguous"),
    ("7168", "4096", "2", "DeepGemmType::GroupedContiguous"),
    ("7168", "2048", "2", "DeepGemmType::GroupedContiguous"),

    ("4096", "7168", "32", "DeepGemmType::GroupedContiguous"),
    ("7168", "4096", "32", "DeepGemmType::GroupedContiguous"),
    ("7168", "2048", "32", "DeepGemmType::GroupedContiguous"),

    ("4096", "7168", "64", "DeepGemmType::GroupedContiguous"),
    ("7168", "4096", "64", "DeepGemmType::GroupedContiguous"),
    ("7168", "2048", "64", "DeepGemmType::GroupedContiguous"),
]

GROUPED_MASKED_GEMM_CASES = [
    ("4096", "7168", "256", "DeepGemmType::GroupedMasked"),
    ("7168", "4096", "256", "DeepGemmType::GroupedMasked"),
    ("7168", "2048", "256", "DeepGemmType::GroupedMasked"),

    ("4096", "7168", "128", "DeepGemmType::GroupedMasked"),
    ("7168", "4096", "128", "DeepGemmType::GroupedMasked"),
    ("7168", "2048", "128", "DeepGemmType::GroupedMasked"),

    ("4096", "7168", "32", "DeepGemmType::GroupedMasked"),
    ("7168", "4096", "32", "DeepGemmType::GroupedMasked"),
    ("7168", "2048", "32", "DeepGemmType::GroupedMasked"),

    ("4096", "7168", "64", "DeepGemmType::GroupedMasked"),
    ("7168", "4096", "64", "DeepGemmType::GroupedMasked"),
    ("7168", "2048", "64", "DeepGemmType::GroupedMasked"),

    ("4096", "7168", "2", "DeepGemmType::GroupedMasked"),
    ("7168", "4096", "2", "DeepGemmType::GroupedMasked"),
    ("7168", "2048", "2", "DeepGemmType::GroupedMasked"),

    ("4096", "7168", "8", "DeepGemmType::GroupedMasked"),
    ("7168", "4096", "8", "DeepGemmType::GroupedMasked"),
    ("7168", "2048", "8", "DeepGemmType::GroupedMasked"),
]

QWEN_CASES = [
    ("9216", "4096", "1", "DeepGemmType::Normal"),
    ("4096", "8192", "1", "DeepGemmType::Normal"),
    ("4608", "4096", "1", "DeepGemmType::Normal"),
    ("4096", "4096", "1", "DeepGemmType::Normal"),
    ("2304", "4096", "1", "DeepGemmType::Normal"),
    ("4096", "2048", "1", "DeepGemmType::Normal"),

    ("3072", "4096", "128", "DeepGemmType::GroupedContiguous"),
    ("4096", "1536", "128", "DeepGemmType::GroupedContiguous"),
    ("3072", "4096", "128", "DeepGemmType::GroupedMasked"),
    ("4096", "1536", "128", "DeepGemmType::GroupedMasked"),

    ("3072", "4096", "64", "DeepGemmType::GroupedContiguous"),
    ("4096", "1536", "64", "DeepGemmType::GroupedContiguous"),
    ("3072", "4096", "64", "DeepGemmType::GroupedMasked"),
    ("4096", "1536", "64", "DeepGemmType::GroupedMasked"),

    ("3072", "4096", "32", "DeepGemmType::GroupedContiguous"),
    ("4096", "1536", "32", "DeepGemmType::GroupedContiguous"),
    ("3072", "4096", "32", "DeepGemmType::GroupedMasked"),
    ("4096", "1536", "32", "DeepGemmType::GroupedMasked"),

    ("3072", "4096", "5", "DeepGemmType::GroupedContiguous"),
    ("4096", "1536", "5", "DeepGemmType::GroupedContiguous"),
    ("3072", "4096", "5", "DeepGemmType::GroupedMasked"),
    ("4096", "1536", "5", "DeepGemmType::GroupedMasked"),

    ("3072", "4096", "4", "DeepGemmType::GroupedContiguous"),
    ("4096", "1536", "4", "DeepGemmType::GroupedContiguous"),
    ("3072", "4096", "4", "DeepGemmType::GroupedMasked"),
    ("4096", "1536", "4", "DeepGemmType::GroupedMasked"),

    ("3072", "4096", "2", "DeepGemmType::GroupedContiguous"),
    ("4096", "1536", "2", "DeepGemmType::GroupedContiguous"),
    ("3072", "4096", "2", "DeepGemmType::GroupedMasked"),
    ("4096", "1536", "2", "DeepGemmType::GroupedMasked"),
]

template_header = """
#include "src/fastertransformer/deep_gemm/utils.h"
#ifdef ENABLE_FP8
#include "src/fastertransformer/deep_gemm/deep_gemm_template.h"
#include "src/fastertransformer/deep_gemm/include/fp8_gemm.cuh"
#endif
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
namespace fastertransformer {
"""
template = """
#ifdef ENABLE_FP8
template void dispatchBlockNK<{0}, {1}, {2}, {3}>(__nv_bfloat16*         output,
                     __nv_fp8_e4m3*         lhs,
                     float*                 lhs_scale,
                     __nv_fp8_e4m3*         rhs,
                     float*                 rhs_scale,
                     int*                   grouped_layout,
                     uint32_t               m,
                     uint32_t               bm,
                     uint32_t               bn,
                     uint32_t               bk,
                     uint32_t               num_stages,
                     uint32_t               num_tma_multicast,
                     cudaStream_t           stream,
                     uint32_t               num_sms,
                     uint32_t               smem_size);
#endif
"""
template_tail = """
}
"""

gen_cpp_code("deepgemm_dpsk", [NORMAL_GEMM_CASES + GROUPED_CONTIGUOUS_GEMM_CASES + GROUPED_MASKED_GEMM_CASES],
             template_header, template, template_tail, element_per_file=1, suffix=".cu")


gen_cpp_code("deepgemm_qwen", [QWEN_CASES],
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

sub_lib("deepgemm_dpsk_inst", ":deepgemm_dpsk")
sub_lib("deepgemm_qwen_inst", ":deepgemm_qwen")

cc_library(
    name = "deep_gemm_plugins_impl",
    hdrs = ["DeepGemmPlugin.h", "utils.h"],
    srcs = ["DeepGemmPlugin.cpp", ":deepgemm_dpsk_inst", ":deepgemm_qwen_inst"],
    copts = copts(),
    deps = [
        "//src/fastertransformer/core:buffer_torch",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "//src/fastertransformer/cuda:nvtx",
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
        "//src/fastertransformer/core:buffer_torch",
        "//src/fastertransformer/devices/testing:device_test_utils",
        "//src/fastertransformer/devices/cuda_impl:gpu_base",
        "//src/fastertransformer/devices/cuda_impl:cuda_impl",
    ],
    tags = ["H20"],
    exec_properties = {'gpu':'H20'},
)
