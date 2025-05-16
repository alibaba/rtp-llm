load("//:def.bzl", "copts", "cuda_copts", "gen_cpp_code")
load("//bazel:arch_select.bzl", "torch_deps")
load("@rules_cc//examples:experimental_cc_shared_library.bzl", "cc_shared_library")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch", "if_cuda")
load("//maga_transformer/cpp/deep_gemm:def.bzl", "sub_lib", "sm90_cuda_copts", "gen_cu_and_lib", "gen_dispatch_code")

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

    # # tp 8
    # ("3072", "1536", "1", "DeepGemmType::Normal"),
    # ("2048", "512", "1", "DeepGemmType::Normal"),
    # ("2304", "7168", "1", "DeepGemmType::Normal"),
    # ("7168", "2304", "1", "DeepGemmType::Normal")
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

    ("4096", "7168", "9", "DeepGemmType::GroupedContiguous"),
    ("7168", "4096", "9", "DeepGemmType::GroupedContiguous"),
    ("7168", "2048", "9", "DeepGemmType::GroupedContiguous"),

    ("4096", "7168", "10", "DeepGemmType::GroupedContiguous"),
    ("7168", "4096", "10", "DeepGemmType::GroupedContiguous"),
    ("7168", "2048", "10", "DeepGemmType::GroupedContiguous"),

    ("4096", "7168", "2", "DeepGemmType::GroupedContiguous"),
    ("7168", "4096", "2", "DeepGemmType::GroupedContiguous"),
    ("7168", "2048", "2", "DeepGemmType::GroupedContiguous"),

    ("4096", "7168", "16", "DeepGemmType::GroupedContiguous"),
    ("7168", "4096", "16", "DeepGemmType::GroupedContiguous"),
    ("7168", "2048", "16", "DeepGemmType::GroupedContiguous"),

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

    ("4096", "7168", "16", "DeepGemmType::GroupedMasked"),
    ("7168", "4096", "16", "DeepGemmType::GroupedMasked"),
    ("7168", "2048", "16", "DeepGemmType::GroupedMasked"),

    ("4096", "7168", "8", "DeepGemmType::GroupedMasked"),
    ("7168", "4096", "8", "DeepGemmType::GroupedMasked"),
    ("7168", "2048", "8", "DeepGemmType::GroupedMasked"),

    ("4096", "7168", "9", "DeepGemmType::GroupedMasked"),
    ("7168", "4096", "9", "DeepGemmType::GroupedMasked"),
    ("7168", "2048", "9", "DeepGemmType::GroupedMasked"),

    ("4096", "7168", "10", "DeepGemmType::GroupedMasked"),
    ("7168", "4096", "10", "DeepGemmType::GroupedMasked"),
    ("7168", "2048", "10", "DeepGemmType::GroupedMasked"),
]

QWEN_NORMAL_CASES = [
    ("9216", "4096", "1", "DeepGemmType::Normal"),
    ("4096", "8192", "1", "DeepGemmType::Normal"),
    ("4608", "4096", "1", "DeepGemmType::Normal"),
    ("4096", "4096", "1", "DeepGemmType::Normal"),
    ("2304", "4096", "1", "DeepGemmType::Normal"),
    ("4096", "2048", "1", "DeepGemmType::Normal"),

    # qwen3-30b-a3b tp=1/2 
    ("5120", "2048", "1", "DeepGemmType::Normal"),
    ("2048", "4096", "1", "DeepGemmType::Normal"),
    ("2560", "2048", "1", "DeepGemmType::Normal"),
    ("2048", "2048", "1", "DeepGemmType::Normal"),    
]

QWEN_CONTIGUOUS_CASES = [
    ("3072", "4096", "128", "DeepGemmType::GroupedContiguous"),
    ("4096", "1536", "128", "DeepGemmType::GroupedContiguous"),

    ("3072", "4096", "64", "DeepGemmType::GroupedContiguous"),
    ("4096", "1536", "64", "DeepGemmType::GroupedContiguous"),

    ("3072", "4096", "32", "DeepGemmType::GroupedContiguous"),
    ("4096", "1536", "32", "DeepGemmType::GroupedContiguous"),
    
    ("3072", "4096", "5", "DeepGemmType::GroupedContiguous"),
    ("4096", "1536", "5", "DeepGemmType::GroupedContiguous"),

    ("3072", "4096", "16", "DeepGemmType::GroupedContiguous"),
    ("4096", "1536", "16", "DeepGemmType::GroupedContiguous"),

    ("3072", "4096", "4", "DeepGemmType::GroupedContiguous"),
    ("4096", "1536", "4", "DeepGemmType::GroupedContiguous"),

    ("3072", "4096", "2", "DeepGemmType::GroupedContiguous"),
    ("4096", "1536", "2", "DeepGemmType::GroupedContiguous"),

    ("3072", "4096", "16", "DeepGemmType::GroupedContiguous"),
    ("4096", "1536", "16", "DeepGemmType::GroupedContiguous"),

    ("1536", "2048", "128", "DeepGemmType::GroupedContiguous"),
    ("2048", "768", "128", "DeepGemmType::GroupedContiguous"),

    ("1536", "2048", "64", "DeepGemmType::GroupedContiguous"),
    ("2048", "768", "64", "DeepGemmType::GroupedContiguous"),
]

QWEN_MASKED_CASES = [
    ("3072", "4096", "128", "DeepGemmType::GroupedMasked"),
    ("4096", "1536", "128", "DeepGemmType::GroupedMasked"),

    ("3072", "4096", "64", "DeepGemmType::GroupedMasked"),
    ("4096", "1536", "64", "DeepGemmType::GroupedMasked"),

    ("3072", "4096", "32", "DeepGemmType::GroupedMasked"),
    ("4096", "1536", "32", "DeepGemmType::GroupedMasked"),

    ("3072", "4096", "5", "DeepGemmType::GroupedMasked"),
    ("4096", "1536", "5", "DeepGemmType::GroupedMasked"),

    ("3072", "4096", "16", "DeepGemmType::GroupedMasked"),
    ("4096", "1536", "16", "DeepGemmType::GroupedMasked"),

    ("3072", "4096", "4", "DeepGemmType::GroupedMasked"),
    ("4096", "1536", "4", "DeepGemmType::GroupedMasked"),

    ("3072", "4096", "2", "DeepGemmType::GroupedMasked"),
    ("4096", "1536", "2", "DeepGemmType::GroupedMasked"),

    ("3072", "4096", "16", "DeepGemmType::GroupedMasked"),
    ("4096", "1536", "16", "DeepGemmType::GroupedMasked"),   

    ("1536", "2048", "128", "DeepGemmType::GroupedMasked"),
    ("2048", "768", "128", "DeepGemmType::GroupedMasked"),

    ("1536", "2048", "64", "DeepGemmType::GroupedMasked"),
    ("2048", "768", "64", "DeepGemmType::GroupedMasked"),
]

template_header = """
#include "maga_transformer/cpp/deep_gemm/utils.h"
#ifdef ENABLE_FP8
#include "maga_transformer/cpp/deep_gemm/deep_gemm_template.h"
#include "maga_transformer/cpp/deep_gemm/include/fp8_gemm.cuh"
#endif
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
namespace rtp_llm {
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

dispatch_template_header = """
#include <cuda.h>
#include "maga_transformer/cpp/deep_gemm/utils.h"
#include "maga_transformer/cpp/utils/AssertUtils.h"
#include "maga_transformer/cpp/cuda/cuda_utils.h"
#include "maga_transformer/cpp/deep_gemm/DeepGemmPlugin.h"
namespace rtp_llm {
#ifdef ENABLE_FP8
template<uint32_t N, uint32_t K, uint32_t GROUP_NUM, DeepGemmType GEMM_TYPE>
void dispatchBlockNK(__nv_bfloat16*         output,
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

#define DISPATCH_DEEP_GEMM(N, K, GROUP_NUM, GEMM_TYPE)                                           \\
    if (n == N && k == K && num_groups == GROUP_NUM && gemm_type == GEMM_TYPE) {                 \\
        dispatchBlockNK<N, K, GROUP_NUM, GEMM_TYPE>(output,                                      \\
                                                    lhs,                                         \\
                                                    lhs_scale,                                   \\
                                                    rhs,                                         \\
                                                    rhs_scale,                                   \\
                                                    grouped_layout,                              \\
                                                    m,                                           \\
                                                    bm,                                          \\
                                                    bn,                                          \\
                                                    bk,                                          \\
                                                    num_stages,                                  \\
                                                    num_tma_multicast,                           \\
                                                    stream,                                      \\
                                                    num_sms,                                     \\
                                                    smem_size);                                  \\
        return;                                                                                  \\
    }

void runDeepGemm(__nv_bfloat16*         output,
                 __nv_fp8_e4m3*         lhs,
                 float*                 lhs_scale,
                 __nv_fp8_e4m3*         rhs,
                 float*                 rhs_scale,
                 int*                   grouped_layout,
                 uint32_t               m,
                 uint32_t               n,
                 uint32_t               k,
                 uint32_t               bm,
                 uint32_t               bn,
                 uint32_t               bk,
                 uint32_t               num_groups,
                 uint32_t               num_stages,
                 uint32_t               num_tma_multicast,
                 DeepGemmType           gemm_type,
                 cudaStream_t           stream,
                 uint32_t               num_sms,
                 uint32_t               smem_size) 
{
    RTP_LLM_LOG_DEBUG("m:%u, n:%u, k:%u , bm:%u, bn:%u, bk:%u, num_groups:%u, num_stages:%u, num_tma_multicast:%u\\n", m, n, k, bm, bn, bk, num_groups, num_stages, num_tma_multicast);
"""

dispatch_template = """
    DISPATCH_DEEP_GEMM({0}, {1}, {2}, {3})
"""

dispatch_template_tail = """
    RTP_LLM_FAIL("DISPATCH_DEEP_GEMM(N=%u, K=%u, NUM_GROUPS=%u, GEMM_TYPE=%u) no template found", n, k, num_groups, gemm_type);
}
#endif
} // namespace rtp_llm
"""

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
        "//maga_transformer/cpp/core:buffer_torch",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "//maga_transformer/cpp/cuda:nvtx",
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
        "//maga_transformer/cpp/core:buffer_torch",
        "//maga_transformer/cpp/devices/testing:device_test_utils",
        "//maga_transformer/cpp/devices/cuda_impl:gpu_base",
        "//maga_transformer/cpp/devices/cuda_impl:cuda_impl",
    ],
    tags = ["H20"],
    exec_properties = {'gpu':'H20'},
)
