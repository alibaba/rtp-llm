load("//:def.bzl", "copts", "cuda_copts", "gen_cpp_code")
load("//bazel:arch_select.bzl", "torch_deps")
load("@rules_cc//examples:experimental_cc_shared_library.bzl", "cc_shared_library")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch", "if_cuda")

sm90_cuda_copts = copts() + cuda_default_copts_without_arch() + if_cuda(["-nvcc_options=objdir-as-tempdir"]) + [
    '--cuda-include-ptx=sm_90a', '--cuda-gpu-arch=sm_90a',
    '--compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi'
]

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



N_K_GROUP_TYPE= [
("2112", "7168", "1", "DeepGemmType::Normal"),
("4096", "7168", "1", "DeepGemmType::Normal"),
("7168", "2048", "1", "DeepGemmType::Normal"),
("2048", "7168", "1", "DeepGemmType::Normal"),
("16384", "512", "1", "DeepGemmType::Normal"),
("24576", "1536", "1", "DeepGemmType::Normal"),
("7168", "16384", "1", "DeepGemmType::Normal"),
("18432", "7168", "1", "DeepGemmType::Normal"),
("7168", "18432", "1", "DeepGemmType::Normal"),

# tp 8
("3072", "1536", "1", "DeepGemmType::Normal"),
("2048", "512", "1", "DeepGemmType::Normal"),
("2304", "7168", "1", "DeepGemmType::Normal"),
("7168", "2304", "1", "DeepGemmType::Normal"),

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

("4096", "7168", "256", "DeepGemmType::GroupedMasked"),
("7168", "4096", "256", "DeepGemmType::GroupedMasked"),
("7168", "2048", "256", "DeepGemmType::GroupedMasked"),

("4096", "7168", "128", "DeepGemmType::GroupedMasked"),
("7168", "4096", "128", "DeepGemmType::GroupedMasked"),
("7168", "2048", "128", "DeepGemmType::GroupedMasked"),

("4096", "7168", "32", "DeepGemmType::GroupedMasked"),
("7168", "4096", "32", "DeepGemmType::GroupedMasked"),
("7168", "2048", "32", "DeepGemmType::GroupedMasked"),

("4096", "7168", "2", "DeepGemmType::GroupedMasked"),
("7168", "4096", "2", "DeepGemmType::GroupedMasked"),
("7168", "2048", "2", "DeepGemmType::GroupedMasked"),
]
template_header = """
#include "src/fastertransformer/deep_gemm/deep_gemm_template.h"
#include "src/fastertransformer/deep_gemm/utils.h"
#include "src/fastertransformer/deep_gemm/include/fp8_gemm.cuh"
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
namespace fastertransformer {
"""
template = """
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
                     bool                   is_tma_multicast_on_a,
                     cudaStream_t           stream,
                     uint32_t               num_sms,
                     uint32_t               smem_size);
"""
template_tail = """
}
"""

gen_cpp_code("deepgemm_inst", [N_K_GROUP_TYPE],
             template_header, template, template_tail, element_per_file=1, suffix=".cu")


cc_library(
    name = "deep_gemm_cu",
    hdrs = [
        "deep_gemm_template.h",
        "utils.h",
    ] + select({
        "@//:using_cuda12": glob([
            "include/*.cuh"
        ]),
        "//conditions:default": [],
    }),
    srcs = select({
        "@//:using_cuda12": [
            "deepgemm_inst",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//src/fastertransformer/core:buffer_torch",
        "//src/fastertransformer/cuda:nvtx",
        "//src/fastertransformer/utils",
        "//src/fastertransformer/cuda:cuda_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "@cutlass//:cutlass"
    ] + torch_deps(),
    copts = sm90_cuda_copts,
    visibility = ["//visibility:public"],
    alwayslink = True,
)

cc_library(
    name = "deep_gemm_plugins_impl",
    hdrs = ["DeepGemmPlugin.h"],
    srcs = ["DeepGemmPlugin.cpp"],
    copts = copts(),
    deps = [
        ":deep_gemm_cu",
        "//src/fastertransformer/core:buffer_torch",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "//src/fastertransformer/cuda:nvtx",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "deep_gemm_plugins_hdr",
    hdrs = ["DeepGemmPlugin.h", "utils.h"],
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
        ":deep_gemm_plugins_hdr",
        "@com_google_googletest//:gtest",
        "//src/fastertransformer/core:buffer_torch",
        "//src/fastertransformer/devices/testing:device_test_utils",
        "//src/fastertransformer/devices/cuda_impl:gpu_base",
        "//src/fastertransformer/devices/cuda_impl:cuda_impl",
    ],
    tags = ["H20"],
    exec_properties = {'gpu':'H20'},
)
