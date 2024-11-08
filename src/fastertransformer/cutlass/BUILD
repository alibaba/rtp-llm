load("//:def.bzl", "copts", "cuda_copts")
load("@rules_cc//examples:experimental_cc_shared_library.bzl", "cc_shared_library")
load("//:def.bzl", "copts", "cuda_copts", "gen_cpp_code")

cc_library(
    name = "cutlass_kernels_common",
    srcs = glob([
        "cutlass_kernels/*.cc",
    ]),
    hdrs = glob([
        "cutlass_kernels/*.h",
    ]),
    deps = [
        "//src/fastertransformer/cutlass/cutlass_extensions:cutlass_extensions",
        "//maga_transformer/cpp:utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "//src/fastertransformer/kernels:kernels",
    ],
    # include_prefix = "src",
    copts = copts(),
    visibility = ["//visibility:public"],
    alwayslink = True,
)

T = [('KernelType::BF16Int4Groupwise', 'BF16DetailsA', 'Int4DetailsW'),
     ('KernelType::BF16Int8PerChannel', 'BF16DetailsA', 'Int8DetailsW'),
     ('KernelType::FP16Int4Groupwise', 'FP16DetailsA', 'Int4DetailsW'),
     ('KernelType::FP16Int8PerChannel', 'FP16DetailsA', 'Int8DetailsW')]

Layout = ['ColumnMajor', 'ColumnMajorInterleaved']
Tile = ['64']

template_header = """
#include "src/fastertransformer/cutlass/cutlass_kernels/weightOnlyBatchedGemv/kernelDispatcher.h"
namespace tensorrt_llm {
namespace kernels {
namespace weight_only {
"""
template = """
INSTANTIATE_WEIGHT_ONLY_CUDA_DISPATCHERS({0}, {1}, {2}, {3}, {4});
"""
template_tail = """
}}}
"""

gen_cpp_code("gemv_inst", [T, Layout, Tile],
             template_header, template, template_tail, element_per_file=4, suffix=".cu")

cc_library(
    name = "weight_only_gemm_cu",
    srcs = glob([
        "decoder_masked_multihead_attention/*.cu",
    ]) + [
        ":gemv_inst"
    ],
    hdrs = glob([
            "cutlass_kernels/weightOnlyBatchedGemv/*.h",
            "interface.h"
        ]),
    deps = [
    	":cutlass_kernels_common",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
    alwayslink = True,
)

cc_library(
    name = "fpA_intB_cu",
    srcs = glob([
        "cutlass_kernels/fpA_intB_gemm/*.cu",
        "cutlass_kernels/fpA_intB_gemm/*.h",
        "cutlass_kernels/fpA_intB_gemm/*.cc",
    ]),
    hdrs = glob([
        "cutlass_kernels/fpA_intB_gemm/*.h",
    ]),
    deps = [
    	":cutlass_kernels_common",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    alwayslink = True,
)

cc_shared_library(
    name = "fpA_intB",
    roots = [":fpA_intB_cu"],
    preloaded_deps = [
    	":cutlass_kernels_common",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "int8_gemm_cu",
    srcs = glob([
        "cutlass_kernels/int8_gemm/*.cu",
        "cutlass_kernels/int8_gemm/*.h",
    ]),
    hdrs = glob([
        "cutlass_kernels/int8_gemm/*.h",
    ]),
    deps = [
    	":cutlass_kernels_common",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    alwayslink = True,
)

cc_shared_library(
    name = "int8_gemm",
    roots = [":int8_gemm_cu"],
    preloaded_deps = [
    	":cutlass_kernels_common",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "moe_cu",
    srcs = glob([
        "cutlass_kernels/moe_gemm/*.cu",
        "cutlass_kernels/moe_gemm/*.h",
    ]),
    hdrs = glob([
        "cutlass_kernels/moe_gemm/moe_gemm_kernels_template.h",
        "cutlass_kernels/moe_gemm/moe_gemm_kernels.h",
        "cutlass_kernels/moe_gemm/moe_kernels.h",
    ]),
    include_prefix = "src",
    deps = [
    	":cutlass_kernels_common",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    alwayslink = True,
)

cc_shared_library(
    name = "moe",
    roots = [":moe_cu"],
    preloaded_deps = [
    	":cutlass_kernels_common",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "group_cu",
    srcs = glob([
        "cutlass_kernels/group_gemm/*.cu",
        "cutlass_kernels/group_gemm/*.h",
    ]),
    hdrs = glob([
        "cutlass_kernels/group_gemm/group_gemm_template.h",
        "cutlass_kernels/group_gemm/group_gemm.h",
    ]),
    include_prefix = "src",
    deps = [
    	":cutlass_kernels_common",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    alwayslink = True,
)

cc_library(
    name = "cutlass_headers",
    deps = [
        "@cutlass//:cutlass",
        "@cutlass//:cutlass_utils",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cutlass_kernels_impl",
    srcs = [
        ":fpA_intB",
        ":moe",
        ":int8_gemm"
    ],
    deps = [
        ":weight_only_gemm_cu",
        ":group_cu",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cutlass_interface",
    hdrs = [
        "interface.h",
        "cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h",
        "cutlass_kernels/int8_gemm/int8_gemm.h",
        "cutlass_kernels/group_gemm/group_gemm.h",
        "cutlass_kernels/moe_gemm/moe_gemm_kernels.h",
        "cutlass_kernels/moe_gemm/moe_kernels.h",
        "cutlass_kernels/cutlass_preprocessors.h",
        "cutlass_kernels/weight_only_quant_op.h",
        "cutlass_kernels/gemm_configs.h",
        "cutlass_kernels/weightOnlyBatchedGemv/details.h",
        "cutlass_kernels/weightOnlyBatchedGemv/kernelLauncher.h",
        "cutlass_kernels/weightOnlyBatchedGemv/common.h",
        "cutlass_kernels/cutlass_heuristic.h",
        "cutlass_kernels/gemm_lut_utils.h",
        "cutlass_kernels/gemm_lut.h",
    ],
    deps = [
        "@cutlass//:cutlass",
    ],
    visibility = ["//visibility:public"],
)
