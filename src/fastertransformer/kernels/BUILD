load("//:def.bzl", "copts", "cuda_copts", "rocm_copts", "any_cuda_copts", "gen_cpp_code")
load("@rules_cc//examples:experimental_cc_shared_library.bzl", "cc_shared_library")
package(default_visibility = ["//maga_transformer:__subpackages__"])

any_cuda_deps = select({
    "@//:using_cuda": [
        "//src/fastertransformer/cuda:memory_utils",
        "//src/fastertransformer/cuda:nvtx",
        "//src/fastertransformer/cuda:cuda_utils",
        "//3rdparty:cub",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    "@//:using_rocm": [
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:hip",
        "//src/fastertransformer/cuda:cuda_utils_rocm",
        "//src/fastertransformer/rocm:rocm_types_hdr",
        "//src/fastertransformer/rocm:rocm_utils",
    ],
    "//conditions:default": [
    ],
})

cc_library(
    name = "kernels_cu",
    srcs = glob([
        "*.cu",
        "mla_kernels/*.cu",
        "eplb/*.cu",
    ], exclude = [
        "layernorm_fp8_kernels.cu",
        "activation_fp8_kernels.cu",
        "unfused_attention_fp8_kernels.cu",
        "moe_kernels.cu",
        "rocm/*.cu",
    ]) + [
        ":mmha1",
        ":mmha2",
        ":dmmha"
    ],
    hdrs = glob([
        "*.h",
        "*.cuh",
        "mla_kernels/*.h",
        "eplb/*.h",
        "kv_cache/*.h"
    ], exclude=[
        "AttentionFP8Weight.h",
        "moe_kernels.h",
        "rocm/*.h",
    ]),
    deps = [
        "//src/fastertransformer/cuda:memory_utils",
        "//src/fastertransformer/cuda:nvtx",
        "//src/fastertransformer/cuda:cuda_utils",
        "//src/fastertransformer/utils",
        "//3rdparty:cub",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + select({
       "//:enable_triton": ["//src/fastertransformer/kernels/triton:triton_kernel"],
       "//conditions:default": [],
    }),
    copts = cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocm_utils",
    hdrs = [
        "_vector_abs_max.h",
        "_convert_from_float.h",
        "_convert_to_float.h",
        "_cast_to_int8.h",
        "_add.h",
        "_mul.h",
        "_fma.h",
        "_sum_dot_zero.h",
        "_logn_attention.h",
        "_convert_from_fp8.h",
        "_convert_to_fp8.h",
    ]
)

cc_library(
    name = "kernels",
    srcs = glob([
        "*.cc",
    ], exclude=[
        "**/*_test.cc",
    ]),
    deps = [
        ":rocm_utils",
        ":mmha_hdrs",
        ":kernels_cu",
        "//3rdparty:cuda_driver",
        "//src/fastertransformer/utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gpt_kernels_rocm",
    srcs = [
        "hello_world.cu",
        "moe_topKSoftmax_kernels.cu",
        "gpt_kernels.cu",
        "add_residual_kernels.cu",
        "alpha_layernorm_kernels.cu",
        "rmsnormKernels.cu",
        "activation_kernels.cu",
        "unfused_attention_kernels.cu",
        "sampling_topk_kernels.cu",
        "sampling_topp_kernels.cu",
        "sampling_penalty_kernels.cu",
        "custom_ar_kernels.cu",
        "rocm/quantization_rocm.cu",
        "rocm/layernorm_kernels.cu",
	    ":mmha1",
        ":mmha2",
        ":dmmha",
    ],
    hdrs = [
        "hello_world.h",
        "moe_topKSoftmax_kernels.h",
        "gpt_kernels.h",
        "add_residual_kernels.h",
        "alpha_layernorm_kernels.h",
        "rmsnormKernels.h",
        "activation_kernels.h",
        "unfused_attention_kernels.h",
        "sampling_topk_kernels.h",
        "sampling_topp_kernels.h",
        "sampling_penalty_kernels.h",
        "penalty_types.h",
        "custom_ar_kernels.h",
        "rocm/quantization_rocm.h",
        "rocm/layernorm_kernels.h",
    ],
    deps = [
        ":rocm_utils",
        ":attn_utils",
        ":mmha_hdrs",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:hip",
        "//src/fastertransformer/cuda:cuda_utils_rocm",
        "//src/fastertransformer/rocm:memory_utils",
        "//src/fastertransformer/rocm:rocm_utils",
        "//src/fastertransformer/rocm:rocm",
        "//src/fastertransformer/utils",
        "//src/fastertransformer/rocm:rocm_types_hdr",
    ],
    copts = rocm_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "hello_kernel",
    srcs = [
        "hello_world.cu",
    ],
    hdrs = [
        "hello_world.h",
    ],
    deps = [
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:hip",
        "//src/fastertransformer/utils",
        "//src/fastertransformer/cuda:cuda_utils_rocm",
        "//src/fastertransformer/rocm:rocm_types_hdr",
        "//src/fastertransformer/rocm:rocm_utils",
    ],
    copts = rocm_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)


T_Tcache = [('uint16_t', 'uint16_t', '0'), ('__nv_bfloat16', '__nv_bfloat16', '0'),
            ('uint16_t', 'int8_t', '0'), ('__nv_bfloat16', 'int8_t', '0'),
            ('uint16_t', '__nv_fp8_e4m3', '1'), ('__nv_bfloat16', '__nv_fp8_e4m3', '1'),
            ('float', 'float', '0')]

Dh = ['64', '96', '128', '192', '256']
DO_MULTI_BLOCK = ['true', 'false']
ROPE_TYPE = ['RopeStyle::No', 'RopeStyle::Base', 'RopeStyle::Glm2', 'RopeStyle::DynamicNTK', 'RopeStyle::QwenDynamicNTK', 'RopeStyle::Yarn', 'RopeStyle::Llama3', 'RopeStyle::Mrope']

template_header = """
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_launch.h"
namespace fastertransformer {
"""
template = """
#if defined(ENABLE_FP8) || !{2}
template void mmha_launch_kernel_ex<{0}, {1}, KVBlockArray, Multihead_attention_params<{0}, false>, {3}, false, {4}, {5}>(
Multihead_attention_params<{0}, false>&, const KVBlockArray&, const cudaStream_t&, int);
#endif
"""
template_tail = """
}
"""

gen_cpp_code("mmha_inst_1", [T_Tcache[:4], Dh, DO_MULTI_BLOCK, ROPE_TYPE],
             template_header, template, template_tail, element_per_file=4, suffix=".cu")
gen_cpp_code("mmha_inst_2", [T_Tcache[4:], Dh, DO_MULTI_BLOCK, ROPE_TYPE],
             template_header, template, template_tail, element_per_file=4, suffix=".cu")

cc_library(
    name = "attn_utils",
    hdrs = [
        "decoder_masked_multihead_attention_utils.h",
        "rotary_position_embedding.h",
        "kv_cache/kv_cache_utils.h",
        "kv_cache/kv_cache_index.h"
    ],
    deps = any_cuda_deps,
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mmha_hdrs",
    hdrs = [
        "decoder_masked_multihead_attention/decoder_masked_multihead_attention.h",
    ],
    deps = any_cuda_deps,
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mmha_cu_1",
    srcs = [
        ":mmha_inst_1"
    ],
    hdrs = glob([
        "decoder_masked_multihead_attention/*.h",
    ], exclude=[
        "decoder_masked_multihead_attention/decoder_masked_multihead_attention.h",
    ]),
    deps = any_cuda_deps + [
        ":rocm_utils",
        ":mmha_hdrs",
        ":attn_utils",
        "//src/fastertransformer/utils",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mmha_cu_2",
    srcs = [
        ":mmha_inst_2"
    ],
    hdrs = glob([
        "decoder_masked_multihead_attention/*.h",
    ], exclude=[
        "decoder_masked_multihead_attention/decoder_masked_multihead_attention.h",
    ]),
    deps = any_cuda_deps + [
        ":rocm_utils",
        ":mmha_hdrs",
        ":attn_utils",
        "//src/fastertransformer/utils",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "dmmha_cu",
    srcs = glob([
        "decoder_masked_multihead_attention/*.cu",
    ]),
    hdrs = glob([
        "decoder_masked_multihead_attention/*.h",
    ], exclude=[
        "decoder_masked_multihead_attention/decoder_masked_multihead_attention.h",
    ]),
    deps = any_cuda_deps + [
        ":rocm_utils",
        ":mmha_hdrs",
        ":attn_utils",
        "//src/fastertransformer/utils",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_shared_library(
    name = "mmha1",
    roots = [":mmha_cu_1"],
    preloaded_deps = any_cuda_deps + [
        ":rocm_utils",
        ":mmha_hdrs",
        ":attn_utils",
        "//src/fastertransformer/utils",
    ],
)

cc_shared_library(
    name = "mmha2",
    roots = [":mmha_cu_2"],
    preloaded_deps = any_cuda_deps + [
        ":rocm_utils",
        ":mmha_hdrs",
        ":attn_utils",
        "//src/fastertransformer/utils",
    ],
)

cc_shared_library(
    name = "dmmha",
    roots = [":dmmha_cu"],
    preloaded_deps = any_cuda_deps + [
        ":rocm_utils",
        ":mmha_hdrs",
        ":attn_utils",
        "//src/fastertransformer/utils",
    ],
)
