
load("//:def.bzl", "copts", "cuda_copts", "rocm_copts", "any_cuda_copts", "gen_cpp_code")
load("//bazel:arch_select.bzl", "torch_deps")
load("@rules_cc//examples:experimental_cc_shared_library.bzl", "cc_shared_library")
package(default_visibility = ["//rtp_llm:__subpackages__"])

cuda_deps = [
    "//rtp_llm/cpp/cuda:cuda_utils",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
]

rocm_deps = [
    ":rocm_utils",
    "@local_config_rocm//rocm:rocm_headers",
    "@local_config_rocm//rocm:hip",
    "//rtp_llm/cpp/cuda:cuda_utils_rocm",
    "//rtp_llm/cpp/rocm:rocm_types_hdr",
    "//rtp_llm/cpp/rocm:rocm_utils",
]

any_cuda_deps = select({
    "@//:using_cuda": cuda_deps,
    "@//:using_rocm": rocm_deps,
    "//conditions:default": [
    ],
})

cc_library(
    name = "cuda_cu",
    srcs = [
        "no_aux_tc_kernels.cu",  # CUDA-only kernel with cooperative_groups
        "per_token_group_quant_8bit.cu",  # CUDA-only kernel with ATen/cuda headers
        "scaled_fp8_quant.cu", # CUDA-only kernel with ATen/cuda headers
        "moe_topk_softmax_kernels.cu",  # CUDA-only kernel with ATen/cuda headers
        "comm_buffer.cu",  # CUDA-only kernel with vec_dtypes.cuh
    ],
    hdrs = [
        "no_aux_tc_kernels.h",
        "per_token_group_quant_8bit.h",
        "scaled_fp8_quant.h",
        "scaled_fp8_quant_utils.h",
        "moe_topk_softmax_kernels.h",
        "comm_buffer.h",
        "vec_dtypes.cuh",  # CUDA-only header with FP8/BF16 types
        "util.h",
    ],
    deps = [
        "//rtp_llm/cpp/cuda:cuda_utils",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = cuda_copts(),
)


cc_library(
    name = "rocm_cu",
    srcs = glob([
        "rocm/*.cu",
        "mla_kernels_rocm/*.cu",
    ]),
    hdrs = glob([
         "rocm/*.h",
         "mla_kernels_rocm/*.h",
    ]),
    deps = rocm_deps,
    copts = rocm_copts(),
)

cc_library(
    name = "rocm_utils",
    hdrs = glob([
        "rocm_utils/*.h",
    ]),
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_cu",
    srcs = glob([
        "*.cu",
        "mla_kernels/*.cu",
        "eplb/*.cu",
        "moe/*.cu",
    ], exclude = [
        "no_aux_tc_kernels.cu",  # CUDA-only kernel with cooperative_groups
        "per_token_group_quant_8bit.cu",  # CUDA-only kernel with ATen/cuda headers
        "scaled_fp8_quant.cu",  # CUDA-only kernel with ATen/cuda headers
        "moe_topk_softmax_kernels.cu",  # CUDA-only kernel with ATen/cuda headers
        "comm_buffer.cu",  # CUDA-only kernel with vec_dtypes.cuh
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
        "kv_cache/*.h",
        "moe/*.h",
    ]),
    deps = [
        ":attn_utils",
        ":mmha_hdrs",
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:system_utils",
    ] + select({
       "//:enable_triton": ["//rtp_llm/cpp/kernels/triton:triton_kernel"],
       "//conditions:default": [],
    }) + select({
        "@//:using_cuda": [
            ":cuda_cu",
        ],
        "@//:using_rocm": [
            ":rocm_cu",
        ],
    }) + any_cuda_deps + torch_deps(),
    copts = any_cuda_copts(),
    include_prefix = "src",
    linkstatic = True,
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
#include "rtp_llm/cpp/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_launch.h"
namespace rtp_llm {
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
    deps = any_cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:system_utils",
        "//rtp_llm/cpp/model_utils:model_utils",
    ],
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
        ":mmha_hdrs",
        ":attn_utils",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:system_utils",    ],
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
        ":mmha_hdrs",
        ":attn_utils",
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
        ":mmha_hdrs",
        ":attn_utils",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:system_utils",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_shared_library(
    name = "mmha1",
    roots = [":mmha_cu_1"],
    preloaded_deps = any_cuda_deps + [
        ":mmha_hdrs",
        ":attn_utils",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:system_utils",
    ],
)

cc_shared_library(
    name = "mmha2",
    roots = [":mmha_cu_2"],
    preloaded_deps = any_cuda_deps + [
        ":mmha_hdrs",
        ":attn_utils",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:system_utils",
    ],
)

cc_shared_library(
    name = "dmmha",
    roots = [":dmmha_cu"],
    preloaded_deps = any_cuda_deps + [
        ":mmha_hdrs",
        ":attn_utils",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:system_utils",
    ],
)
