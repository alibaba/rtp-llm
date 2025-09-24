load("//:def.bzl", "copts", "cuda_copts", "rocm_copts", "any_cuda_copts", "gen_cpp_code")
load("//bazel:arch_select.bzl", "torch_deps")
load("@rules_cc//examples:experimental_cc_shared_library.bzl", "cc_shared_library")
package(default_visibility = ["//rtp_llm:__subpackages__"])

cuda_deps = [
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
]

rocm_deps = [
    "//rtp_llm/cpp/rocm:rocm_host_utils",
    "@local_config_rocm//rocm:rocm_headers",
    "@local_config_rocm//rocm:hip",
    "//rtp_llm/cpp/rocm:rocm_types_hdr",
]

any_cuda_deps = select({
    "@//:using_cuda": cuda_deps,
    "@//:using_rocm": rocm_deps,
    "//conditions:default": [
    ],
})

preloaded_deps = any_cuda_deps + [
    ":mmha_hdrs",
    "//rtp_llm/cpp/kernels:attn_utils",
    "//rtp_llm/cpp/utils:core_utils",
    "//rtp_llm/cpp/cuda:cuda_host_utils",
    "//rtp_llm/cpp/cuda:cuda_utils_cu",
    "//rtp_llm/cpp/cuda:launch_utils",
]

# =============================================================================
# MMHA (Multi-head Masked Attention) related targets
# =============================================================================

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


# MMHA headers
cc_library(
    name = "mmha_hdrs",
    hdrs = [
        "decoder_masked_multihead_attention.h",
    ],
    deps = any_cuda_deps,
    copts = any_cuda_copts(),
    visibility = ["//visibility:public"],
)

# MMHA implementation 1
cc_library(
    name = "mmha_cu_1",
    srcs = [
        ":mmha_inst_1"
    ],
    hdrs = glob([
        "*.h",
    ], exclude=[
        "decoder_masked_multihead_attention.h",
    ]),
    deps = any_cuda_deps + [
        ":mmha_hdrs",
        "//rtp_llm/cpp/kernels:attn_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
    ],
    copts = any_cuda_copts(),
    visibility = ["//visibility:public"],
)

# MMHA implementation 2
cc_library(
    name = "mmha_cu_2",
    srcs = [
        ":mmha_inst_2"
    ],
    hdrs = glob([
        "*.h",
    ], exclude=[
        "decoder_masked_multihead_attention.h",
    ]),
    deps = any_cuda_deps + [
        ":mmha_hdrs",
        "//rtp_llm/cpp/kernels:attn_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
    ],
    copts = any_cuda_copts(),
    visibility = ["//visibility:public"],
)

# DMMHA implementation
cc_library(
    name = "dmmha_cu",
    srcs = glob([
        "*.cu",
    ]),
    hdrs = glob([
        "*.h",
    ], exclude=[
        "decoder_masked_multihead_attention.h",
    ]),
    deps = any_cuda_deps + [
        ":mmha_hdrs",
        "//rtp_llm/cpp/kernels:attn_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
    ],
    copts = any_cuda_copts(),
    visibility = ["//visibility:public"],
)

# Shared libraries
cc_shared_library(
    name = "mmha1",
    roots = [":mmha_cu_1"],
    preloaded_deps = preloaded_deps,
)

cc_shared_library(
    name = "mmha2",
    roots = [":mmha_cu_2"],
    preloaded_deps = preloaded_deps,
)

cc_shared_library(
    name = "dmmha",
    roots = [":dmmha_cu"],
    preloaded_deps = preloaded_deps,
)

# Aggregated target for kernels_cu dependency
cc_library(
    name = "mmha",
    deps = [":mmha_hdrs"],
    srcs = [
        ":mmha1",
        ":mmha2", 
        ":dmmha",
    ],
    visibility = ["//visibility:public"],
)
