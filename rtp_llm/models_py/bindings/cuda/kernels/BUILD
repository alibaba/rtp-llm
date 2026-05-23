load("//:def.bzl", "copts", "cuda_copts", "any_cuda_copts")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch")
load("//bazel:arch_select.bzl", "torch_deps")
package(default_visibility = ["//rtp_llm:__subpackages__"])

sm100_cuda_copts = copts() + cuda_default_copts_without_arch() + [
    '--cuda-include-ptx=sm_100', '--cuda-gpu-arch=sm_100',
    '--cuda-include-ptx=sm_100a', '--cuda-gpu-arch=sm_100a',
]

cuda_deps = [
    "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
    ":cuda_utils_common",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
]

cc_library(
    name = "cuda_utils_common",
    hdrs = [
        "vec_dtypes.cuh",
        "util.h",
    ],
    deps = [
        "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_fused_qk_rmsnorm",
    srcs = ["fused_qk_rmsnorm.cu"],
    hdrs = ["fused_qk_rmsnorm.h"],
    deps = cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings/cuda:cuda_utils_cu",
        "//rtp_llm/models_py/bindings/cuda:launch_utils",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_no_aux_tc",
    srcs = ["no_aux_tc_kernels.cu"],
    hdrs = ["no_aux_tc_kernels.h"],
    deps = [
        "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        "//rtp_llm/models_py/bindings/cuda:launch_utils",
        "//rtp_llm/models_py/bindings/cuda:cuda_utils_cu",
        ":cuda_utils_common",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = cuda_copts(),
)

cc_library(
    name = "scaled_fp8_quant",
    srcs = ["scaled_fp8_quant.cu"],
    hdrs = [
        "scaled_fp8_quant.h",
        "scaled_fp8_quant_utils.h",
    ],
    deps = [
        "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        "//rtp_llm/models_py/bindings/cuda:cuda_utils_cu",
        "//rtp_llm/models_py/bindings/cuda:launch_utils",
        ":cuda_utils_common",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "scaled_fp4_quant",
    srcs = ["scaled_fp4_quant.cu"],
    hdrs = ["scaled_fp4_quant.h"],
    deps = [
        "//rtp_llm/models_py/bindings/cuda/cutlass:fp4_gemm_cu",
        "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        ":cuda_utils_common",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    copts = sm100_cuda_copts + [
        "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_quantization",
    srcs = [
        "per_token_group_quant_8bit.cu",
        "per_token_group_quant_8bit_v2.cu",
    ],
    hdrs = [
        "per_token_group_quant_8bit.h",
        "per_token_group_quant_8bit_v2.h",
    ],
    deps = [
        "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        "//rtp_llm/models_py/bindings/cuda:cuda_utils_cu",
        ":cuda_utils_common",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = cuda_copts(),
)

cc_library(
    name = "fast_topk",
    srcs = ["fast_topk.cu"],
    hdrs = ["fast_topk.h"],
    deps = [
        "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        ":cuda_utils_common",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "xgrammar_kernels",
    srcs = ["xgrammar_kernels.cu"],
    hdrs = ["xgrammar_kernels.h"],
    deps = cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "persistent_topk",
    srcs = ["persistent_topk.cu"],
    hdrs = [
        "persistent_topk.h",
        "persistent_topk.cuh",
    ],
    deps = [
        "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        ":cuda_utils_common",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "attention_input_metadata",
    srcs = ["attention_input_metadata.cu"],
    hdrs = ["attention_input_metadata.h"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

# Vendored from vLLM (csrc/persistent_topk.cuh + csrc/topk.cu @ b55d830).
# Replaces torch.topk on the DSv4 indexer decode hot path:
#   K=512 / T_max=2048: torch.topk ~30us → this kernel ~5-10us.
# See dsv4_persistent_topk.h for the contract. Distinct from the legacy
# ``persistent_topk`` (rtp_llm:: namespace) above — the dsv4 variant lives
# in ``torch_ext::`` and writes -1 padding past per-row lengths inline.
cc_library(
    name = "dsv4_persistent_topk",
    srcs = ["dsv4_persistent_topk.cu"],
    hdrs = [
        "dsv4_persistent_topk.h",
        "dsv4_persistent_topk.cuh",
    ],
    deps = [
        "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        ":cuda_utils_common",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_graph_prepare",
    srcs = ["cuda_graph_prepare.cu"],
    hdrs = ["cuda_graph_prepare.h"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

# Per-row TopK for the prefill phase of the DeepSeek-V4 indexer.
# Vendored from vLLM (csrc/sampler.cu::top_k_per_row_prefill). Hybrid:
# insertion-sort blocks for the first 12288 rows, radix-select beyond.
cc_library(
    name = "dsv4_top_k_per_row_prefill",
    srcs = ["dsv4_top_k_per_row_prefill.cu"],
    hdrs = [
        "dsv4_top_k_per_row_prefill.h",
        "dsv4_top_k_per_row_prefill.cuh",
    ],
    deps = [
        "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        ":cuda_utils_common",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mtp_target_verify_prepare",
    srcs = ["mtp_target_verify_prepare.cu"],
    hdrs = ["mtp_target_verify_prepare.h"],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mha_paged_attn_plan",
    srcs = ["mha_paged_attn_plan.cu"],
    hdrs = ["mha_paged_attn_plan.h"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mla_quant_kernel",
    srcs = ["mla_quant_kernel.cu"],
    hdrs = ["mla_quant_kernel.h"],
    deps = [
        "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        "//rtp_llm/models_py/bindings/cuda:cuda_utils_cu",
        ":cuda_utils_common",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_cu",
    deps = [
        ":cuda_no_aux_tc",
        ":cuda_quantization",
        ":cuda_fused_qk_rmsnorm",
        ":mla_quant_kernel",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "user_buffers",
    srcs = ["user_buffer/user_buffers.cu"],
    hdrs = ["user_buffer/user_buffers.h"],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_speculative_sampling",
    srcs = glob(["speculative_sampling/*.cu"]),
    hdrs = glob([
        "speculative_sampling/*.h",
        "speculative_sampling/*.cuh",
    ]),
    deps = cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings/cuda:launch_utils",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kernels_mla",
    srcs = glob(["mla_kernels/*.cu"]),
    hdrs = glob([
        "mla_kernels/*.h",
        "mla_kernels/*.cuh",
    ]),
    deps = cuda_deps + [
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = any_cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)
