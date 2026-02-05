load("//:def.bzl", "rocm_copts")
load("//bazel:arch_select.bzl", "torch_deps")

config_setting(
    name = "enable_deep_ep",
    define_values = {"enable_deep_ep": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "use_accl_ep",
    define_values = {"use_accl_ep": "true"},
)

# Stub nvshmem.h for ROCm+deep_ep: avoid system nvshmem device headers that
# redefine __syncwarp and break with HIP. Must be first in deps when enable_deep_ep
# so #include <nvshmem.h> in deep_ep configs_hip.cuh finds this instead of /usr/include.
cc_library(
    name = "nvshmem_rocm_stub_hdrs",
    hdrs = [
        "nvshmem_rocm_stub/nvshmem.h",
        "nvshmem_rocm_stub/nvshmemx.h",
    ],
    strip_include_prefix = "nvshmem_rocm_stub",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "deep_ep_buffer",
    hdrs = [
        "DeepEPDefs.h",
        "DeepEPBuffer.h",
    ],
    srcs = [
        "DeepEPBuffer.cc",
    ],
    deps = select({
        ":enable_deep_ep": [":nvshmem_rocm_stub_hdrs"],
        "//conditions:default": [],
    }) + [
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:hip",
        "//rtp_llm/cpp/rocm:rocm_types_hdr",
        "//rtp_llm/cpp/kernels:rocm_utils",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/devices:device_utils",
        "//rtp_llm/cpp/pybind:th_utils",
        "@nvshmem_rocm//:nvshmem_rocm_hdrs",
    ] + select({
        ":use_accl_ep": [
            "//3rdparty/accl_ep:accl_ep",
        ],
        ":enable_deep_ep": ["@deep_ep_rocm//:deep_ep"],
        "//conditions:default": [],
    }) + torch_deps(),
    copts = rocm_copts() + [
        "-DUSE_ROCM=1",
        "-DDISABLE_INTERNODE=1",
        "-DFORCE_NVSHMEM_API=1",
    ] + select({
        ":enable_deep_ep": ["-DENABLE_DEEP_EP=1"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"]
)

cc_library(
    name = "rocm_impl",
    hdrs = glob([
        "*.h",
    ], exclude = [
        "DeepEPBuffer.h",
        "DeepEPDefs.h",
    ]),
    srcs = glob([
        "*.cc",
    ], exclude = [
        "DeepEPBuffer.cc",
    ]),
    deps = [
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:hip",
        "@local_config_rocm//rocm:rocblas",
        "@local_config_rocm//rocm:hipblaslt",
        "@local_config_rocm//rocm:rccl",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/devices:devices_base_impl",
        "//rtp_llm/cpp/core:allocator",
        "//rtp_llm/cpp/core:types_hdr",
        "//rtp_llm/cpp/rocm:rocm_types_hdr",
        "//rtp_llm/cpp/rocm:rocm_host_utils",
        "//rtp_llm/cpp/rocm:rocm_custom_ar",
	    "//rtp_llm/cpp/kernels:kernels_cu",
        "//rtp_llm/cpp/kernels:rocm_atrex_pa_hdrs",
        "//rtp_llm/cpp/rocm:speculative_sampling",
        "//rtp_llm/cpp/cuda:nccl_util",
        "//rtp_llm/cpp/pybind:th_utils",
        "//rtp_llm/cpp/devices/torch_impl:torch_beam_search_op_impl",
    ] + torch_deps() + select({
        "@//:using_aiter_src": [
            "@aiter_src//:module_aiter_enum",
            "@aiter_src//:module_quant",
            "@aiter_src//:module_smoothquant",
            "@aiter_src//:module_gemm_a8w8_blockscale",
            "@aiter_src//:module_gemm_a8w8_bpreshuffle",
            "@aiter_src//:module_gemm_a8w8",
            "@aiter_src//:module_moe_sorting",
            "@aiter_src//:module_moe_asm",
            "@aiter_src//:module_pa",
            "@aiter_src//:module_activation",
            "@aiter_src//:module_deepgemm",
            "@aiter_src//:module_rmsnorm",
            "@aiter_src//:module_norm",
            "@aiter_src//:module_mha_fwd",
            "@aiter_src//:module_fmha_v3_varlen_fwd",
            "@aiter_src//:module_moe_ck2stages",
        ],
        "//conditions:default": [
            "@aiter//:module_aiter_enum",
            "@aiter//:module_quant",
            "@aiter//:module_smoothquant",
            "@aiter//:module_gemm_a8w8_blockscale",
            "@aiter//:module_gemm_a8w8_bpreshuffle",
            "@aiter//:module_gemm_a8w8",
            "@aiter//:module_moe_sorting",
            "@aiter//:module_moe_asm",
            "@aiter//:module_pa",
            "@aiter//:module_activation",
            "@aiter//:module_rmsnorm",
            "@aiter//:module_norm",
            "@aiter//:module_mha_fwd",
            "@aiter//:module_fmha_v3_varlen_fwd",
            "@aiter//:module_moe_ck2stages",
            "@aiter//:module_deepgemm",
        ]}) + select({
        ":enable_deep_ep": [
            ":deep_ep_buffer",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    copts = rocm_copts() + [
        "-DUSE_ROCM=1",
        "-DDISABLE_INTERNODE=1",
        "-DFORCE_NVSHMEM_API=1",
    ] + select({
        ":enable_deep_ep": ["-DENABLE_DEEP_EP=1"],
        "//conditions:default": [],
    }),
    alwayslink = True,
)
