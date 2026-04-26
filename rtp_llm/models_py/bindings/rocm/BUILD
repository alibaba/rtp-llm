load("//:def.bzl", "copts", "rocm_copts")
load("//bazel:arch_select.bzl", "torch_deps")

# --- Targets moved from cpp/rocm/ ---

cc_library(
    name = "rocm_types_hdr",
    hdrs = [
        "cuda_shims.h",
        "amd_bfloat16.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocm_host_utils",
    srcs = [
        "hip_host_utils.cc",
    ],
    hdrs = [
        "hip_host_utils.h",
        "cuda_shims.h",
        "amd_bfloat16.h",
    ],
    deps = [
        "@local_config_rocm//rocm:rocm",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:rocblas",
        "@local_config_rocm//rocm:hipblaslt",
        "//rtp_llm/models_py/bindings/core:op_data",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings/core:types",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@havenask//aios/autil:scope",
    ],
    copts = rocm_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocm",
    deps = [
        ":rocm_host_utils",
    ],
    copts = rocm_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "speculative_sampling",
    srcs = [
        "speculative_sampling/sampling_ops.cu",
        "speculative_sampling/sampling.cu"
    ],
    hdrs = [
        "speculative_sampling/sampling.cuh",
        "speculative_sampling/utils.cuh",
        "speculative_sampling/vec_dtypes.cuh"
    ],
    deps = torch_deps(),
    copts = rocm_copts() + ["-DHIP_ENABLE_WARP_SYNC_BUILTINS=1"],
    strip_include_prefix = "speculative_sampling",
    visibility = ["//visibility:public"],
)

# --- Bindings targets ---

cc_library(
    name = "rocm_bindings",
    srcs = glob([
        "*.cc",
    ], exclude = [
        "RegisterRocmOps.cc",
        "hip_host_utils.cc",
    ]),
    hdrs = glob([
        "*.h",
    ], exclude = [
        "hip_host_utils.h",
        "cuda_shims.h",
        "amd_bfloat16.h",
    ]),
    copts = rocm_copts(),
    deps = torch_deps() + [
        "//rtp_llm/models_py/bindings/core:types_hdr",
        "//rtp_llm/models_py/bindings/common/kernels:kernels_kv_cache",
        "//rtp_llm/models_py/bindings/rocm/kernels:rocm_atrex_pa",
        "//rtp_llm/cpp/utils:debug_utils",
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/models_py/bindings/common:common",
        "//rtp_llm/models_py/bindings/rocm/kernels:rocm_trtllm_allreduce_fusion",
    ],
)

# These bindings are to be reused by ppu bindings.
cc_library(
    name = "rocm_bindings_base",
    hdrs = [
        "RegisterBaseBindings.hpp",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings:register_ops_hdr",
        ":rocm_bindings",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_library(
    name = "rocm_attn_bindings_base",
    hdrs = [
        "RegisterAttnOpBindings.hpp"
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings:register_ops_hdr",
        ":rocm_bindings",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_library(
    name = "rocm_bindings_register",
    srcs = [
        "RegisterRocmOps.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings:register_ops_hdr",
        ":rocm_bindings_base",
        ":rocm_attn_bindings_base",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
