load("//:def.bzl", "copts", "rocm_copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_library(
    name = "dcu_types_hdr",
    hdrs = [
        "cuda_shims.h",
        "amd_bfloat16.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "dcu_host_utils",
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
        ":dcju_host_utils",
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

cc_library(
    name = "dcu_bindings",
    srcs = glob([
        "*.cc",
    ], exclude = [
        "RegisterDcuOps.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    copts = rocm_copts(),
    deps = torch_deps() + [
        "//rtp_llm/models_py/bindings/core:types_hdr",
	"//rtp_llm/models_py/bindings/common/kernels:kernels_kv_cache",
        "//rtp_llm/models_py/bindings/dcu/kernels:rocm_cu",
        "//rtp_llm/cpp/utils:debug_utils",
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/models_py/bindings/common:common",
    ],
)

cc_library(
    name = "dcu_bindings_base",
    hdrs = [
        "RegisterBaseBindings.hpp",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings:register_ops_hdr",
        ":dcu_bindings",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_library(
    name = "dcu_bindings_register",
    srcs = [
        "RegisterDcuOps.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings:register_ops_hdr",
        ":dcu_bindings_base",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

