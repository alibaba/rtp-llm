load("@arch_config//:arch_select.bzl", "cuda_register", "flashinfer_deps", "flashmla_deps", "torch_deps")
load("//:def.bzl", "copts", "cuda_copts")

cuda_register()

flashinfer_deps()

flashmla_deps()

cc_library(
    name = "flashinfer_runtime",
    visibility = ["//visibility:public"],
    deps = [":flashinfer"],
)

# exec_ctx_fwd provides ExecOps.h header + transitive deps for all platforms.
# The actual method implementations are linked via gpu_base (CUDA) or exec_ctx_rocm (ROCm).
cc_library(
    name = "exec_ctx_fwd",
    visibility = ["//visibility:public"],
    deps = ["//rtp_llm/models_py/bindings/core:exec_ops_hdr"],
)

cc_library(
    name = "gpu_base",
    srcs = select({
        "@//:using_cuda12": [
            "CudaXqa.cc",
        ],
        "//conditions:default": [],
    }),
    hdrs = select({
        "@//:using_cuda12": [
            "CudaXqa.h",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":flashmla",
        "//rtp_llm/cpp/cache:cache_group_type",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/config:static_config",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/cpp/model_utils",
        "//rtp_llm/cpp/models:stats",
        "//rtp_llm/cpp/models/models_weight:weights_define",
        "//rtp_llm/cpp/pybind:th_utils",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:kv_cache_utils",
        "//rtp_llm/cpp/utils:profiling_scope",
        "//rtp_llm/cpp/utils:signal_utils",
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/models_py/bindings/common/kernels:kernels_cu",
        "//rtp_llm/models_py/bindings/core:device_data",
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
        "//rtp_llm/models_py/bindings/core:op_data",
        "//rtp_llm/models_py/bindings/core:type_convert",
        "//rtp_llm/models_py/bindings/core:types_hdr",
        "//rtp_llm/models_py/bindings/cuda",
        "//rtp_llm/models_py/bindings/cuda:launch_utils",
        "@havenask//aios/autil:scope",
        "@havenask//aios/autil:stack_tracer",
        "@havenask//aios/autil:string_helper",
        "@local_config_cuda//cuda:cuda_headers",
    ] + select({
        "@//:using_cuda12": [
            "//3rdparty/xqa",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

# gpu_register is kept as an alias for backward compatibility with internal builds.
cc_library(
    name = "gpu_register",
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_base",
    ],
    alwayslink = 1,
)

cc_library(
    name = "standalone_ops",
    srcs = ["StandaloneOps.cc"],
    hdrs = ["StandaloneOps.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings/common/kernels:kernels_activation",
        "//rtp_llm/models_py/bindings/common/kernels:kernels_sampling",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "cuda_impl",
    srcs = ["//rtp_llm/models_py/bindings/core:exec_ctx_impl_srcs"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":flashinfer",
        ":gpu_base",
        "//3rdparty/flashinfer",
        "//rtp_llm/cpp/utils:debug_utils",
        "//rtp_llm/models_py/bindings/common/kernels:vocab_prune_kernels",
        "//rtp_llm/models_py/bindings/core:tensor_holder",
        "//rtp_llm/models_py/bindings/core/ops:torch_beam_search_op_impl",
        "//rtp_llm/models_py/bindings/cuda/kernels:kernels_speculative_sampling",
        "//rtp_llm/models_py/bindings/cuda/kernels:scaled_fp8_quant",
        "//rtp_llm/models_py/bindings/cuda/kernels/sampling",
    ],
    alwayslink = 1,
)
