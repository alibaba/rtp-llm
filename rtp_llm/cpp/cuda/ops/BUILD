load("//:def.bzl", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "cuda_register", "trt_plugins", "flashinfer_deps", "flashmla_deps")
load("@//bazel:arch_select.bzl", "torch_deps")

cuda_register()
trt_plugins()
flashinfer_deps()
flashmla_deps()

config_setting(
    name = "use_accl_ep",
    values = {
        "define": "use_accl_ep=1",
    },
)

# exec_ctx_fwd provides ExecOps.h header + transitive deps for all platforms.
# The actual method implementations are linked via gpu_base (CUDA) or exec_ctx_rocm (ROCm).
cc_library(
    name = "exec_ctx_fwd",
    deps = ["//rtp_llm/cpp/core:exec_ops_hdr"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gpu_base",
    hdrs = [
        "CudaFlashInfer.h",
    ] + select({
        "@//:using_cuda12": [
            "CudaXqa.h"
        ],
        "//conditions:default": [],
    }),
    srcs = [
        "CudaFlashInfer.cc",
    ] + ["//rtp_llm/cpp/core:devices_base_srcs"] + select({
        "@//:using_cuda12": [
            "CudaXqa.cc"
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//rtp_llm/cpp/core:device_data",
        "//rtp_llm/cpp/core:devices_base",
        "//rtp_llm/cpp/core:distributed_comm_hdr",
        "//rtp_llm/cpp/core:event",
        "//rtp_llm/cpp/core:exec_ops_hdr",
        "//rtp_llm/cpp/core:op_data",
        "//rtp_llm/cpp/core:type_convert",
        "//rtp_llm/cpp/utils:signal_utils",
        "//rtp_llm/cpp/utils:kv_cache_utils",
        "//rtp_llm/cpp/config:static_config",
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/cpp/cuda:cuda",
        "//rtp_llm/cpp/cuda:launch_utils",
        "//rtp_llm/cpp/cuda:quantize_utils",
        "//rtp_llm/cpp/core:types_hdr",
        "//rtp_llm/cpp/core:torch_event",
        "//rtp_llm/models_py/bindings/common/kernels:kernels_cu",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/cuda/deep_gemm:deep_gemm_plugins_impl",
        "//rtp_llm/cpp/pybind:th_utils",
        "//rtp_llm/cpp/cuda:trt_plugins",
        "//rtp_llm/cpp/models:stats",
        "//rtp_llm/cpp/models/models_weight:weights_define",
        "//rtp_llm/cpp/cache:cache_group_type",
        "@havenask//aios/autil:stack_tracer",
        "@havenask//aios/autil:string_helper",
        "@havenask//aios/autil:scope",
        "@local_config_cuda//cuda:cuda_headers",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store_interface",
        "//rtp_llm/cpp/utils:profiling_scope",
        ":flashinfer",
        "//3rdparty/flashinfer:flashinfer",
        ":flashmla",
    ] + select({
        "@//:using_cuda12": [
            "//3rdparty/xqa:xqa"
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
)

# gpu_register is kept as an alias for backward compatibility with internal builds.
cc_library(
    name = "gpu_register",
    deps = [
        ":gpu_base",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
)

cc_library(
    name = "torch_beam_search_op_impl",
    hdrs = ["BeamSearchOp.h"],
    deps = torch_deps(),
    visibility = ["//visibility:public"],
    copts = copts(),
)

cc_library(
    name = "standalone_ops",
    hdrs = ["StandaloneOps.h"],
    srcs = ["StandaloneOps.cc"],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings/common/kernels:kernels_activation",
        "//rtp_llm/models_py/bindings/common/kernels:kernels_sampling",
        "@local_config_cuda//cuda:cuda_headers",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

cc_library(
    name = "cuda_impl",
    srcs = ["//rtp_llm/cpp/core:exec_ctx_impl_srcs"],
    deps = [
        ":gpu_base",
        ":torch_beam_search_op_impl",
        "//rtp_llm/models_py/bindings/cuda/kernels:scaled_fp8_quant",
        "//rtp_llm/cpp/utils:debug_utils",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
)
