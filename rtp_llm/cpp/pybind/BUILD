load("//:def.bzl", "copts")
load(
    "@arch_config//:arch_select.bzl",
    "select_py_bindings",
    "torch_deps",
)

package(default_visibility = ["//:__subpackages__"])

py_test(
    name = "config_pickle_test",
    srcs = ["config_pickle_test.py"],
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer_config",
    ],
    deps = ["//rtp_llm:ops"],
)

# Process-state implementations must be linked exactly once, into
# librtp_compute_ops.so. CUDA sampling is provided transitively by
# select_py_bindings() -> cuda_bindings -> cuda_impl; ROCm sampling is listed
# explicitly below. Do not add sampling_ops_impl to the CUDA branch because it
# would duplicate the alwayslink sources already in cuda_impl.
_COMPUTE_IMPL_DEPS = [
    "//rtp_llm/cpp/cache:cache_store_writer_impl",
    "//rtp_llm/cpp/comm:collective_backend_impl",
    "//rtp_llm/cpp/core:copy_ops_impl",
    "//rtp_llm/cpp/models:sampling_ops_impl",
    "//rtp_llm/cpp/models:stats_impl",
    "//rtp_llm/cpp/runtime:cuda_runtime_impl",
    "//rtp_llm/models_py/bindings/core:weight_preprocess_impl",
]

cc_library(
    name = "py_utils",
    hdrs = [
        "PyUtils.h",
    ],
    srcs = [
        "PyUtils.cc",
    ],
    deps = [
        "//:rtp_compute_ops",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "th_transformer_config_lib",
    srcs = [
        "common/blockUtil.cc",
        "ConfigInit.cc",
    ],
    hdrs = [
        "common/blockUtil.h",
    ],
    deps = [
        ":py_utils",
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/multimodal_processor:multimodal_input_class",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/config:model_config",  # Add ModelConfig implementation
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/cpp/config:eplb_config",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:hash_util",
    ],
    copts = copts(),
    alwayslink = True,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "th_compute_lib",
    srcs = [
        "ComputeInit.cc",
    ],
    hdrs = [],
    deps = [
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "//rtp_llm/cpp/cache:block_info",
        "//rtp_llm/cpp/cache:cache_config",
        "//rtp_llm/cpp/cache:cp_slot_mapper",
        "//rtp_llm/cpp/comm:collective_backend_pybind_hdrs",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/config:model_config",
        "//rtp_llm/cpp/runtime:cuda_runtime_hdrs",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:debug_utils",
        "//rtp_llm/cpp/utils:device_perf_wrapper",
        "//rtp_llm/models_py/bindings/core:weight_preprocess_hdrs",
        "@havenask//aios/autil:base64",
        "@havenask//aios/autil:zlib",
    ] + torch_deps() + select_py_bindings() + select({
        "@//:using_cuda": _COMPUTE_IMPL_DEPS,
        "@//:using_rocm": _COMPUTE_IMPL_DEPS,
        "//conditions:default": [],
    }) + select({
        "@//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
        ],
        "@//:using_rocm": [
            "//rtp_llm/models_py/bindings/core:sampling_ops_impl",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
    alwayslink = True,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "th_transformer_lib",
    srcs = [
        "init.cc",
        "common/InitEngineOps.cc",
        "multi_gpu_gpt/RtpEmbeddingOp.cc",
        "multi_gpu_gpt/RtpLLMOp.cc",
    ],
    hdrs = [
        "common/InitEngineOps.h",
        "common/blockUtil.h",
        "multi_gpu_gpt/RtpEmbeddingOp.h",
        "multi_gpu_gpt/RtpLLMOp.h",
    ],
    deps = [
        "//rtp_llm/cpp/api_server:http_api_server",
        "//rtp_llm/cpp/model_rpc:model_rpc_server",
        "//rtp_llm/cpp/engine_base/schedulers:schedulers",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/telemetry",
        "//rtp_llm/cpp/utils:signal_utils",
        "//:rtp_compute_ops",
        "@xgrammar//:xgrammar",
        # Process-state implementations belong only to th_compute_lib above.
        # th_transformer still owns stateless engine implementations selected
        # below, while runtime state is resolved through librtp_compute_ops.so.
    ] + torch_deps() + select({
        "//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
            "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
        ],
        "@//:using_rocm": [
            "//rtp_llm/models_py/bindings/core:sampling_ops_impl",
            "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
    alwayslink = True,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "th_transformer_gpu",
    srcs = [],
    deps = [] + torch_deps(),
    copts = copts(),
    alwayslink = True,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "th_utils",
    hdrs = [
        "th_utils.h",
    ],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)
