load("@arch_config//:arch_select.bzl", "torch_deps")

cc_library(
    name = "normal_engine",
    hdrs = glob([
        "*.h",
        "speculative/*.h",
    ]) + [
        "pipeline/PPExecutor.h",
        "pipeline/PPSerialization.h",
        "pipeline/PPTransport.h",
        "pipeline/PPTypes.h",
    ],
    srcs = glob([
        "*.cc",
        "speculative/*.cc",
    ]) + [
        "pipeline/PPExecutor.cc",
        "pipeline/PPTransport.cc",
    ],
    deps = [
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/config:role_types",
        "//rtp_llm/cpp/cuda_graph:cuda_graph_hdrs_lib",
        "//rtp_llm/cpp/engine_base:executor",
        "//rtp_llm/cpp/metrics",
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/models:logits_processor",
        "//rtp_llm/cpp/engine_base:engine_base",
        "//rtp_llm/cpp/utils:device_pin",
        "//rtp_llm/cpp/engine_base/stream:generate_config",
        "//rtp_llm/cpp/engine_base:worker_status_info",
        "//rtp_llm/cpp/engine_base:profiler",
        "//rtp_llm/cpp/engine_base/schedulers:schedulers",
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/engine_base/system_prompt:system_prompt",
        "//rtp_llm/cpp/engine_base/system_prompt:system_prompt_constructor",
        "//rtp_llm/cpp/models:eplb",
        "//rtp_llm/cpp/cuda_graph:cuda_graph_base",
        "//rtp_llm/cpp/utils:tensor_debug_utils",
        "//rtp_llm/cpp/utils:profiling_scope",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/multimodal_processor:multimodal_input_utils",
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
        "//rtp_llm/models_py/bindings/core:op_data",
        "//rtp_llm/models_py/bindings/core:tensor_holder",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@havenask//aios/autil:json",
    ] + select({
        "//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda/ops:standalone_ops",
            "//rtp_llm/models_py/bindings/cuda/kernels:mtp_target_verify_prepare",
            "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
        ],
        "@//:using_rocm": [
            "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
        ],
        "//conditions:default": [],
    }) + torch_deps(),
    visibility = ["//visibility:public"],
)
