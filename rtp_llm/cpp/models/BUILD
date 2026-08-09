load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

# Sample infos - model sampling data structures
cc_library(
    name = "sample_infos",
    hdrs = [
        "SampleInfos.h",
    ],
    deps = [
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/utils:tensor_debug_utils",
    ],
    visibility = ["//visibility:public"],
)

# Position IDs generator - model inference component
cc_library(
    name = "position_ids_generator",
    hdrs = [
        "position_ids/PositionIdsGenerator.h",
    ],
    srcs = [
        "position_ids/PositionIdsGenerator.cc",
    ],
    deps = [
        "//:rtp_compute_ops"
    ],
    visibility = ["//visibility:public"],
)

# DFA utilities - includes PrefixToCandidateTokens functionality
cc_library(
    name = "dfa_utils",
    hdrs = [
        "logits_processor/DFAUtil.h",
        "logits_processor/PrefixToCandidateTokens.h",
    ],
    deps = [
        "//:rtp_compute_ops",
        "@rapidjson//:rapidjson",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

# Grammar processors co-located so LogitsProcessorFactory dispatches without a runtime plug-in hook.
cc_library(
    name = "logits_processor",
    hdrs = [
        "logits_processor/BaseLogitsProcessor.h",
        "logits_processor/SpecLogitsProcessor.h",
        "logits_processor/MultiSeqLogitsProcessor.h",
        "logits_processor/LogitsProcessorStates.h",
        "logits_processor/ThinkModeLogitsProcessor.h",
        "logits_processor/TreeLogitsProcessor.h",
        "logits_processor/RecommendationLogitsProcessor.h",
        "logits_processor/LogitsProcessorFactory.h",
        "logits_processor/GrammarLogitsProcessor.h",
        "logits_processor/SpecLogitsVerifyRunner.h",
        "logits_processor/BitmaskUtils.h",
    ],
    srcs = [
        "logits_processor/BaseLogitsProcessor.cc",
        "logits_processor/BitmaskUtils.cc",
        "logits_processor/MultiSeqLogitsProcessor.cc",
        "logits_processor/LogitsProcessorStates.cc",
        "logits_processor/ThinkModeLogitsProcessor.cc",
        "logits_processor/TreeLogitsProcessor.cc",
        "logits_processor/RecommendationLogitsProcessor.cc",
        "logits_processor/LogitsProcessorFactory.cc",
        "logits_processor/GrammarLogitsProcessor.cc",
        "logits_processor/SpecLogitsVerifyRunner.cc",
    ],
    deps = [
        ":sample_infos",
        ":dfa_utils",
        "//rtp_llm/cpp/engine_base/grammar:grammar_cpp_lib",
        "//rtp_llm/cpp/cuda_graph:cuda_graph_hdrs_lib",
        "//rtp_llm/cpp/engine_base/stream:generate_types",
        "//rtp_llm/cpp/engine_base/stream:generate_config",
        "//rtp_llm/cpp/multimodal_processor:multimodal_types",
        "//rtp_llm/cpp/config:model_config",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/pybind:py_utils",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:profiling_scope",
        "//rtp_llm/cpp/core:copy_ops_hdrs",
        "@havenask//aios/autil:json",
    ] + select({
        "//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda/ops:standalone_ops",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
    visibility = ["//visibility:public"],
)

# Context Parallel token layout shared by scheduling and execution.
cc_library(
    name = "context_parallel_layout",
    hdrs = [
        "context_parallel/ZigzagTokenLayout.h",
    ],
    srcs = [
        "context_parallel/ZigzagTokenLayout.cc",
    ],
    visibility = ["//visibility:public"],
)

# Context Parallel Interface - base interface for context parallel processors
cc_library(
    name = "context_parallel_interface",
    hdrs = [
        "context_parallel/ContextParallelProcessorBase.h",
    ],
    deps = [
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "context_parallel",
    hdrs = [
        "context_parallel/ZigzagProcessor.h",
    ],
    srcs = [
        "context_parallel/ContextParallelProcessorBase.cc",
        "context_parallel/ZigzagProcessor.cc",
        "context_parallel/ContextParallelProcessorFactory.cc",
    ],
    deps = [
        ":context_parallel_layout",
        ":context_parallel_interface",
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/comm:collective_backend_hdrs",
        "//rtp_llm/cpp/multimodal_processor:multimodal_input_utils",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings/core:op_data",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "model_inputs_logger",
    srcs = ["ModelInputsLogger.cc"],
    hdrs = ["ModelInputsLogger.h"],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings/core:op_data",
        "@havenask//aios/autil:env_util",
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
    ] + torch_deps(),
    visibility = ["//rtp_llm/cpp/models/test:__pkg__"],
)

cc_library(
    name = "models",
    hdrs = glob(
        ["*.h"],
        exclude = [
            "ModelInputsLogger.h",
            "SamplingOps.h",
            "SamplingTypes.h",
        ],
    ),
    srcs = glob(
        ["*.cc"],
        exclude = [
            "ModelInputsLogger.cc",
            "SamplingOps.cc",
        ],
    ),
    deps = [
        ":model_inputs_logger",
        ":sample_infos",
        ":sampling_ops",
        ":stats",
        ":logits_processor",
        ":dfa_utils",
        ":context_parallel",
        "//:rtp_compute_ops",
        "//rtp_llm/models_py/bindings/core:tensor_holder",
        "//rtp_llm/cpp/pybind:py_utils",
        "//rtp_llm/cpp/engine_base/stream:complete_token_ids",
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/metrics",
        "//rtp_llm/models_py/bindings/core:cache_store_async_writer",
        "//rtp_llm/cpp/utils:debug_utils",
        "//rtp_llm/cpp/utils:profiling_scope",
        "//rtp_llm/cpp/utils:tensor_debug_utils",
        "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
        "//rtp_llm/cpp/cuda_graph:cuda_graph_base",
        "//rtp_llm/cpp/cuda_graph:cuda_graph_hdrs_lib",
        "//rtp_llm/cpp/multimodal_processor:multimodal_types",
        "//rtp_llm/cpp/comm:collective_backend_hdrs",
        "//rtp_llm/cpp/core:copy_ops_hdrs",
        "//rtp_llm/cpp/runtime:cuda_runtime_hdrs",
    ] + select({
        "//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda/kernels:attention_input_metadata",
            "//rtp_llm/models_py/bindings/cuda:bf16_gemm_op",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "sampling_types",
    hdrs = ["SamplingTypes.h"],
    deps = torch_deps(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "sampling_ops",
    hdrs = ["SamplingOps.h"],
    deps = [
        ":sampling_types",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "sampling_ops_impl",
    srcs = ["SamplingOps.cc"],
    deps = [":sampling_ops"],
    alwayslink = True,
    visibility = [
        "//rtp_llm/cpp/pybind:__pkg__",
        "//rtp_llm/models_py/bindings/core:__pkg__",
    ],
)

# Expert statistics - monitoring component
cc_library(
    name = "stats_hdrs",
    hdrs = [
        "eplb/stats/ExpertStats.h",
        "eplb/stats/MoeExpertStats.h",
    ],
    deps = torch_deps(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "stats_impl",
    srcs = [
        "eplb/stats/MoeExpertStats.cc",
    ],
    deps = [":stats_hdrs"] + torch_deps(),
    alwayslink = True,
    visibility = [
        "//rtp_llm/cpp/pybind:__pkg__",
        "//rtp_llm/models_py/bindings/core:__pkg__",
    ],
)

alias(
    name = "stats",
    actual = ":stats_hdrs",
    visibility = ["//visibility:public"],
)

# EPLB configuration
cc_library(
    name = "eplb_config",
    hdrs = [
        "eplb/EplbConfig.h",
    ],
    visibility = ["//visibility:public"],
)

# Expert Load Balancer
cc_library(
    name = "eplb",
    hdrs = [
        "eplb/ExpertBalancer.h",
        "eplb/ExpertBalancerPythonWrapper.h",
    ],
    srcs = [
        "eplb/ExpertBalancer.cc",
        "eplb/ExpertBalancerPythonWrapper.cc",
    ],
    deps = [
        ":models",
        ":dfa_utils",
        "//rtp_llm/cpp/config:eplb_config",
        "//rtp_llm/cpp/metrics",
        "//rtp_llm/cpp/pybind:py_utils",
        "//rtp_llm/cpp/cache:cache_types",
    ],
    visibility = ["//visibility:public"],
)
