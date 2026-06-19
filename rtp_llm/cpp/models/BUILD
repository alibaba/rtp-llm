load("//:def.bzl", "copts")

# Sample infos - model sampling data structures
cc_library(
    name = "sample_infos",
    hdrs = [
        "SampleInfos.h",
        "SpecLogitsProcessorTypes.h",
    ],
    deps = [
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/utils:tensor_debug_utils",
        "//rtp_llm/cpp/pybind:py_utils",
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

# Logits processor - model output processing.
# The grammar-constrained processors are co-located here so that
# LogitsProcessorFactory can dispatch structured-output requests directly
# without a runtime plug-in hook (see dsv4 composition root).
cc_library(
    name = "logits_processor",
    hdrs = [
        "logits_processor/BaseLogitsProcessor.h",
        "logits_processor/LogitsProcessorException.h",
        "logits_processor/SpecLogitsProcessor.h",
        "logits_processor/MultiSeqLogitsProcessor.h",
        "logits_processor/LogitsProcessorStates.h",
        "logits_processor/ThinkModeLogitsProcessor.h",
        "logits_processor/TreeLogitsProcessor.h",
        "logits_processor/RecommendationLogitsProcessor.h",
        "logits_processor/LogitsProcessorFactory.h",
        "logits_processor/GrammarLogitsProcessor.h",
        "logits_processor/ReasoningGrammarLogitsProcessor.h",
        "logits_processor/SpecLogitsVerifyRunner.h",
        "logits_processor/BitmaskUtils.h",
    ],
    srcs = [
        "logits_processor/BaseLogitsProcessor.cc",
        "logits_processor/MultiSeqLogitsProcessor.cc",
        "logits_processor/LogitsProcessorStates.cc",
        "logits_processor/ThinkModeLogitsProcessor.cc",
        "logits_processor/TreeLogitsProcessor.cc",
        "logits_processor/RecommendationLogitsProcessor.cc",
        "logits_processor/LogitsProcessorFactory.cc",
        "logits_processor/GrammarLogitsProcessor.cc",
        "logits_processor/ReasoningGrammarLogitsProcessor.cc",
        "logits_processor/SpecLogitsVerifyRunner.cc",
    ],
    deps = [
        ":sample_infos",
        ":dfa_utils",
        "//rtp_llm/cpp/engine_base/grammar:grammar_cpp_lib",
        "//rtp_llm/cpp/engine_base/stream:generate_types",
        "//rtp_llm/cpp/engine_base/stream:generate_config",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/pybind:py_utils",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:profiling_scope",
    ] + select({
        "//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda/ops:standalone_ops",
        ],
        "@//:using_rocm": [],
        "//conditions:default": [],
    }),
    copts = copts(),
    visibility = ["//visibility:public"],
)

# Back-compat alias: callers historically depended on :grammar_logits_processor;
# the grammar processors now live in :logits_processor.
alias(
    name = "grammar_logits_processor",
    actual = ":logits_processor",
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
        ":context_parallel_interface",
        "//:rtp_compute_ops",
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "models",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        ":sample_infos",
        ":logits_processor",
        ":dfa_utils",
        ":context_parallel",
        "//:rtp_compute_ops",
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
        "//rtp_llm/cpp/pybind:py_utils",
        "//rtp_llm/cpp/engine_base/stream:complete_token_ids",
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/models_py/bindings/core:cache_store_async_writer",
        "//rtp_llm/cpp/utils:debug_utils",
        "//rtp_llm/cpp/utils:profiling_scope",
        # cuda_graph_base is platform-agnostic struct definitions used by
        # PyWrappedModel.h (GraphParams/CudaGraphState); safe on all platforms.
        # cuda_graph_impl/cuda_graph_hdrs_lib pull in the device-specific
        # runner srcs/headers and only link on CUDA/ROCm.
        "//rtp_llm/cpp/cuda_graph:cuda_graph_base",
        "//rtp_llm/models_py/bindings/common:fuse_copy_op",
        "//rtp_llm/cpp/multimodal_processor:multimodal_types",
    ] + select({
        "//:using_cuda": [
            "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
            "//rtp_llm/cpp/cuda_graph:cuda_graph_hdrs_lib",
        ],
        "@//:using_rocm": [
            "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
            "//rtp_llm/cpp/cuda_graph:cuda_graph_hdrs_lib",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

# Expert statistics - monitoring component
cc_library(
    name = "stats",
    hdrs = [
        "eplb/stats/ExpertStats.h",
    ],
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
