load("//:def.bzl", "copts")

# Sample infos - model sampling data structures
cc_library(
    name = "sample_infos",
    hdrs = [
        "SampleInfos.h",
    ],
    deps = [
        "//:rtp_compute_ops",
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
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

# Logits processor - model output processing
cc_library(
    name = "logits_processor",
    hdrs = [
        "logits_processor/BaseLogitsProcessor.h",
        "logits_processor/MultiSeqLogitsProcessor.h",
        "logits_processor/LogitsProcessorStates.h",
        "logits_processor/ThinkModeLogitsProcessor.h",
        "logits_processor/TreeLogitsProcessor.h",
        "logits_processor/LogitsProcessorFactory.h",
    ],
    srcs = [
        "logits_processor/BaseLogitsProcessor.cc",
        "logits_processor/MultiSeqLogitsProcessor.cc",
        "logits_processor/LogitsProcessorStates.cc",
        "logits_processor/ThinkModeLogitsProcessor.cc",
        "logits_processor/TreeLogitsProcessor.cc",
        "logits_processor/LogitsProcessorFactory.cc",
    ],
    deps = [
        ":sample_infos",
        ":dfa_utils",
        "//rtp_llm/cpp/engine_base/stream:generate_types",
        "//rtp_llm/cpp/engine_base/stream:generate_config",
        "//rtp_llm/cpp/multimodal_processor:multimodal_types",
        "//rtp_llm/cpp/pybind:py_utils",
    ],
    visibility = ["//visibility:public"],
)

# LoRA - model fine-tuning
cc_library(
    name = "lora",
    hdrs = [
        "lora/LoraManager.h",
    ],
    srcs = [
        "lora/LoraManager.cc",
    ],
    deps = [
        "//:rtp_compute_ops",
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
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/engine_base/stream:complete_token_ids",
        "//rtp_llm/cpp/cache:cache_types",
    ] + select({
        "@//:using_rocm": [
            "//rtp_llm/cpp/rocm:hip_capture_check",
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
        "//rtp_llm/cpp/pybind:py_utils",
        "//rtp_llm/cpp/cache:cache_types",
    ],
    visibility = ["//visibility:public"],
)
