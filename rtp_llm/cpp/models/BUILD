load("//:def.bzl", "copts")

# Sample infos - model sampling data structures
cc_library(
    name = "sample_infos",
    hdrs = [
        "SampleInfos.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//:rtp_compute_ops",
    ],
)

# Position IDs generator - model inference component
cc_library(
    name = "position_ids_generator",
    srcs = [
        "position_ids/PositionIdsGenerator.cc",
    ],
    hdrs = [
        "position_ids/PositionIdsGenerator.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//:rtp_compute_ops",
    ],
)

# DFA utilities - includes PrefixToCandidateTokens functionality
cc_library(
    name = "dfa_utils",
    hdrs = [
        "logits_processor/DFAUtil.h",
        "logits_processor/PrefixToCandidateTokens.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//:rtp_compute_ops",
    ],
)

# Logits processor - model output processing
cc_library(
    name = "logits_processor",
    srcs = [
        "logits_processor/BaseLogitsProcessor.cc",
        "logits_processor/LogitsProcessorStates.cc",
        "logits_processor/MultiSeqLogitsProcessor.cc",
        "logits_processor/ThinkModeLogitsProcessor.cc",
        "logits_processor/TreeLogitsProcessor.cc",
    ],
    hdrs = [
        "logits_processor/BaseLogitsProcessor.h",
        "logits_processor/LogitsProcessorStates.h",
        "logits_processor/MultiSeqLogitsProcessor.h",
        "logits_processor/ThinkModeLogitsProcessor.h",
        "logits_processor/TreeLogitsProcessor.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":dfa_utils",
        ":sample_infos",
        "//rtp_llm/cpp/cache2",
        "//rtp_llm/cpp/engine_base/stream:generate_config",
        "//rtp_llm/cpp/engine_base/stream:generate_types",
        "//rtp_llm/cpp/multimodal_processor:multimodal_types",
        "//rtp_llm/cpp/pybind:py_utils",
    ],
)

# LoRA - model fine-tuning
cc_library(
    name = "lora",
    srcs = [
        "lora/LoraManager.cc",
    ],
    hdrs = [
        "lora/LoraManager.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//:rtp_compute_ops",
    ],
)

cc_library(
    name = "models",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":dfa_utils",
        ":logits_processor",
        ":sample_infos",
        "//:rtp_compute_ops",
    ],
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
    srcs = [
        "eplb/ExpertBalancer.cc",
        "eplb/ExpertBalancerPythonWrapper.cc",
    ],
    hdrs = [
        "eplb/ExpertBalancer.h",
        "eplb/ExpertBalancerPythonWrapper.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":dfa_utils",
        ":models",
        "//rtp_llm/cpp/pybind:py_utils",
    ],
)
