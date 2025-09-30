load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

# Sample infos - model sampling data structures
cc_library(
    name = "sample_infos",
    hdrs = [
        "SampleInfos.h",
    ],
    deps = [
        "//rtp_llm/cpp/core:buffer_hdr",
        "//rtp_llm/cpp/core:types_hdr",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/utils:core_utils",
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
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/config:config_modules",
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
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/config:config_modules",
    ],
    implementation_deps = [
        "//rtp_llm/cpp/config:static_config",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

# Logits processor - model output processing
cc_library(
    name = "logits_processor",
    hdrs = [
        "logits_processor/BaseLogitsProcessor.h",
        "logits_processor/BeamSearchLogitsProcessor.h", 
        "logits_processor/LogitsProcessorStates.h",
        "logits_processor/ThinkModeLogitsProcessor.h",
        "logits_processor/TreeLogitsProcessor.h",
    ],
    srcs = [
        "logits_processor/BaseLogitsProcessor.cc",
        "logits_processor/BeamSearchLogitsProcessor.cc",
        "logits_processor/LogitsProcessorStates.cc", 
        "logits_processor/ThinkModeLogitsProcessor.cc",
        "logits_processor/TreeLogitsProcessor.cc",
    ],
    deps = [
        ":sample_infos",
        ":dfa_utils",
        "//rtp_llm/cpp/engine_base/stream:generate_types",
        "//rtp_llm/cpp/multimodal_processor:multimodal_types",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/pybind:py_utils",
        "//rtp_llm/cpp/config:config_modules",
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
        "//rtp_llm/cpp/models/models_weight:weights_define",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/core:buffer",
        "//rtp_llm/cpp/core:types",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code", 
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
    deps = torch_deps() + [
        ":sample_infos",
        ":logits_processor",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        ":dfa_utils",
        "@havenask//aios/autil:scope",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/models_py/bindings:op_defs"
    ] + select({
        "@//:using_cuda": [
            "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
        ],
        "@//:using_rocm": [
            "//rtp_llm/cpp/devices/rocm_impl:rocm_impl",
        ],
        "@//:using_arm": [
            "//rtp_llm/cpp/devices/arm_impl:arm_cpu_impl",
        ],
        "//conditions:default": [
            "//rtp_llm/cpp/devices/cpu_impl:cpu_impl",
        ],
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
    deps = torch_deps() + [
        ":models",
        ":stats",
        ":eplb_config",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        ":dfa_utils",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/pybind:py_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    visibility = ["//visibility:public"],
)
