load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "generate_config",
    hdrs = [
        "GenerateConfig.h",
    ],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "@havenask//aios/autil:json",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "generate_types",
    hdrs = [
        "GenerateTypes.h",
    ],
    deps = torch_deps() + [
        ":generate_config",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/models:position_ids_generator",
        "//rtp_llm/cpp/core:buffer_hdr",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/multimodal_processor:multimodal_types",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "stream",
    hdrs = glob([
        "*.h",
    ], exclude = ["GenerateTypes.h"]),
    srcs = glob([
        "*.cc",
    ]),
    deps = torch_deps() + [
        ":generate_types",
        ":generate_config",
        "//rtp_llm/cpp/engine_base/system_prompt:system_prompt",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:hash_util",
        "//rtp_llm/cpp/models:dfa_utils",
        "@havenask//aios/autil:synchronized_queue",
    ],
    visibility = ["//visibility:public"],
)