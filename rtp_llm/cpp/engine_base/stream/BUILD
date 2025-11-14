load("//:def.bzl", "copts")

cc_library(
    name = "generate_config",
    hdrs = [
        "GenerateConfig.h",
    ],
    deps = [
        "//rtp_llm/cpp/config:role_types",
        "//rtp_llm/cpp/config:special_tokens",
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "generate_types",
    hdrs = [
        "GenerateTypes.h",
    ],
    deps = [
        "//rtp_llm/cpp/models:position_ids_generator",
        "//rtp_llm/cpp/multimodal_processor:multimodal_types",
        "//:rtp_compute_ops",
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
    deps = [
        ":generate_types",
        ":generate_config",
        "//rtp_llm/cpp/engine_base/system_prompt:system_prompt",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/cache",
        "//rtp_llm/cpp/utils:hash_util",
        "//rtp_llm/cpp/models:dfa_utils",
        "@havenask//aios/autil:synchronized_queue",
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
)