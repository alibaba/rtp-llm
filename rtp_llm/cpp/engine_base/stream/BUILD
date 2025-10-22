load("//:def.bzl", "copts")

cc_library(
    name = "generate_config",
    hdrs = [
        "GenerateConfig.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//:rtp_compute_ops",
    ],
)

cc_library(
    name = "generate_types",
    hdrs = [
        "GenerateTypes.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/models:position_ids_generator",
        "//rtp_llm/cpp/multimodal_processor:multimodal_types",
    ],
)

cc_library(
    name = "stream",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob(
        [
            "*.h",
        ],
        exclude = ["GenerateTypes.h"],
    ),
    visibility = ["//visibility:public"],
    deps = [
        ":generate_config",
        ":generate_types",
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/cache2",
        "//rtp_llm/cpp/engine_base/system_prompt",
        "//rtp_llm/cpp/models",
        "//rtp_llm/cpp/models:dfa_utils",
        "//rtp_llm/cpp/utils:hash_util",
        "@havenask//aios/autil:synchronized_queue",
    ],
)
