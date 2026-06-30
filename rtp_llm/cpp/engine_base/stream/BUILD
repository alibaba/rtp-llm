load("//:def.bzl", "copts")

cc_library(
    name = "generate_config",
    hdrs = [
        "GenerateConfig.h",
    ],
    deps = [
        "//rtp_llm/cpp/config:role_types",
        "//rtp_llm/cpp/config:special_tokens",
        "@havenask//aios/autil:json",
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
        ":generate_config",
        "//rtp_llm/cpp/models:position_ids_generator",
        "//rtp_llm/cpp/multimodal_processor:multimodal_types",
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "input_embeddings_utils",
    srcs = [
        "InputEmbeddingsUtils.cc",
    ],
    hdrs = [
        "InputEmbeddingsUtils.h",
    ],
    deps = [
        ":generate_types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "complete_token_ids",
    hdrs = ["CompleteTokenIds.h"],
    srcs = ["CompleteTokenIds.cc"],
    deps = [
        ":generate_types",
        ":generate_config",
        "//rtp_llm/cpp/utils:error_code",
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "stream",
    hdrs = glob([
        "*.h",
    ], exclude = ["GenerateTypes.h", "CompleteTokenIds.h"]),
    srcs = glob([
        "*.cc",
    ], exclude = ["CompleteTokenIds.cc", "InputEmbeddingsUtils.cc"]),
    deps = [
        ":generate_types",
        ":generate_config",
        ":complete_token_ids",
        "//rtp_llm/cpp/engine_base/system_prompt:system_prompt",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/utils:hash_util",
        "//rtp_llm/cpp/models:dfa_utils",
        "@havenask//aios/autil:synchronized_queue",
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/utils:linear_blocks_util",
        "//rtp_llm/cpp/model_rpc:tensor_pb_convert",
    ],
    visibility = ["//visibility:public"],
)
