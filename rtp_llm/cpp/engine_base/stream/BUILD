load("//:def.bzl", "copts")

cc_library(
    name = "generate_config",
    hdrs = [
        "GenerateConfig.h",
    ],
    deps = [
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
    name = "complete_token_ids",
    hdrs = ["CompleteTokenIds.h"],
    srcs = ["CompleteTokenIds.cc"],
    deps = [
        ":generate_types",
        ":generate_config",
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
    ], exclude = ["CompleteTokenIds.cc"]),
    deps = [
        ":generate_types",
        ":generate_config",
        ":complete_token_ids",
        "//rtp_llm/cpp/engine_base/system_prompt:system_prompt",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/cache_new:cache_new",
        "//rtp_llm/cpp/utils:hash_util",
        "//rtp_llm/cpp/models:dfa_utils",
        "@havenask//aios/autil:synchronized_queue",
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "stream_cache_resource_headers",
    hdrs = [
        "ResourceContext.h",
        "StreamCacheResource.h",
    ],
    deps = [
        ":complete_token_ids",
        "//rtp_llm/cpp/cache_new:batch_kv_cache_resource",
        "//rtp_llm/cpp/cache_new:cache_new_types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    visibility = ["//visibility:public"],
)