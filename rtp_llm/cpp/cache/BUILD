load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "cache_types",
    hdrs = [
        "Types.h",
        "CacheConfig.h",
    ],
    deps = [
        ":batch_kv_cache_resource",
        "//rtp_llm/cpp/core:buffer",
        "//rtp_llm/cpp/core:types",
        "//:rtp_compute_ops",
    ] + torch_deps(),
    copts = copts(),
    visibility = ["//visibility:public"],
)


cc_library(
    name = "batch_kv_cache_resource",
    hdrs = [
        "BatchKVCacheResource.h",
        "KVCacheResource.h",
    ],
    srcs = ["KVCacheResource.cc"],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cache_core",
    hdrs = glob([
        "*.h",
    ], exclude = ["Types.h", "CacheConfig.h", "KVCacheResource.h", "BatchKVCacheResource.h", "KVCacheManager.h"]),
    srcs = glob([
        "*.cc",
    ], exclude = ["KVCacheResource.cc", "KVCacheManager.cc"]),
    deps = [
        ":cache_types",
        ":batch_kv_cache_resource",
        "//rtp_llm/cpp/engine_base/stream:complete_token_ids",
        "//rtp_llm/cpp/model_rpc:model_rpc_pool",
        "//rtp_llm/cpp/model_rpc:rpc_error_code",
        "//rtp_llm/cpp/utils:lru_cache",
        "//rtp_llm/cpp/metrics:metrics",
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/config:model_config",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:hash_util",
        "//rtp_llm/cpp/utils:kv_cache_utils",
        "//rtp_llm/cpp/core:buffer",
        "//rtp_llm/cpp/core:buffer_torch",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ] + torch_deps(),
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cache",
    hdrs = ["KVCacheManager.h"],
    srcs = ["KVCacheManager.cc"],
    deps = [
        ":cache_core",
        ":cache_types",
        "//rtp_llm/cpp/cache/connector:connector",
    ] + torch_deps(),
    copts = copts(),
    visibility = ["//visibility:public"],
)
