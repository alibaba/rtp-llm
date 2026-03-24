load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "connector",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    copts = copts() + select({
        "//conditions:default": [],
        "//:using_remote_kv_cache": ["-DUSE_REMOTE_KV_CACHE=1"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/cache:cache_core",
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/cache/connector/memory:memory_connector",
        "//rtp_llm/cpp/metrics",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "@havenask//aios/autil:json",
        "@havenask//aios/autil:thread",
    ] + torch_deps() + select({
        "//:using_remote_kv_cache": [
            "//rtp_llm/cpp/cache/connector/remote_connector",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "connector_base",
    hdrs = [
        "AsyncContext.h",
        "KVCacheConnector.h",
        "Meta.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
    ],
)
