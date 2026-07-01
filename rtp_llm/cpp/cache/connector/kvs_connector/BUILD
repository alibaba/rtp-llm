load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "kvs_connector",
    srcs = [
        "KVSClient.cc",
        "KVSConnector.cc",
    ],
    hdrs = [
        "KVSClient.h",
        "KVSConnector.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "//rtp_llm/cpp/cache:block_info",
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/cache:kv_cache_allocator",
        "//rtp_llm/cpp/cache/connector:connector_base",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/utils:core_utils",
        "@curl//:curl",
        "@havenask//aios/autil:env_util",
        "@havenask//aios/autil:thread",
        "@rapidjson//:rapidjson",
    ] + torch_deps() + select({
        "@//:using_cuda": [
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
        "//conditions:default": [],
    }),
)
