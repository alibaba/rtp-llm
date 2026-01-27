load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "memory_connector",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        "//rtp_llm/cpp/cache:cache_core",
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/cache/connector:connector_header",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/model_rpc:broadcast_manager",
        "//rtp_llm/cpp/utils:lru_cache",
        "@havenask//aios/autil:json",
        "@havenask//aios/autil:lock_free",
    ] + torch_deps(),
    copts = copts(),
    visibility = ["//visibility:public"],
)
