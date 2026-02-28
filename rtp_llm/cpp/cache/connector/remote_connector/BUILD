load("//:def.bzl", "copts")

config_setting(
    name = "RECO_INTERNAL",
    values = {"copt": "-DRECO_INTERNAL=1"},
)

cc_library(
    name = "remote_connector",
    srcs = [
        "GroupPolicy.cc",
        "RemoteConnector.cc",
        "RemoteConnectorConfig.cc",
    ],
    hdrs = [
        "GroupPolicy.h",
        "RemoteConnector.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":client",
        ":config",
        ":subscriber",
        "//rtp_llm/cpp/cache:kv_cache_allocator",
        "//rtp_llm/cpp/cache/connector:connector_base",
        "//rtp_llm/cpp/model_rpc:broadcast_manager",
    ],
)

cc_library(
    name = "config",
    srcs = [
        "RemoteConnectorConfig.cc",
    ],
    hdrs = [
        "RemoteConnectorConfig.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@havenask//aios/autil:json",
    ],
)

cc_library(
    name = "client",
    srcs = [
        "ClientFactory.cc",
        "ClientWrapper.cc",
    ],
    hdrs = [
        "ClientFactory.h",
        "ClientWrapper.h",
        "RemoteConnectorConfig.h",
    ],
    copts = copts(),
    visibility = ["//rtp_llm/cpp/cache/connector/remote_connector:__subpackages__"],
    deps = [
        ":config",
        ":subscriber",
        "//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_client",
        "//rtp_llm/cpp/cache:storage_3fs",
        "@grpc//:grpc++",
    ],
)

cc_library(
    name = "subscriber",
    srcs = [
        "VIPServerSubscriber.cc",
    ],
    hdrs = [
        "DirectSubscriber.h",
        "Subscriber.h",
        "VIPServerSubscriber.h",
    ],
    copts = copts(),
    visibility = ["//rtp_llm/cpp/cache/connector/remote_connector:__subpackages__"],
    deps = [
        "//:rtp_compute_ops",
        "@havenask//aios/autil:env_util",
        "@havenask//aios/autil:lock_free",
    ] + select({
        ":RECO_INTERNAL": [
            "@vipserver",
        ],
        "//conditions:default": [],
    }),
)
