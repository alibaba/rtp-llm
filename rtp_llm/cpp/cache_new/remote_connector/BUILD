config_setting(
    name = "kvcm_internal",
    values = {"copt": "-DKVCM_INTERNAL=1"},
)

cc_library(
    name = "remote_connector",
    srcs = [
        "ClientFactory.cc",
        "ClientWrapper.cc",
        "GroupPolicy.cc",
        "RemoteConnector.cc",
        "RemoteConnectorConfig.cc",
    ],
    hdrs = [
        "ClientFactory.h",
        "ClientWrapper.h",
        "GroupPolicy.h",
        "RemoteConnector.h",
        "RemoteConnectorConfig.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":subscriber",
        "//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_client",
        "//rtp_llm/cpp/cache_new",
        "@havenask//aios/autil:json",
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
    visibility = ["//rtp_llm/cpp/cache_new/remote_connector:__subpackages__"],
    deps = [
        "//:rtp_compute_ops",
        "@havenask//aios/autil:env_util",
        "@havenask//aios/autil:lock_free",
    ] + select({
        ":kvcm_internal": [
            "@vipserver",
        ],
        "//conditions:default": [],
    }),
)
