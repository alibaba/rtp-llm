load("//:def.bzl", "copts")

config_setting(
    name = "RECO_INTERNAL",
    values = {"copt": "-DRECO_INTERNAL=1"},
)

cc_library(
    name = "kvc_m_storage_backend",
    srcs = ["KVCMStorageBackend.cc"],
    hdrs = ["KVCMStorageBackend.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":client",
        ":group_policy",
        "//rtp_llm/cpp/cache/block_tree_cache",
        "//rtp_llm/cpp/model_rpc:broadcast_manager",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings/core:types",
        "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        "@havenask//aios/autil:env_util",
        "@havenask//aios/autil:json",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
)

cc_library(
    name = "group_policy",
    srcs = ["GroupPolicy.cc"],
    hdrs = ["GroupPolicy.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_client_headers",
        "//rtp_llm/cpp/cache/block_tree_cache",
        "//rtp_llm/cpp/utils:core_utils",
    ],
)

cc_library(
    name = "config",
    srcs = ["RemoteConnectorConfig.cc"],
    hdrs = ["RemoteConnectorConfig.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = ["@havenask//aios/autil:json"],
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
    srcs = ["VIPServerSubscriber.cc"],
    hdrs = [
        "DirectSubscriber.h",
        "Subscriber.h",
        "VIPServerSubscriber.h",
    ],
    copts = copts(),
    visibility = ["//rtp_llm/cpp/cache/connector/remote_connector:__subpackages__"],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "@havenask//aios/autil:env_util",
        "@havenask//aios/autil:lock_free",
    ] + select({
        ":RECO_INTERNAL": ["@vipserver"],
        "//conditions:default": [],
    }),
)
