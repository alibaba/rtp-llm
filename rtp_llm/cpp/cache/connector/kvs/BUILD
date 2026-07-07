load("//:def.bzl", "copts")

cc_library(
    name = "kvs_object_types",
    hdrs = ["KVSObjectTypes.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kvs_object_backend",
    hdrs = [
        "KVSObjectBackend.h",
        "KVSConnectorConfig.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [":kvs_object_types"],
)

cc_library(
    name = "kvs_object_store",
    srcs = ["KVSObjectStore.cc"],
    hdrs = [
        "KVSObjectStore.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":kvs_object_backend",
    ],
)

cc_library(
    name = "kvs_connector",
    srcs = ["KVSConnector.cc"],
    hdrs = [
        "KVSConnector.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":kvs_object_store",
        "//rtp_llm/cpp/cache:block_info",
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/cache/connector:connector_base",
        "//rtp_llm/cpp/model_rpc:broadcast_manager",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/utils:core_utils",
        "@havenask//aios/autil:lock_free",
    ],
)

cc_library(
    name = "kvs_client_backend",
    srcs = [
        "KVSClientBackend.cc"
    ],
    hdrs = [
        "KVSClientBackend.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":kvs_object_backend",
        "//rtp_llm/cpp/utils:core_utils",
        "@v6d_kvs_sdk//:kvs_client",
    ] + select({
        "@//:using_cuda": [
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
        "//conditions:default": [],
    }),
)
