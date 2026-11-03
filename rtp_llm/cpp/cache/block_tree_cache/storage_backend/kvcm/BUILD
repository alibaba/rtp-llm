load("//:def.bzl", "copts")

cc_library(
    name = "decode_publication_accounting",
    hdrs = ["DecodePublicationAccounting.h"],
    copts = copts(),
    deps = ["//rtp_llm/cpp/cache/block_tree_cache:storage_completion_admission"],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "KVCM_INTERNAL",
    values = {"copt": "-DKVCM_INTERNAL=1"},
)

cc_library(
    name = "sdk_buffer_safety",
    hdrs = ["SdkBufferSafety.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "dtop_origin_metadata",
    hdrs = ["DtopOriginMetadata.h"],
    deps = [
        "//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_client_headers",
        "//rtp_llm/cpp/utils:decode_cache_origin",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "RECO_INTERNAL",
    values = {"copt": "-DRECO_INTERNAL=1"},
)

cc_library(
    name = "kvcm_storage_backend",
    srcs = ["KVCMStorageBackend.cc"],
    hdrs = ["KVCMStorageBackend.h", "DtopCompletionBridge.inc", "WorkerPayloadProof.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":client",
        ":decode_publication_accounting",
        ":dtop_origin_metadata",
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "//rtp_llm/cpp/cache:cp_slot_mapper",
        ":group_policy",
        "//rtp_llm/cpp/cache/block_tree_cache",
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/model_rpc:broadcast_manager",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:decode_cache_origin",
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
    srcs = ["KVCMConfig.cc"],
    hdrs = ["KVCMConfig.h"],
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
        "PackedSsdTransferClient.h",
        "SdkTransferObserver.h",
        "ClientWrapper.h",
        "KVCMConfig.h",
    ],
    copts = copts(),
    visibility = ["//rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm:__subpackages__"],
    deps = [
        ":config",
        ":subscriber",
        ":sdk_buffer_safety",
        "@havenask//aios/autil:env_util",
        "@havenask//aios/autil:json",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_client",
        "//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_client_headers",
        "//rtp_llm/cpp/cache:storage_3fs",
        "//rtp_llm/cpp/utils:core_utils",
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
    visibility = [
        "//rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm:__subpackages__",
        "//rtp_llm/cpp/cache/events:__pkg__",
    ],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "@havenask//aios/autil:env_util",
        "@havenask//aios/autil:lock_free",
    ] + select({
        ":KVCM_INTERNAL": ["@vipserver"],
        ":RECO_INTERNAL": ["@vipserver"],
        "//conditions:default": [],
    }),
)

exports_files(
    ["VIPServerSubscriber.cc", "VIPServerSubscriber.h", "Subscriber.h"],
    visibility = ["//rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test:__pkg__"],
)
