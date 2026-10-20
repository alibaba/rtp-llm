load("//:def.bzl", "copts")
load("//bazel:tf_proto.bzl", "cc_proto_library")

cc_proto_library(
    name = "kvcm_discovery_proto",
    srcs = ["kvcm_discovery.proto"],
    cc_libs = ["@com_google_protobuf//:protobuf"],
)

cc_library(
    name = "kv_cache_event",
    hdrs = [
        "KVCacheEvent.h",
        "KVCacheEventPublisher.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kv_cache_event_queue",
    srcs = [
        "KVCacheEventQueue.cc",
    ],
    hdrs = [
        "KVCacheEventQueue.h",
    ],
    copts = copts(),
    visibility = [
        "//rtp_llm/cpp/cache/events:__pkg__",
        "//rtp_llm/cpp/cache/events/test:__pkg__",
    ],
    deps = [
        ":kv_cache_event",
    ],
)

cc_library(
    name = "kv_cache_event_publisher",
    srcs = [
        "KVCMPublisher.cc",
    ],
    hdrs = [
        "KVCMPublisher.h",
        "KVCacheEventPublisherConfig.h",
        "KVCacheEventReporter.h",
        "KVCMPublisherUtils.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_event",
        ":kv_cache_event_queue",
        ":kvcm_discovery_proto",
        "//rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm:subscriber",
        "//rtp_llm/cpp/utils:core_utils",
        "@curl//:curl",
        "@grpc//:grpc++",
        "@rapidjson//:rapidjson",
    ],
)
