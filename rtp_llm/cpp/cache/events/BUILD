load("//:def.bzl", "copts")

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
    name = "kv_cache_event_publisher",
    srcs = [
        "KVCMPublisher.cc",
        "KVCacheEventPublisherFactory.cc",
        "KVCacheEventQueue.cc",
        "KVCacheEventQueue.h",
        "LogPublisher.cc",
        "NullPublisher.cc",
    ],
    hdrs = [
        "KVCMPublisher.h",
        "KVCacheEventPublisherConfig.h",
        "KVCacheEventPublisherFactory.h",
        "KVCacheEventReporter.h",
        "LogPublisher.h",
        "NullPublisher.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_event",
        "//rtp_llm/cpp/utils:core_utils",
        "@curl//:curl",
        "@rapidjson//:rapidjson",
    ],
)
