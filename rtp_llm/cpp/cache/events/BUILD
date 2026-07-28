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
        "KVCacheEventPublisherFactory.cc",
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
        ":kv_cache_event_queue",
        "//rtp_llm/cpp/utils:core_utils",
        "@curl//:curl",
        "@rapidjson//:rapidjson",
    ],
)
