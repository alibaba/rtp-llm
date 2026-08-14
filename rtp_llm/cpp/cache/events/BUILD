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
    name = "kv_cache_event_publisher_config",
    hdrs = [
        "KVCacheEventPublisherConfig.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kv_cache_event_reporter",
    hdrs = [
        "KVCacheEventReporter.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kvcm_publisher_utils",
    srcs = [
        "KVCMPublisherUtils.cc",
    ],
    hdrs = [
        "KVCMPublisherUtils.h",
    ],
    copts = copts(),
    visibility = [
        "//rtp_llm/cpp/cache/events:__pkg__",
        "//rtp_llm/cpp/cache/events/test:__pkg__",
    ],
    deps = [
        ":kv_cache_event_publisher_config",
        "@rapidjson",
    ],
)

cc_library(
    name = "kvcm_request_builder",
    srcs = [
        "KVCMRequestBuilder.cc",
    ],
    hdrs = [
        "KVCMRequestBuilder.h",
    ],
    copts = copts(),
    visibility = [
        "//rtp_llm/cpp/cache/events:__pkg__",
        "//rtp_llm/cpp/cache/events/test:__pkg__",
    ],
    deps = [
        ":kv_cache_event",
        ":kv_cache_event_publisher_config",
        "@rapidjson",
    ],
)

cc_library(
    name = "kvcm_logical_mirror",
    srcs = [
        "KVCMLogicalMirror.cc",
    ],
    hdrs = [
        "KVCMLogicalMirror.h",
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
    name = "curl_kv_cache_event_reporter",
    srcs = [
        "CurlKVCacheEventReporter.cc",
    ],
    hdrs = [
        "CurlKVCacheEventReporter.h",
    ],
    copts = copts(),
    visibility = ["//rtp_llm/cpp/cache/events:__pkg__"],
    deps = [
        ":kv_cache_event_publisher_config",
        ":kv_cache_event_reporter",
        "//rtp_llm/cpp/utils:core_utils",
        "@curl",
    ],
)

cc_library(
    name = "kv_cache_event_publisher_assembly",
    hdrs = [
        "KVCacheEventPublisherAssembly.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_event",
        ":kv_cache_event_publisher_config",
        "//rtp_llm/cpp/config:config_modules",
    ],
)

cc_library(
    name = "kv_cache_event_metrics",
    srcs = [
        "KVCacheEventMetrics.cc",
    ],
    hdrs = [
        "KVCacheEventMetrics.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_event",
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
    ],
)

cc_library(
    name = "kv_cache_event_admission_gate",
    hdrs = [
        "KVCacheEventAdmissionGate.h",
    ],
    copts = copts(),
    visibility = [
        "//rtp_llm/cpp/cache/events:__pkg__",
        "//rtp_llm/cpp/cache/events/test:__pkg__",
    ],
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
    linkopts = ["-pthread"],
    visibility = [
        "//rtp_llm/cpp/cache/events:__pkg__",
        "//rtp_llm/cpp/cache/events/test:__pkg__",
    ],
    deps = [
        ":kv_cache_event",
        ":kv_cache_event_admission_gate",
        ":kv_cache_event_publisher_config",
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
        "KVCacheEventPublisherFactory.h",
        "LogPublisher.h",
        "NullPublisher.h",
    ],
    copts = copts(),
    implementation_deps = [
        ":curl_kv_cache_event_reporter",
        ":kv_cache_event_admission_gate",
        ":kv_cache_event_queue",
        ":kvcm_logical_mirror",
        ":kvcm_publisher_utils",
        ":kvcm_request_builder",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_event",
        ":kv_cache_event_publisher_config",
        ":kv_cache_event_reporter",
    ],
)
