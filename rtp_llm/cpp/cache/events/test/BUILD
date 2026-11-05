load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test = cc_test_wrapper

cc_test(
    name = "kvcm_event_report_test",
    srcs = [
        "KVCMEventReportTest.cc",
        "KVCacheEventStateTest.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache/events:kv_cache_event",
        "//rtp_llm/cpp/cache/events:kvcm_event_report",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@rapidjson//:rapidjson",
    ],
)

cc_test(
    name = "kv_cache_event_queue_test",
    srcs = [
        "KVCacheEventQueueTest.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache/events:kv_cache_event_queue",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "kv_cache_event_publisher_test",
    size = "small",
    srcs = [
        "KVCacheEventPublisherTest.cc",
    ],
    copts = copts(),
    timeout = "short",
    deps = [
        "//rtp_llm/cpp/cache/events:kv_cache_event_publisher",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@rapidjson//:rapidjson",
    ],
)
