load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test = cc_test_wrapper

cc_test(
    name = "kv_cache_event_publisher_test",
    srcs = [
        "KVCacheEventPublisherTest.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache/events:kv_cache_event_publisher",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
