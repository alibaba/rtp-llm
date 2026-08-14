load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test = cc_test_wrapper

cc_test(
    name = "kv_cache_event_queue_test",
    size = "small",
    srcs = [
        "KVCacheEventQueueTest.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache/events:kv_cache_event_admission_gate",
        "//rtp_llm/cpp/cache/events:kv_cache_event_publisher_config",
        "//rtp_llm/cpp/cache/events:kv_cache_event_queue",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "kv_cache_event_metrics_test",
    size = "small",
    srcs = [
        "KVCacheEventMetricsTest.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache/events:kv_cache_event_metrics",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "kv_cache_event_publisher_test",
    size = "medium",
    srcs = [
        "KVCacheEventPublisherTest.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/config/test:kv_cache_event_validation_cases",
        "//rtp_llm/cpp/cache/events:kv_cache_event_publisher",
        "//rtp_llm/cpp/cache/events:kvcm_publisher_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "kvcm_request_builder_test",
    size = "small",
    srcs = [
        "KVCMRequestBuilderTest.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache/events:kvcm_request_builder",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "kvcm_logical_mirror_test",
    size = "small",
    srcs = [
        "KVCMLogicalMirrorTest.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache/events:kvcm_logical_mirror",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "kv_cache_event_publisher_assembly_test",
    size = "small",
    srcs = [
        "KVCacheEventPublisherAssemblyTest.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache/events:kv_cache_event_publisher_assembly",
        "//rtp_llm/cpp/config:config_modules",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
