load("//:def.bzl", "copts")

test_copts = [
    "-fno-access-control",
] + copts()

cc_library(
    name = "mock_kv_cache_allocator",
    srcs = [
    ],
    hdrs = [
        "MockKVCacheAllocator.h",
    ],
    deps = [
        "//rtp_llm/cpp/cache_new:cache_new",
        "@com_google_googletest//:gtest",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "layer_cache_buffer_test",
    srcs = [
        "LayerCacheBufferTest.cc",
        "main.cc",
    ],
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache_new/p2p_connector:basic_types",
        "//rtp_llm/cpp/core:types",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "//rtp_llm/cpp/metrics:metrics",
    ],
    env = {},
)

cc_test(
    name = "cache_store_test",
    srcs = [
        "CacheStoreTest.cc",
        "main.cc",
    ],
    data = [],
    copts = test_copts,
    deps = [
        ":mock_kv_cache_allocator",
        "//rtp_llm/cpp/cache_new/p2p_connector:client",
        "//rtp_llm/cpp/cache_new/p2p_connector:server",
        "//rtp_llm/cpp/cache_new/p2p_connector:basic_types",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    env = {},
)
