load("//:def.bzl", "copts")

test_copts = [
    "-fno-access-control",
] + copts()

cc_test(
    name = "layer_cache_buffer_test",
    srcs = [
        "LayerCacheBufferTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache_new/p2p_connector:basic_types",
        "//rtp_llm/cpp/core:types",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    env = {},
)

