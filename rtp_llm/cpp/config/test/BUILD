load("//:def.bzl", "copts")

test_copts = [
    "-fno-access-control",
] + copts()

google_test_deps = [
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
]

cc_test(
    name = "cache_store_config_test",
    srcs = ["CacheStoreConfigTest.cc"],
    copts = test_copts,
    deps = google_test_deps + [
        "//rtp_llm/cpp/cache/connector/p2p:p2p_connector_config",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/config:rdma_device_health_config",
    ],
)
