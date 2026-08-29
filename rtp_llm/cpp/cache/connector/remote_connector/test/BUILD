load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test_wrapper(
    name = "client_wrapper_lifecycle_test",
    srcs = [
        "ClientWrapperLifecycleTest.cc",
        "MockKVCMClient.h",
    ],
    copts = copts(),
    target_compatible_with = select({
        "//:using_remote_kv_cache": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//rtp_llm/cpp/cache/connector/remote_connector:client",
        "//rtp_llm/cpp/cache/connector/remote_connector:subscriber",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_wrapper(
    name = "group_policy_test",
    srcs = ["GroupPolicyTest.cc"],
    copts = copts(),
    target_compatible_with = select({
        "//:using_remote_kv_cache": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//rtp_llm/cpp/cache:kv_cache_specs",
        "//rtp_llm/cpp/cache/connector/remote_connector:group_policy",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_wrapper(
    name = "remote_connector_config_test",
    srcs = ["RemoteConnectorConfigTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache/connector/remote_connector:config",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@havenask//aios/autil:json",
    ],
)
