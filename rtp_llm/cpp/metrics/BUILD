load("//:def.bzl", "copts")

cc_library(
    name = "metrics",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob(
        [
            "*.cc",
        ],
        exclude = [
            "*Test.cc",
        ],
    ),
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_test(
    name = "rtp_llm_token_ps_metrics_collector_test",
    srcs = [
        "RtpLLMTokenPSMetricsCollectorTest.cc",
    ],
    deps = [
        ":metrics",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

# R6 DEV-zeta — canary G5 gauge counter coverage.
cc_test(
    name = "kv_cache_canary_metrics_test",
    srcs = [
        "KVCacheCanaryMetricsTest.cc",
    ],
    deps = [
        ":metrics",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
