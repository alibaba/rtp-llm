load("//:def.bzl", "copts")

cc_test(
    name = "kmonitor_worker_lifecycle_test",
    srcs = ["KmonitorWorkerLifecycleTest.cc"],
    deps = [
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "remote_cache_payload_reporter_test",
    srcs = ["RemoteCachePayloadReporterTest.cc"],
    deps = [
        ":metrics",
        "@com_google_googletest//:gtest_main",
    ],
)

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

cc_test(
    name = "rtp_llm_eplb_metrics_test",
    srcs = [
        "RtpLLMEplbMetricsTest.cc",
    ],
    deps = [
        ":metrics",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "rtp_llm_speculative_metrics_test",
    srcs = [
        "RtpLLMSpeculativeMetricsTest.cc",
    ],
    deps = [
        ":metrics",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
