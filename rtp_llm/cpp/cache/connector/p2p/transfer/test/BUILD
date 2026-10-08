load("//:def.bzl", "copts")

test_copts = [
    "-fno-access-control",
] + copts()

cc_test(
    name = "transfer_task_store_test",
    srcs = [
        "TransferTaskStoreTest.cc",
        "main.cc",
    ],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/utils:time_util",
        "@com_google_googletest//:gtest",
        "@havenask//aios/autil:log",
    ],
)

cc_test(
    name = "transfer_task_test",
    srcs = [
        "TransferTaskTest.cc",
        "main.cc",
    ],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/cpp/metrics:metrics",
        "//rtp_llm/cpp/utils:time_util",
        "@com_google_googletest//:gtest",
        "@havenask//aios/autil:log",
    ],
)
