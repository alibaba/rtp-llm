load("//:def.bzl", "copts")
load("//bazel:device_defs.bzl", "device_impl_target")

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

# Retain the complete original Native A10 declaration; main same-name rule remains unchanged.
cc_test(
    name = "transfer_task_store_test_native_a10",
    srcs = [
        "TransferTaskStoreTest.cc",
        "main.cc",
    ],
    copts = test_copts,
    exec_properties = {'gpu':'A10'},
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/cpp/utils:time_util",
        "@com_google_googletest//:gtest",
    ] + device_impl_target(),
    tags = ["A10"],
)

# Retain the complete original Native A10 declaration; main same-name rule remains unchanged.
cc_test(
    name = "transfer_task_test_native_a10",
    srcs = [
        "TransferTaskTest.cc",
        "main.cc",
    ],
    copts = test_copts,
    exec_properties = {'gpu':'A10'},
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/cpp/utils:time_util",
        "@com_google_googletest//:gtest",
    ] + device_impl_target(),
    tags = ["A10"],
)
