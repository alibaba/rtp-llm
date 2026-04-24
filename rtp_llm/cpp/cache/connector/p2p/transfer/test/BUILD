load("//:def.bzl", "copts")

load("//bazel:device_defs.bzl", "device_impl_target", "device_test_envs")

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
    exec_properties = {'gpu':'H20'},
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/cpp/utils:time_util",
        "@com_google_googletest//:gtest",
    ] + device_impl_target(),
)

cc_test(
    name = "transfer_task_test",
    srcs = [
        "TransferTaskTest.cc",
        "main.cc",
    ],
    copts = test_copts,
    exec_properties = {'gpu':'H20'},
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/cpp/utils:time_util",
        "@com_google_googletest//:gtest",
    ] + device_impl_target(),
)
