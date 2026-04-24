load("//:def.bzl", "copts")

load("//bazel:arch_select.bzl", "torch_deps")
load("//bazel:device_defs.bzl", "device_impl_target", "device_test_envs")

test_copts = [
    "-fno-access-control",
] + copts()

cc_test(
    name = "tcp_task_context_test",
    srcs = [
        "TcpTaskContextTest.cc",
        "main.cc",
    ],
    copts = test_copts,
    exec_properties = {'gpu':'H20'},
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p/transfer/tcp:tcp_task_context",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/utils:time_util",
        "//rtp_llm/cpp/metrics:metrics",
        "@havenask//aios/autil:log",
        "@com_google_googletest//:gtest",
    ] + device_impl_target() + torch_deps(),
)

cc_test(
    name = "tcp_transfer_service_test",
    srcs = [
        "TcpTransferServiceTest.cc",
        "main.cc",
    ],
    copts = test_copts,
    exec_properties = {'gpu':'H20'},
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p/transfer/tcp:tcp_transfer_service",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/utils:time_util",
        "//rtp_llm/cpp/metrics:metrics",
        "@havenask//aios/autil:log",
        "@com_google_googletest//:gtest",
    ] + device_impl_target() + torch_deps(),
)

cc_test(
    name = "tcp_sender_receiver_test",
    srcs = [
        "TcpSenderReceiverTest.cc",
        "main.cc",
    ],
    copts = test_copts,
    exec_properties = {'gpu':'H20'},
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p/transfer/tcp:tcp_backend",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/utils:time_util",
        "//rtp_llm/cpp/metrics:metrics",
        "@havenask//aios/autil:log",
        "@com_google_googletest//:gtest",
    ] + device_impl_target() + torch_deps(),
)
