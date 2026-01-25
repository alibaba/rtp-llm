load("//:def.bzl", "copts")

load("//rtp_llm/cpp/devices:device_defs.bzl", "device_impl_target", "device_test_envs")

test_copts = [
    "-fno-access-control",
] + copts()

cc_test(
    name = "layer_cache_buffer_store_test",
    srcs = [
        "LayerCacheBufferStoreTest.cc",
        "main.cc",
    ],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/cpp/utils:time_util",
        "@com_google_googletest//:gtest",
    ] + device_impl_target(),
)

cc_library(
    name = "transfer_server_client_test",
    srcs = [
        "TransferServerClientTest.cc",
    ],
    hdrs = [
        "TransferServerClientTest.h",
    ],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p/transfer:transfer_server",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:transfer_client",
        "//rtp_llm/cpp/devices/testing:device_test_utils",
        "@com_google_googletest//:gtest",
        "@havenask//aios/autil:net",
    ],
    visibility = ['//visibility:public'],
)

cc_test(
    name = "transfer_server_client_tcp_test",
    srcs = [
        "TransferServerClientTcpTest.cc",
        "main.cc",
    ],
    copts = test_copts,
    deps = [
        ":transfer_server_client_test",
       "@havenask//aios/autil:net",
    ] + device_impl_target(),
    env = device_test_envs(),
)


cc_test(
    name = "transfer_task_store_test",
    srcs = [
        "TransferTaskStoreTest.cc",
        "main.cc",
    ],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
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
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/cpp/utils:time_util",
        "@com_google_googletest//:gtest",
    ] + device_impl_target(),
)

