# RDMA hardware tests moved from the internal tree. They link the closed
# accl-barex library and need real RDMA NICs, so srcs and impl deps appear
# only when //:using_rdma is ON (internal overlay builds). All targets are
# manual + pool-tagged: public wildcard lanes never schedule them; run them
# explicitly on an internal RDMA worker.

load("@arch_config//:arch_select.bzl", "torch_deps")
load("//:def.bzl", "copts")
load("//bazel:device_defs.bzl", "device_impl_target")

test_copts = [
    "-fno-access-control",
] + copts()

cc_test(
    name = "rdma_memory_manager_test",
    srcs = select({
        "//:using_rdma": [
            "RdmaMemoryManagerTest.cc",
            "main.cc",
        ],
        "//conditions:default": [],
    }),
    copts = test_copts,
    env = {
        "DEVICE_RESERVE_MEMORY_BYTES": "128000000",
    },
    exec_properties = {"gpu": "L20_CU13"},
    tags = [
        "L20_CU13",
        "manual",
    ],
    deps = [
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + device_impl_target() + torch_deps() + select({
        "//:using_rdma": [
            "//rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma:rdma_interface",
            "//rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma:transfer_rdma_impl",
            "//rtp_llm/cpp/testing:device_test_utils",
            "@havenask//aios/autil:log",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "rdma_client_server_test",
    srcs = select({
        "//:using_rdma": [
            "RdmaClientServerTest.cc",
            "main.cc",
        ],
        "//conditions:default": [],
    }),
    copts = test_copts,
    env = {
        "DEVICE_RESERVE_MEMORY_BYTES": "128000000",
        "FIC2_OOO_DISABLE_0115": "1",
    },
    exec_properties = {"gpu": "L20_CU13"},
    tags = [
        "L20_CU13",
        "manual",
    ],
    deps = [
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + device_impl_target() + torch_deps() + select({
        "//:using_rdma": [
            "//rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma:rdma_interface",
            "//rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma:transfer_rdma_impl",
            "//rtp_llm/cpp/testing:device_test_utils",
            "@havenask//aios/autil:log",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "rdma_transfer_service_test",
    srcs = select({
        "//:using_rdma": [
            "RdmaTransferServiceTest.cc",
            "main.cc",
        ],
        "//conditions:default": [],
    }),
    copts = test_copts,
    env = {},
    tags = [
        "manual",
    ],
    deps = [
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + select({
        "//:using_rdma": [
            "//rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma:rdma_interface",
            "//rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma:rdma_transfer_service",
            "//rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma:transfer_rdma_impl",
            "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
            "//rtp_llm/cpp/config:config_modules",
            "//rtp_llm/cpp/utils:time_util",
            "@havenask//aios/autil:log",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "rdma_sender_receiver_test",
    srcs = select({
        "//:using_rdma": [
            "RdmaSenderReceiverTest.cc",
            "main.cc",
        ],
        "//conditions:default": [],
    }),
    copts = test_copts,
    env = {
        "DEVICE_RESERVE_MEMORY_BYTES": "512000000",
        "FIC2_OOO_DISABLE_0115": "1",
    },
    exec_properties = {"gpu": "L20_CU13"},
    tags = [
        "L20_CU13",
        "manual",
    ],
    deps = [
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + device_impl_target() + torch_deps() + select({
        "//:using_rdma": [
            "//rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma:rdma_backend",
            "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
            "//rtp_llm/cpp/config:config_modules",
            "//rtp_llm/cpp/metrics",
            "//rtp_llm/cpp/utils:time_util",
            "//rtp_llm/models_py/bindings/core:exec_ctx_state",
            "@havenask//aios/autil:log",
        ],
        "//conditions:default": [],
    }),
)
