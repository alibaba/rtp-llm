load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

test_deps = [
    "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
    "//rtp_llm/cpp/config:config_modules",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
] + torch_deps()

test_copts = [
    "-fno-access-control",
] + copts()

cc_test(
    name = "group_policy_test",
    srcs = [
        "GroupPolicyTest.cc",
    ],
    copts = test_copts,
    exec_properties = {"gpu": "A10"},
    # tags = ["H20"],
    deps = test_deps + [
        "//rtp_llm/cpp/cache/connector/remote_connector",
    ],
)

cc_test(
    name = "remote_connector_internal_test",
    srcs = [
        "RemoteConnectorInternalTest.cc",
    ],
    copts = test_copts,
    exec_properties = {"gpu": "A10"},
    # tags = ["H20"],
    deps = test_deps + [
        "//rtp_llm/cpp/cache/connector/remote_connector",
    ],
)

cc_test(
    name = "remote_connector_mock_full_linear_test",
    srcs = [
        "RemoteConnectorMockFullLinearTest.cc",
    ],
    copts = test_copts,
    exec_properties = {"gpu": "A10"},
    # tags = ["H20"],
    deps = [
        ":remote_connector_mock_base",
    ],
)

cc_test(
    name = "remote_connector_mock_only_full_test",
    srcs = [
        "RemoteConnectorMockOnlyFullTest.cc",
    ],
    copts = test_copts,
    exec_properties = {"gpu": "A10"},
    # tags = ["H20"],
    deps = [
        ":remote_connector_mock_base",
    ],
)

cc_test(
    name = "client_wrapper_test",
    srcs = [
        "ClientWrapperTest.cc",
    ],
    copts = test_copts,
    exec_properties = {"gpu": "A10"},
    # tags = ["H20"],
    deps = test_deps + [
        ":mock",
    ],
)

cc_library(
    name = "remote_connector_mock_base",
    hdrs = [
        "RemoteConnectorMockTestBase.h",
    ],
    deps = test_deps + [
        ":mock",
        "//rtp_llm/cpp/cache/connector/remote_connector",
    ],
)

cc_library(
    name = "mock",
    hdrs = [
        "MockKVCMClient.h",
    ],
    deps = [
        "//rtp_llm/cpp/cache/connector/remote_connector:client",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
