load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

test_deps = [
    "//rtp_llm/cpp/devices:devices_base",
    "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
    "//rtp_llm/cpp/config:gpt_init_params",
    "//rtp_llm/cpp/cache_new",
    "//rtp_llm/cpp/utils:core_utils",
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
    deps = test_deps + [
        "//rtp_llm/cpp/cache_new/remote_connector",
    ],
)

cc_test(
    name = "remote_connector_internal_test",
    srcs = [
        "RemoteConnectorInternalTest.cc",
    ],
    copts = test_copts,
    deps = test_deps + [
        "//rtp_llm/cpp/cache_new/remote_connector",
    ],
)

cc_test(
    name = "remote_connector_mock_full_linear_test",
    srcs = [
        "RemoteConnectorMockFullLinearTest.cc",
    ],
    copts = test_copts,
    deps = test_deps + [
        ":mock",
    ],
)

cc_test(
    name = "client_wrapper_test",
    srcs = [
        "ClientWrapperTest.cc",
    ],
    copts = test_copts,
    deps = test_deps + [
        ":mock",
    ],
)

cc_library(
    name = "mock",
    hdrs = [
        "MockKVCMClient.h",
    ],
    deps = [
        "//rtp_llm/cpp/cache_new/remote_connector",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
