load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")
load("//bazel:device_defs.bzl", "device_impl_target")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

common_deps = [
    "//rtp_llm/cpp/cache",
    "//rtp_llm/cpp/cache/test:cache_config_test_utils",
    "//rtp_llm/cpp/config:config_modules",
    "//rtp_llm/cpp/metrics:metrics",
    "//rtp_llm/cpp/utils:core_utils",
    "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
    "@com_google_googletest//:gtest",
    "@havenask//aios/autil:log",
] + device_impl_target() + torch_deps()

cc_test(
    name = "cache_lifecycle_smoke_test",
    srcs = [
        "CacheLifecycleSmokeTest.cc",
        "CacheSmokeMain.cc",
        "CacheSmokeTestUtils.h",
    ],
    copts = test_copts,
    deps = common_deps,
    env = {},
    exec_properties = {"gpu": "H20"},
)

cc_test(
    name = "cache_pd_smoke_test",
    srcs = [
        "CachePDSmokeTest.cc",
        "CacheSmokeMain.cc",
        "CacheSmokeTestUtils.h",
    ],
    copts = test_copts,
    deps = common_deps + [
        "//rtp_llm/cpp/cache/connector/p2p:connector",
        "//rtp_llm/cpp/utils:time_util",
        "@havenask//aios/autil:net",
    ],
    env = {},
    exec_properties = {"gpu": "H20"},
)

test_suite(
    name = "cache_framework_smoke",
    tests = [
        ":cache_lifecycle_smoke_test",
        ":cache_pd_smoke_test",
    ],
)
