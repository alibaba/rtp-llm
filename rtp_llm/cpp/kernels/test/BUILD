load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

cc_test(
    name = "virtual_cache_key_test",
    srcs = [
        "VirtualCacheKeyTest.cc",
    ],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/utils:hash_util",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
