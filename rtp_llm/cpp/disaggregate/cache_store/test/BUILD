package(default_visibility = ["//rtp_llm/cpp/disaggregate/cache_store:__subpackages__"])

load("//:def.bzl", "cc_test_wrapper", "copts")
load("//rtp_llm/cpp/devices:device_defs.bzl", "device_impl_target", "device_test_envs")

cc_test = cc_test_wrapper

test_base_sources = [
    "test_util/BlockBufferUtil.cpp",
    "test_util/DeviceUtil.cpp",
]

test_base_includes = [
    "CacheStoreTestBase.h",
    "test_util/MockMemoryUtil.h",
    "test_util/BlockBufferUtil.h",
    "test_util/DeviceUtil.h",
]

cc_library(
    name = "cache_store_test_base",
    srcs = test_base_sources,
    hdrs = test_base_includes,
    deps = [
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store",
        "//rtp_llm/cpp/devices:devices_base",
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
    ],
    visibility=['//visibility:public'],
)

cc_test(
    name = "cache_store_gtest",
    srcs = glob(["*.cpp"], exclude = test_base_sources),
    deps = [
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store",
        ":cache_store_test_base",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + device_impl_target(),
    env = {
        "CACHE_STORE_RDMA_MODE": "0",
        "DEVICE_RESERVE_MEMORY_BYTES": "128000000",
    },
    copts = ["-fno-access-control"] + copts(),
    exec_properties = {'gpu':'A10'},
)

