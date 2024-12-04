package(default_visibility = ["//maga_transformer/cpp/disaggregate/cache_store:__subpackages__"])
load("//:def.bzl", "copts")

cc_library(
    name = "cache_store_gmock",
    hdrs = [
        "MockMemoryUtil.h",
    ],
    deps = [
        "//maga_transformer/cpp/disaggregate/cache_store:cache_store",
        "@com_google_googletest//:gtest",
    ]
)

cc_library(
    name = "block_buffer_util",
    srcs = [
        "BlockBufferUtil.cpp",
        "DeviceUtil.cpp",
    ],
    hdrs = [
        "BlockBufferUtil.h",
        "DeviceUtil.h",
    ],
    deps = [
        "//maga_transformer/cpp/disaggregate/cache_store:cache_store",
        "//src/fastertransformer/devices:devices_base",
    ],
    visibility=['//visibility:public'],
)

cc_test(
    name = "cache_store_gtest",
    srcs = [
        "main.cpp",
        "NormalCacheStoreTest.cpp",
        "MemoryUtilTest.cpp",
        "MessagerClientTest.cpp",
        "CacheLoadServiceClosureTest.cpp",
        "CacheStoreServiceImplContextTest.cpp",
        "RequestBlockBufferTest.cpp",
        "RequestBlockBufferStoreTest.cpp",
    ],
    deps = [
        "//maga_transformer/cpp/disaggregate/cache_store:cache_store",
        "//src/fastertransformer/devices/cuda_impl:cuda_impl",
        "//src/fastertransformer/devices/cuda_impl/tests:cuda_test_util",
        "//src/fastertransformer/devices/testing:device_test_utils",
        "//src/fastertransformer/devices/base_tests:base_tests",
        ":block_buffer_util",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        ":cache_store_gmock",
    ],
    env = {
        "CACHE_STORE_RDMA_MODE": "0",
        "DEVICE_RESERVE_MEMORY_BYTES": "128000000",
    },
    copts = ["-fno-access-control",] + copts(),
    exec_properties = {'gpu':'A10'},
    linkstatic = 1,
)

