package(default_visibility = ["//maga_transformer/cpp/disaggregate/cache_store:__subpackages__"])
load("//:def.bzl", "copts")

cc_library(
    name = "cache_store_test_base",
    srcs = [
        "test_util/BlockBufferUtil.cpp",
        "test_util/DeviceUtil.cpp",
    ],
    hdrs = [
        "CacheStoreTestBase.h",
        "test_util/MockMemoryUtil.h",
        "test_util/BlockBufferUtil.h",
        "test_util/DeviceUtil.h",
    ],
    deps = [
        "//maga_transformer/cpp/disaggregate/cache_store:cache_store",
        "//src/fastertransformer/devices:devices_base",
        "@com_google_googletest//:gtest",
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
        ":cache_store_test_base",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    env = {
        "CACHE_STORE_RDMA_MODE": "0",
        "DEVICE_RESERVE_MEMORY_BYTES": "128000000",
    },
    copts = ["-fno-access-control",] + copts(),
    exec_properties = {'gpu':'A10'},
    linkstatic = 1,
)

