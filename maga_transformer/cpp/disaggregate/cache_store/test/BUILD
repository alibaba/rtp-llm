package(default_visibility = ["//maga_transformer/cpp/disaggregate/cache_store:__subpackages__"])

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

cc_test(
    name = "cache_store_gtest",
    srcs = [
        "main.cpp",
        "NormalCacheStoreTest.cpp",
        "MemoryUtilTest.cpp",
        "RequestBlockBufferTest.cpp",
        "RequestBlockBufferStoreTest.cpp",
        "MessagerClientTest.cpp",
        "CacheLoadServiceClosureTest.cpp",
        "CacheStoreServiceImplContextTest.cpp",
    ],
    deps = [
        "//maga_transformer/cpp/disaggregate/cache_store:cache_store",
        "//maga_transformer/cpp/disaggregate/cache_store:block_buffer_util",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        ":cache_store_gmock",
    ],
    copts = ["-fno-access-control",],
    tags = ["manual"],
    exec_properties = {'gpu':'A10'},
    linkstatic = 1,
)
