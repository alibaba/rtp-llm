package(default_visibility = ["//rtp_llm/cpp/disaggregate/cache_store:__subpackages__"])

load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:device_defs.bzl", "device_impl_target", "device_test_envs")

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
        "@com_google_googletest//:gtest",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    visibility=['//visibility:public'],
)

cc_test(
    name = "cache_store_gtest",
    srcs = glob(["*.cpp"], exclude = test_base_sources),
    deps = [
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store",
        ":cache_store_test_base",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + device_impl_target(),
    env = {
        "CACHE_STORE_RDMA_MODE": "0",
    },
    copts = ["-fno-access-control"] + copts(),
    exec_properties = {'gpu':'H20'},
)

