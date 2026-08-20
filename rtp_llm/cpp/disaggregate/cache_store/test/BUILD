package(default_visibility = ["//rtp_llm/cpp/disaggregate/cache_store:__subpackages__"])

load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:device_defs.bzl", "device_impl_target", "device_test_envs")

cc_test = cc_test_wrapper

cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

cuda13_torch_link_deps = select({
    "@//:using_cuda13_x86": [
        ":cuda13_torch_nvshmem",
    ],
    "//conditions:default": [],
})

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
        "//rtp_llm/cpp/runtime:cuda_runtime_hdrs",
        "//rtp_llm/models_py/bindings/core:device_ops_test_impls",
        "@com_google_googletest//:gtest",
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
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + device_impl_target() + cuda13_torch_link_deps,
    env = {
        "CACHE_STORE_RDMA_MODE": "0",
    },
    copts = ["-fno-access-control"] + copts(),
    exec_properties = {'gpu':'H20'},
)
