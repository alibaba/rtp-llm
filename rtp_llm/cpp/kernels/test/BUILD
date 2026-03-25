load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

cc_test(
    name = "cp_cache_scatter_kernel_test",
    srcs = [
        "CPCacheScatterKernelTest.cc",
    ],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/kernels:cp_cache_scatter",
        "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/utils:core_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    exec_properties = {'gpu': 'A10'},
)

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
