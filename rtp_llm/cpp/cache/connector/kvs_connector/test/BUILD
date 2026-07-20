load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

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

cc_test(
    name = "kvs_connector_test",
    srcs = [
        "KVSConnectorTest.cc",
    ],
    copts = [
        "-fno-access-control",
    ] + copts(),
    deps = [
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/cache:kv_cache_allocator",
        "//rtp_llm/cpp/cache/connector:connector_base",
        "//rtp_llm/cpp/cache/connector/kvs_connector",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/utils:core_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps() + cuda13_torch_link_deps,
)

cc_test(
    name = "kvs_client_integration_test",
    srcs = [
        "KVSClientIntegrationTest.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache/connector/kvs_connector",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + cuda13_torch_link_deps,
)
