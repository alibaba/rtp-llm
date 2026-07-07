load("//:def.bzl", "cc_test_wrapper", "copts")

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

cc_test_wrapper(
    name = "kvs_object_store_test",
    srcs = ["KVSObjectStoreTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache/connector/kvs:kvs_object_store",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_wrapper(
    name = "kvs_connector_test",
    srcs = ["KVSConnectorTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/cache/connector/kvs:kvs_connector",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + cuda13_torch_link_deps,
)

cc_test_wrapper(
    name = "kvs_client_backend_link_test",
    srcs = ["KVSClientBackendLinkTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache/connector/kvs:kvs_client_backend",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
