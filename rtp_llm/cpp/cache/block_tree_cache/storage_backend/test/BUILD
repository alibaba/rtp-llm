load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test_wrapper(
    name = "storage_backend_test",
    srcs = ["StorageBackendTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache/block_tree_cache",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
)
