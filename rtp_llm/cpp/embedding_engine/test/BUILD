load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

cc_test_wrapper(
    name = "embedding_stream_shutdown_test",
    srcs = ["EmbeddingStreamShutdownTest.cc"],
    copts = copts(),
    timeout = "short",
    exec_properties = {"gpu": "H20"},
    deps = [
        "//rtp_llm/cpp/embedding_engine:embedding_engine",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps() + select({
        "@//:using_cuda13_x86": [":cuda13_torch_nvshmem"],
        "//conditions:default": [],
    }),
)
