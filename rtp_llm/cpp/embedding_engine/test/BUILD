load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test_wrapper(
    name = "embedding_scheduler_stop_test",
    srcs = ["EmbeddingSchedulerStopTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/embedding_engine:embedding_engine",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    exec_properties = {'gpu':'H20'},
)
