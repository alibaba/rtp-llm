load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test(
    name = "embedding_input_metadata_test",
    srcs = ["embedding_input_metadata_test.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/embedding_engine:embedding_input_metadata",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
)
