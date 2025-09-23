load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")



test_deps = [
    "@com_google_googletest//:gtest",
] + torch_deps()


cc_test(
    name = "qbuffer_test",
    srcs = [
        "QBufferTest.cc",
    ],
    copts = copts(),
    deps = test_deps + [
        "//rtp_llm/cpp/core:buffer",
        "//rtp_llm/cpp/core:types",
        "//rtp_llm/cpp/core:buffer_torch",

    ],
    visibility = ["//visibility:public"],
)
