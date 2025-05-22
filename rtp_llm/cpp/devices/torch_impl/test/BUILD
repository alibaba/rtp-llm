load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_deps = [
    "@com_google_googletest//:gtest",
] + torch_deps()

cc_test(
    name = "torch_beam_search_test",
    srcs = [
        "TorchBeamSearchOpTest.cc",
    ],
    data = [],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/devices/torch_impl:torch_beam_search_op_impl",
    ] + test_deps,
)