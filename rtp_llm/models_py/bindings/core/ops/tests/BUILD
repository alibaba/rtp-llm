load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

cc_test(
    name = "torch_beam_search_test",
    srcs = [
        "TorchBeamSearchOpTest.cc",
    ],
    data = [],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings/core/ops:torch_beam_search_op_impl",
        "@com_google_googletest//:gtest",
    ] + torch_deps(),
)
