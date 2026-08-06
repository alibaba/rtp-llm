load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test(
    name = "collective_backend_test",
    srcs = ["CollectiveBackendTest.cc"],
    deps = [
        "//rtp_llm/cpp/comm:collective_backend_impl",
        "//rtp_llm/cpp/comm:collective_backend_pybind_hdrs",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    copts = copts(),
    linkstatic = True,
)
