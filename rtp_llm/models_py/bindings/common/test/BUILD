load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test_wrapper(
    name = "fused_copy_host_source_test",
    srcs = ["FusedCopyHostSourceTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings/common:fuse_copy_op",
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
        "//rtp_llm/models_py/bindings/core:types_hdr",
        "//rtp_llm/models_py/bindings/core:tensor_holder",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    tags = ["L20_CU13"],
)
