load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")
load("//bazel:device_defs.bzl", "device_impl_target")

cc_test_wrapper(
    name = "rtp_llm_op_stop_test",
    srcs = ["RtpLLMOpStopTest.cc"],
    copts = copts() + ["-fno-access-control"],
    deps = [
        "//rtp_llm/cpp/pybind:th_transformer_lib",
        "//rtp_llm/cpp/model_rpc:model_rpc_server",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "@havenask//aios/autil:base64",
        "@havenask//aios/autil:zlib",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps() + device_impl_target(),
    timeout = "short",
    exec_properties = {"gpu": "SM100_ARM_CU13"},
    tags = ["SM100_ARM_CU13"],
)
