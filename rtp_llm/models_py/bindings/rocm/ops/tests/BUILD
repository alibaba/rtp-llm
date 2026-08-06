load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

test_deps = [
    "//rtp_llm/cpp/testing:device_test_utils",
    "//rtp_llm/cpp/testing:base_tests",
    "//rtp_llm/models_py/bindings/core:sampling_ops_impl",
] + torch_deps()

cc_test(
    name = "beam_search_op_test",
    srcs = [
        "RocmBeamSearchOpTest.cc",
    ],
    data = [],
    copts = copts(),
    deps = test_deps,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

cc_test(
    name = "runtime_ops_test",
    srcs = ["RocmRuntimeOpsTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/core:copy_ops_hdrs",
        "//rtp_llm/cpp/runtime:cuda_runtime_hdrs",
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/core:runtime_ops_test_impls",
    ] + torch_deps(),
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)
