load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

test_deps = [
    "//rtp_llm/models_py/bindings/core:exec_ctx_rocm",
    "//rtp_llm/cpp/testing:device_test_utils",
    "//rtp_llm/cpp/testing:base_tests",
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
    name = "rejection_sampling_op_test",
    srcs = [
        "RocmRejectionSamplingOpTest.cc",
    ],
    data = [],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/testing:rejection_sampling_test_utils",
    ] + torch_deps(),
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)
