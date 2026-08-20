load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

cuda13_torch_link_deps = select({
    "@//:using_cuda13_x86": [
        ":cuda13_torch_nvshmem",
    ],
    "//conditions:default": [],
})

test_copts = [
    "-fno-access-control",
] + copts()

cc_library(
    name = "test_log_capture",
    hdrs = ["TestLogCapture.h"],
    deps = ["//rtp_llm/cpp/utils:core_utils"],
    visibility = ["//visibility:public"],
    copts = copts(),
)

cc_library(
    name = "device_test_utils",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob(
        ["*.cc"],
        exclude = ["BasicDeviceTest.cc", "DistributedTest.cc"],
    ),
    deps = [
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "//rtp_llm/cpp/engine_base/stream:complete_token_ids",
        "//rtp_llm/cpp/runtime:cuda_runtime_hdrs",
        "//rtp_llm/cpp/runtime:cuda_runtime_impl",
    ] + cuda13_torch_link_deps,
    visibility = ["//visibility:public"],
    copts = copts(),
)

cc_library(
    name = "basic_test_cases",
    srcs = ["BasicDeviceTest.cc"],
    deps = [
        ":device_test_utils",
    ],
    visibility = ["//visibility:public"],
    copts = test_copts,
)

cc_library(
    name = "distributed_test",
    srcs = ["DistributedTest.cc"],
    deps = [
        ":device_test_utils",
    ],
    visibility = ["//visibility:public"],
    copts = test_copts,
)

cc_library(
    name = "base_tests",
    hdrs = [
        "BeamSearchOpTest.hpp",
    ],
    deps = [
        ":device_test_utils",
        "//rtp_llm/cpp/models:sampling_ops",
        "//rtp_llm/models_py/bindings/core/ops:torch_beam_search_op_impl",
        "//rtp_llm/cpp/models:models",
    ],
    visibility = ["//visibility:public"],
    copts = test_copts,
)

cc_library(
    name = "rejection_sampling_test_utils",
    hdrs = [
        "RejectionSamplingOpTest.hpp",
    ],
    deps = [
        "//rtp_llm/models_py/bindings/core:op_data",
        "//rtp_llm/models_py/bindings/core:rejection_sampling_ops_test_lib",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
    copts = test_copts,
)
