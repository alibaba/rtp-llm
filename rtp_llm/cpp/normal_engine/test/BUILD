load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

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

cc_library(
    name = "mock_engine",
    hdrs = [
        "MockEngine.h"
    ],
    srcs = [
    ],
    copts = test_copts,
    deps =  [
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/engine_base:weights_converter",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

# No exec_properties on purpose: this depends only on the header-only warmup_role_gate target, so
# it needs neither a GPU nor torch. That is why the role classification was split out of
# NormalEngine.cc -- every other test in this directory requires an H20 host.
# Known gap: this target pins role classification only. WarmupTraceScope's sample/reset ordering
# and makeWarmUpResult's non-torch warning threshold still live in NormalEngine.cc and have no
# host-only orchestration test because executor reset and CUDA synchronization are not injectable.
cc_test(
    name = "warmup_role_gate_test",
    srcs = ["WarmupRoleGateTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/normal_engine:warmup_role_gate",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "normal_output_dispatcher_test",
    srcs = ["NormalOutputDispatcherTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    exec_properties = {"gpu": "H20"},
)

cc_test(
    name = "batch_stream_processor_test",
    srcs = glob([
        "NormalBatchStreamProcessorTest.cc",
    ]),
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/models:models",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "engine_test",
    srcs = glob([
        "EngineTest.cc",
    ]),
    data = [],
    copts = test_copts,
    deps =  [
        ":mock_engine",
    ] + cuda13_torch_link_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "async_runner_test",
    srcs = [
        "AsyncRunnerTest.cc",
    ],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'H20'},
)
