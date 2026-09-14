load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

cc_library(
    name = "cuda13_arm_torch_runtime",
    # Torch 2.11 ARM bundles these outside the legacy torch.libs directory.
    # Test executables need explicit link and runfiles dependencies.
    srcs = [
        "@pip_cuda13_arm_torch_torch//:site-packages/torch/lib/libarm_compute.so",
        "@pip_cuda13_arm_torch_torch//:site-packages/torch/lib/libarm_compute_graph.so",
        "@pip_cuda13_arm_torch_torch//:site-packages/torch/lib/libgfortran.so.5",
        "@pip_cuda13_arm_torch_torch//:site-packages/torch/lib/libnvpl_blas_core.so.0",
        "@pip_cuda13_arm_torch_torch//:site-packages/torch/lib/libnvpl_blas_lp64_gomp.so.0",
        "@pip_cuda13_arm_torch_torch//:site-packages/torch/lib/libnvpl_lapack_core.so.0",
        "@pip_cuda13_arm_torch_torch//:site-packages/torch/lib/libnvpl_lapack_lp64_gomp.so.0",
        "@pip_cuda13_arm_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
        "@pip_cuda13_arm_torch_nvidia_nvshmem_cu13//:site-packages/nvidia/nvshmem/lib/libnvshmem_host.so.3",
    ],
)

cuda13_torch_link_deps = select({
    "@//:using_cuda13_x86": [
        ":cuda13_torch_nvshmem",
    ],
    "@//:using_cuda13_arm": [
        ":cuda13_arm_torch_runtime",
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

cc_test(
    name = "deferred_engine_start_test",
    srcs = ["DeferredEngineStartTest.cc"],
    copts = test_copts,
    deps = [
        ":mock_engine",
        "//rtp_llm/cpp/model_rpc:model_rpc_server",
        "@havenask//aios/autil:base64",
        "@havenask//aios/autil:zlib",
    ] + cuda13_torch_link_deps,
    env = {"TEST_USING_DEVICE": "CUDA"},
    exec_properties = {"gpu": "H20"},
)
