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

# Keep the speculative lifecycle and sampler tests runnable on both GPU stacks.
_spec_gpu_deps = select({
    "@//:using_rocm": [
        "//rtp_llm/models_py/bindings/core:exec_ctx_rocm",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:rocm",
    ],
    "//conditions:default": [
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
})

cc_test(
    name = "speculative_sampler_test",
    srcs = [
        "SpeculativeSamplerTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + _spec_gpu_deps + torch_deps(),
    # CPU tensor tests still link the selected GPU runtime.
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "mtp_batch_stream_processor_test",
    srcs = glob([
        "MtpBatchStreamProcessorTest.cc",
    ]),
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/models:logits_processor",
        "//rtp_llm/cpp/config:config_modules",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + _spec_gpu_deps + torch_deps() + cuda13_torch_link_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "mtp_executor_test",
    srcs = glob([
        "MtpExecutorTest.cc",
    ]),
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/normal_engine/test:mock_engine",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + _spec_gpu_deps + torch_deps() + cuda13_torch_link_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)
