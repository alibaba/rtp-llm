load("//:def.bzl", "cc_test_wrapper", "copts", "cuda_copts")
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
    name = "mtp_multimodal_tp_sync_wrapper_lib",
    testonly = True,
    srcs = ["mtp_multimodal_tp_sync_wrapper.cc"],
    copts = cuda_copts(),
    deps = [
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
    ] + torch_deps() + cuda13_torch_link_deps,
    alwayslink = True,
)

cc_binary(
    name = "mtp_multimodal_tp_sync_wrapper",
    testonly = True,
    deps = [":mtp_multimodal_tp_sync_wrapper_lib"],
    linkshared = True,
)

py_test(
    name = "mtp_multimodal_tp_sync_test",
    srcs = ["mtp_multimodal_tp_sync_test.py"],
    data = [":mtp_multimodal_tp_sync_wrapper"],
    deps = ["//rtp_llm:testlib"],
    timeout = "moderate",
    env = {"GPU_COUNT": "2"},
    tags = ["open_skip"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "2"},
)

cc_test(
    name = "speculative_sampler_test",
    srcs = [
        "SpeculativeSamplerTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/normal_engine:normal_engine",
        # FastTopKSampler::forward calls execMappingDraft2Target (draft-to-target
        # vocab mapping); pair exec_ops_test_lib with cuda_impl like models/test.
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    # CPU-only logic, but normal_engine links libtorch_cuda, so the loader needs a
    # CUDA host. Any generic CUDA worker suffices (default gpu=gpu_cuda12) — no H20.
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
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/models:logits_processor",
        "//rtp_llm/cpp/config:config_modules",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
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
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/normal_engine/test:mock_engine",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)

# Isolated process: executor env switches are statically cached.
cc_test(
    name = "mtp_host_compact_prepare_test",
    srcs = glob([
        "MtpExecutorTest.cc",
    ]),
    args = ["--gtest_filter=MtpExecutorTest.testQwenMrope*"],
    data = [],
    copts = test_copts + ["-DRTP_MTP_GRAPH_PREPARATION_REGRESSION"],
    deps = [
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/normal_engine/test:mock_engine",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
        "GPU_COUNT": "1",
        "RTP_LLM_MTP_ASYNC_PREPARE": "1",
        "RTP_LLM_STREAM_ASYNC": "0",
        "RTP_LLM_MTP_ASYNC_DEVICE_STATE": "0",
        "RTP_LLM_DEVICE_INPUT": "0",
        "RTP_LLM_DEVICE_INPUT_CHECK": "0",
    },
    exec_properties = {"gpu": "SM100_ARM"},
)

# Isolated process: executor env switches are statically cached.
cc_test(
    name = "mtp_dense_prepare_test",
    srcs = glob([
        "MtpExecutorTest.cc",
    ]),
    args = ["--gtest_filter=MtpExecutorTest.testQwenMrope*"],
    data = [],
    copts = test_copts + ["-DRTP_MTP_GRAPH_PREPARATION_REGRESSION", "-DRTP_MTP_DENSE_PREPARATION_REGRESSION"],
    deps = [
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/normal_engine/test:mock_engine",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
        "GPU_COUNT": "1",
        "RTP_LLM_MTP_ASYNC_PREPARE": "1",
        "RTP_LLM_STREAM_ASYNC": "0",
        "RTP_LLM_MTP_ASYNC_DEVICE_STATE": "1",
        "RTP_LLM_DEVICE_INPUT": "0",
        "RTP_LLM_DEVICE_INPUT_CHECK": "0",
    },
    exec_properties = {"gpu": "SM100_ARM"},
)
