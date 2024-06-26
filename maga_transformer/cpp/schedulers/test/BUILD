load("//:def.bzl", "copts", "cuda_copts", "torch_deps")

test_linkopts = [
    "-lpython3.10",
    "-ltorch",
    "-lc10",
    "-ltorch_cpu",
    "-ltorch_python",
    "-L/usr/local/cuda/lib64",
    "-lcudart",
    "-lcuda",
    "-lnccl",
    "-lnvToolsExt",
]

test_copts = [
    "-fno-access-control",
] + cuda_copts() + copts()

test_deps = [
    "//src/fastertransformer/devices/testing:device_test_utils",
    "//src/fastertransformer/devices/cuda_impl:cuda_impl",
    "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_impl",
    "//3rdparty/trt_fused_multihead_attention:trt_fused_multihead_attention_impl",
    "//3rdparty/flash_attention2:flash_attention2_impl",
    "//src/fastertransformer/utils:utils",
    "//maga_transformer/cpp:schedulers",
    "//:gpt_init_params",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda",
    "@local_config_cuda//cuda:cudart",
] + torch_deps()

cc_test(
    name = "fifo_scheduler_test",
    srcs = [
        "FIFOSchedulerTest.cc",
    ],
    data = [],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },    
)
