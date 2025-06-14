load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_envs = {
    "DEVICE_RESERVE_MEMORY_BYTES": "0",
}

cc_library(
    name = "cuda_test_util",
    hdrs = [
        "CudaTestUtils.h",
    ],
    srcs = [],
    deps = [
        "@com_google_googletest//:gtest",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

test_deps = [
    "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
    "//rtp_llm/cpp/devices/testing:device_test_utils",
    "//rtp_llm/cpp/devices/base_tests:base_tests",
    ":cuda_test_util",
] + torch_deps()


cc_test(
    name = "cuda_basic_test",
    srcs = [],
    env = test_envs,
    copts = copts(),
    deps = test_deps + [
        "//rtp_llm/cpp/devices/base_tests:basic_test_cases"
    ],
    exec_properties = {'gpu':'A10'},
)

cc_binary(
    name = "cuda_dist_test_binary",
    srcs = [],
    copts = copts(),
    deps = test_deps + [
        "//rtp_llm/cpp/devices/base_tests:distributed_test",
        "@local_config_cuda//cuda:nccl",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "cuda_dist_test",
    srcs = ["cuda_dist_test_runner.py"],
    main = "cuda_dist_test_runner.py",
    env = {
        "TEST_USING_DEVICE": "CUDA",
        "FT_DISABLE_CUSTOM_AR": "1"
    },
    deps = [
        "//rtp_llm/test/utils:device_resource",
        "//rtp_llm:torch",
    ],
    data = [":cuda_dist_test_binary"],
    args = ["$(location :cuda_dist_test_binary)"],
    tags = ["multi_device"],
)

cc_binary(
    name = "custom_ar_test",
    srcs = glob([
        "CustomAllReduceTest.cc",
    ]),
    deps = test_deps + [
        "//rtp_llm/cpp/devices/torch_impl:torch_reference_impl",
        "//rtp_llm/cpp:models",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)

cc_test(
    name = "cuda_custom_all_reduce_test",
    srcs = [
        "CudaCustomAllReduceTest.cc"
    ],
    data = [],
    env = test_envs,
    copts = copts(),
    deps = test_deps + [
        "//rtp_llm/cpp/devices/cuda_impl/tests:custom_ar_test"
    ],
    tags = ["custom_ar_ut"],
)
cc_test(
    name = "cuda_custom_all_reduce_test2",
    srcs = [
        "CudaCustomAllReduceTest.cc"
    ],
    data = [],
    env = test_envs,
    copts = copts(),
    deps = test_deps + [
        "//rtp_llm/cpp/devices/cuda_impl/tests:custom_ar_test"
    ],
    tags = ["custom_ar_ut"],
)


cc_test(
    name = "cuda_general_ops_test",
    srcs = [
        "CudaGeneralOpsTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = copts(),
    deps = test_deps,
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "cuda_gemm_op_test",
    srcs = [
        "CudaGemmOpTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = copts(),
    deps = test_deps,
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "cuda_mask_logits_op_test",
    srcs = [
        "CudaMaskLogitsOpTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = copts(),
    deps = test_deps + [
        "//rtp_llm/cpp/devices/cuda_impl:gpu_base",
    ],
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "cuda_group_gemm_op_test",
    srcs = [
        "CudaGroupGemmOpTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = copts(),
    deps = test_deps,
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "cuda_act_op_test",
    srcs = [
        "CudaActOpTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = copts(),
    deps = test_deps,
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "cuda_ffn_op_test",
    srcs = [
        "CudaFFnOpTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = copts(),
    deps = test_deps,
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "cuda_fp8_moe_op_test",
    srcs = [
        "CudaFp8MoeOpTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = copts(),
    deps = test_deps,
    tags = ['H20'],
    exec_properties = {'gpu':'H20'}
)

cc_test(
    name = "cuda_attention_op_test",
    srcs = [
        "CudaAttentionOpTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = copts() + ["-fno-access-control"],
    deps = test_deps,
    tags = ['H20'],
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "cuda_attention_layer_test",
    srcs = [
        "CudaAttentionLayerTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = copts(),
    deps = test_deps,
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "cuda_softmax_op_test",
    srcs = [
        "CudaSoftmaxOpTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = copts(),
    deps = test_deps,
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "cuda_sampler_test",
    srcs = [
        "CudaSamplerTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = copts(),
    deps = test_deps,
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "layernorm_test",
    srcs = [
        "LayernormTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = copts(),
    deps = test_deps,
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "lora_linear_layer_test",
    srcs = [
        "CudaLoraLinearLayerTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = copts() + ["-fno-access-control"],
    deps = test_deps,
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "beam_search_op_test",
    srcs = [
        "CudaBeamSearchOpTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = copts(),
    deps = test_deps,
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "cuda_quantize_test",
    srcs = [
        "CudaQuantizeTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = copts(),
    linkopts = [
        "-L/usr/local/cuda/lib64",
        "-lnvToolsExt",
        "-L/usr/local/cuda/extras/CUPTI/lib64",
        "-lcupti",
    ],
    deps = test_deps,
    exec_properties = {'gpu':'A10'},
)

