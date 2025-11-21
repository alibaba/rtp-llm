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
        "@local_config_cuda//cuda:cuda_headers",
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
    linkstatic = False,
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
        "//rtp_llm/cpp/models:models",
    ],
    copts = copts(),
    linkstatic = False,
    visibility = ["//visibility:public"],
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
    tags = ["open_skip", "H20"],
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
    tags = ["open_skip", "H20"],
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
    name = "cuda_unfused_attention_test",
    srcs = [
        "CudaUnfusedAttentionTest.cc",
    ],
    data = [],
    env = test_envs,
    copts = copts() + ["-fno-access-control"],
    deps = test_deps,
    tags = ['A10'],
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
    deps = test_deps,
    exec_properties = {'gpu':'A10'},
)

cc_library(
    name = "test_cuda_graph_decode_libs",
    srcs = glob([
        "CudaGraphDecodePaddingOp.cc"
    ]),
    hdrs = [
        "CudaGraphDecodePaddingOp.h"
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/pybind:py_utils",
        "//rtp_llm/cpp/pybind:th_compute_lib",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
    alwayslink = True, ## make sure pybind11 link
)

cc_library(
    name = "test_cuda_graph_prefill_libs",
    srcs = glob([
        "CudaGraphPrefillOp.cc"
    ]),
    hdrs = [
        "CudaGraphPrefillOp.h"
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/pybind:py_utils",
        "//rtp_llm/cpp/pybind:th_compute_lib",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
    alwayslink = True, ## make sure pybind11 link
)

cc_binary(
    name = "test_cuda_graph_decode_ops",
    deps = [
        ":test_cuda_graph_decode_libs",
    ],
    linkshared = 1,
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "test_cuda_graph_prefill_ops",
    deps = [
        ":test_cuda_graph_prefill_libs",
    ],
    linkshared = 1,
    visibility = ["//visibility:public"],
)

py_test(
    name = "CudaGraphDecodePadding",
    srcs = [
        "CudaGraphDecodePadding.py",
    ],
    data = [
        ":test_cuda_graph_decode_ops",
        "//:th_transformer"
    ],
    deps = [
        "//rtp_llm/test/model_test/test_util:test_util",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
        "HACK_LAYER_NUM" : "1",
        "CUDA_LAUNCH_BLOCKING" : "1",
    },
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "CudaGraphPrefill",
    srcs = [
        "CudaGraphPrefill.py",
    ],
    data = [
        ":test_cuda_graph_prefill_ops",
        "//:th_transformer"
    ],
    deps = [
        "//rtp_llm/test/model_test/test_util:test_util",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
        "HACK_LAYER_NUM" : "1",
        "CUDA_LAUNCH_BLOCKING" : "1",
        "ENABLE_CUDA_GRAPH_DEBUG_MODE" : "1",
    },
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "cuda_graph_copy_kernel_test",
    srcs = [
        "CudaGraphCopyKernelTest.cc",
    ],
    data = [],
    env = {
        "TEST_USING_DEVICE": "CUDA",
        "HACK_LAYER_NUM" : "1",
        "CUDA_LAUNCH_BLOCKING" : "1",
        "ENABLE_CUDA_GRAPH_DEBUG_MODE" : "1",
    },
    copts = copts() + ["-fno-access-control"],
    deps = test_deps,
    exec_properties = {'gpu':'A10'},
)
