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


py_test (
    name = "model_rpc_client_test",
    srcs = ["model_rpc_client_test.py"],
    deps = [
        "//maga_transformer/cpp:model_rpc_client",
        "//maga_transformer:testlib",
        "//maga_transformer/test/model_test/test_util:test_util"
    ],
)

# cc_test(
#     name = "model_rpc_server_test",
#     srcs = [
#         "ModelRpcServerTest.cc",
#     ],
#     copts = test_copts,
#     linkopts = test_linkopts,
#     deps = [
#         "//maga_transformer/cpp/proto:model_rpc_service_cc_proto",
#         "//maga_transformer/cpp:components",
#         "@com_google_googletest//:gtest",
#         "@com_google_googletest//:gtest_main",
#         "//maga_transformer/cpp:model_rpc_server",
#         "//maga_transformer/cpp:utils",
#     ] + torch_deps(),
# )

cc_test(
    name = "query_converter_test",
    srcs = [
        "QueryConverterTest.cc",
    ],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = [
        "//maga_transformer/cpp/proto:model_rpc_service_cc_proto",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "//maga_transformer/cpp:model_rpc_server",
        "//:gpt_init_params",
        "//maga_transformer/cpp:utils",
        "//maga_transformer/cpp:dataclass",
        "//src/fastertransformer/devices:devices_base",
        "//src/fastertransformer/devices/cuda_impl:cuda_impl",
        "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_impl",
        "//3rdparty/trt_fused_multihead_attention:trt_fused_multihead_attention_impl",
        "//3rdparty/flash_attention2:flash_attention2_impl",
        "//src/fastertransformer/devices/testing:device_test_utils"
    ] + torch_deps(),
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
)
