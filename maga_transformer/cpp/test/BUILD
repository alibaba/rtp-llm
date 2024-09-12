load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")
load("//src/fastertransformer/devices:device_defs.bzl",
    "device_impl_target", "device_test_envs", "device_linkopts")

test_copts = [
    "-fno-access-control",
] + copts()

test_linkopts = [
    "-lpython3.10",
    "-ltorch",
    "-lc10",
    "-ltorch_cpu",
    "-ltorch_python",
] + device_linkopts()

cc_library(
    name = "test_headers",
    hdrs = glob([
        "utils/*.h",
    ]),
)

test_deps = [
    "//src/fastertransformer/devices/testing:device_test_utils",
    "//maga_transformer/cpp:models",
    "//:gpt_init_params",
    ":test_headers",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
] + torch_deps() + device_impl_target()

cc_library(
    name = "test_utils",
    hdrs = [
        "ModelTestUtil.h",
    ],
    srcs = [
        "ModelTestUtil.cc",
    ],
    copts = test_copts,
    deps = test_deps,
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

cc_test(
    name = "gpt_model_test",
    srcs = [
        "GptModelTest.cc",
    ],
    data = [
        "//maga_transformer/test/model_test/fake_test/testdata:testdata",
    ],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps + [
        ":test_utils",
        "//src/fastertransformer/devices/torch_impl:torch_reference_impl",
        "//src/fastertransformer/utils:utils",
    ],
    env = device_test_envs(),
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "sampler_test",
    srcs = [
        "SamplerTest.cc",
    ],
    data = [
        "//maga_transformer/test/model_test/fake_test/testdata:testdata"
    ],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps + [
        ":test_utils",
    ],
    env = select({
        "//:using_rocm": {"TEST_USING_DEVICE": "ROCM",},
        "//conditions:default": {"TEST_USING_DEVICE": "CUDA",},
    }),
    exec_properties = {'gpu':'A10'},
)

cc_test(
    name = "unittest",
    srcs = [
        "HttpApiServerTest.cc",
    ],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps + [
        ":unittest_lib",
    ],
    env = select({
        "//:using_rocm": {"TEST_USING_DEVICE": "ROCM",},
        "//conditions:default": {"TEST_USING_DEVICE": "CUDA",},
    }),
)

cc_library(
    name = "unittest_lib",
    srcs = [
        "//:th_transformer_lib_files",
    ],
    deps = [
        "//:gpt_init_params_hdr",
    	"//:th_op_hdrs",
        "//src/fastertransformer/utils:utils",
        "//maga_transformer/cpp:http_api_server",
        "//maga_transformer/cpp:model_rpc_server",
        "@grpc//:grpc++",
    ] + select({
        "//:using_cuda": [
            "//src/fastertransformer/cuda:allocator_torch",
        ],
        "//conditions:default": [],
    }),
    copts = test_copts,
    alwayslink = True,
    linkstatic = True,  # 这个很关键, 单测必须要加这个
    visibility = ["//visibility:public"],
)