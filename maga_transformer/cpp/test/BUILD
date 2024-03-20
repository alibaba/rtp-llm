load("//:def.bzl", "copts", "cuda_copts", "torch_deps")

test_copts = [
    "-fno-access-control",
] + cuda_copts() + copts()

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

cc_library(
    name = "test_headers",
    hdrs = glob([
        "utils/*.h",
    ]),
)

test_deps = [
    "//src/fastertransformer/devices/testing:device_test_utils",
    "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_impl",
    "//maga_transformer/cpp:maga_op_lib",
    ":test_headers",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda",
    "@local_config_cuda//cuda:cudart",
] + torch_deps()

cc_library(
    name = "test_utils",
    hdrs = [
        "ModelTestUtil.h"
    ],
    srcs = [
        "ModelTestUtil.cc",
    ],
    copts = test_copts,
    deps = test_deps,
    alwayslink = 1,
)

cc_test(
    name = "gpt_model_test",
    srcs = [
        "GptModelTest.cc",
    ],
    data = [
        "//maga_transformer/test/model_test/fake_test/testdata:testdata"
    ],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps + [
        ":test_utils",
    ],
)

