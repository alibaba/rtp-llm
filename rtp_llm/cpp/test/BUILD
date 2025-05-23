load("//:def.bzl", "cuda_copts", "copts")
load("//bazel:arch_select.bzl", "torch_deps")
load("//rtp_llm/cpp/devices:device_defs.bzl",
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
    "//rtp_llm/cpp/devices/testing:device_test_utils",
    "//rtp_llm/cpp:models",
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
        "//rtp_llm/test/model_test/fake_test/testdata:testdata",
    ],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps + [
        ":test_utils",
        "//rtp_llm/cpp/devices/torch_impl:torch_reference_impl",
        "//rtp_llm/cpp:utils",
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
        "//rtp_llm/test/model_test/fake_test/testdata:testdata"
    ],
    copts = test_copts,
    linkopts = test_linkopts,
    deps = test_deps + [
        ":test_utils",
    ],
    env = select({
        "@//:using_rocm": {"TEST_USING_DEVICE": "ROCM",},
        "//conditions:default": {"TEST_USING_DEVICE": "CUDA",},
    }),
    exec_properties = {'gpu':'A10'},
)

cc_binary(
    name = "long_seq_perf",
    srcs = [
        "long_seq.cu",
    ],
    deps = [
        "//rtp_llm/cpp/cutlass:cutlass_kernels_impl",
        "//rtp_llm/cpp/cuda:cuda",
        "//rtp_llm/cpp:utils",
        "//rtp_llm/cpp/cuda:allocator_cuda",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "gemm_perf",
    srcs = [
        "int4_perf.cu"
    ],
    deps = [
        "//rtp_llm/cpp/cutlass:cutlass_kernels_impl",
        "//rtp_llm/cpp/cuda:cuda",
        "//rtp_llm/cpp:utils",
        "//rtp_llm/cpp/cuda:allocator_cuda",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "int8_perf",
    srcs = [
        "int8_gemm_perf.cu",
    ],
    deps = [
        "//rtp_llm/cpp/cutlass:cutlass_kernels_impl",
        "//rtp_llm/cpp/cuda:cuda",
        "//rtp_llm/cpp:utils",
        "//rtp_llm/cpp/cuda:allocator_cuda",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "int8_test",
    srcs = [
        "int8_gemm_test.cu",
    ],
    deps = [
        "//rtp_llm/cpp/cutlass:cutlass_kernels_impl",
        "//rtp_llm/cpp/cuda:cuda",
        "//rtp_llm/cpp:utils",
        "//rtp_llm/cpp/cuda:allocator_cuda",
        "//:th_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_sample_helpers",
    hdrs = [
        "helper_cuda.h",
        "helper_string.h",
    ],
    strip_include_prefix = "",
)

cc_test(
    name = "memory_test",
    srcs = [
        "MemoryTest.cc",
    ],
    deps = [
        "//rtp_llm/cpp/core:allocator",
        "//rtp_llm/cpp/core:memory_tracker",
        "//rtp_llm/cpp/cuda:allocator_cuda",
        "@com_google_googletest//:gtest"
    ],
    copts = copts(),
    exec_properties = {'gpu':'A10'},
)

cc_binary(
    name = "blas_perf",
    srcs = [
        "blas_perf.cc",
    ],
    deps = [
        "//rtp_llm/cpp/cuda:cuda",
        "//rtp_llm/cpp:utils",
        "//rtp_llm/cpp/cuda:allocator_cuda",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)
