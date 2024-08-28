load("//:def.bzl", "cuda_copts", "copts")

cc_binary(
    name = "long_seq_perf",
    srcs = [
        "long_seq.cu",
    ],
    deps = [
        "//src/fastertransformer/cutlass:cutlass_kernels_impl",
        "//src/fastertransformer/cuda:cuda",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/cuda:allocator_cuda",
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
        "//src/fastertransformer/cutlass:cutlass_kernels_impl",
        "//src/fastertransformer/cuda:cuda",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/cuda:allocator_cuda",
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
        "//src/fastertransformer/cutlass:cutlass_kernels_impl",
        "//src/fastertransformer/cuda:cuda",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/cuda:allocator_cuda",
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
        "//src/fastertransformer/cutlass:cutlass_kernels_impl",
        "//src/fastertransformer/cuda:cuda",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/cuda:allocator_cuda",
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
        "//src/fastertransformer/core:allocator",
        "//src/fastertransformer/core:memory_tracker",
        "//src/fastertransformer/cuda:allocator_cuda",
        "@com_google_googletest//:gtest"
    ],
    copts = copts(),
    exec_properties = {'gpu':'A10'},
)
