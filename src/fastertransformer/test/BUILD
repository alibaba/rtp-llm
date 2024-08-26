load("//:def.bzl", "cuda_copts", "copts")
load("//bazel:arch_select.bzl", "cutlass_kernels_interface")

cutlass_kernels_interface()

cc_binary(
    name = "long_seq_perf",
    srcs = [
        "long_seq.cu",
    ],
    deps = [
        "cutlass_kernels_interface",
        "//src/fastertransformer/cuda:cuda",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/cuda:allocator_cuda",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "gemm_perf",
    srcs = [
        "fpa_intb_perf.cu"
    ],
    deps = [
        "cutlass_kernels_interface",
        "//src/fastertransformer/cuda:cuda",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/cuda:allocator_cuda",
        "@local_config_cuda//cuda:cuda",
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
        "cutlass_kernels_interface",
        "//src/fastertransformer/cuda:cuda",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/cuda:allocator_cuda",
        "@local_config_cuda//cuda:cuda",
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
        "cutlass_kernels_interface",
        "//src/fastertransformer/cuda:cuda",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/cuda:allocator_cuda",
        "//:th_utils",
        "@local_config_cuda//cuda:cuda",
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
