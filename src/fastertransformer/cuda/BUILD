load("//:def.bzl", "rpm_library", "torch_deps", "copts", "cuda_copts",)

cc_library(
    name = "cublas",
    srcs = [
        "cublas/cublasAlgoMap.cc",
        # "cublas/cublasFP8MMWrapper.cc",
        "cublas/cublasINT8MMWrapper.cc",
        "cublas/cublasMMWrapper.cc",
    ],
    hdrs = [
        "cublas/cublasAlgoMap.h",
        # "cublas/cublasFP8MMWrapper.h",
        "cublas/cublasINT8MMWrapper.h",
        "cublas/cublasMMWrapper.h",
        "cublas/cublas.h",
    ],
    deps = [
        ":cuda_utils",
        "//src/fastertransformer/core:allocator",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nccl",
    srcs = [
        "nccl/nccl_utils_torch.cc",
        "nccl/nccl_utils.cc",
    ],
    hdrs = [
        "nccl/nccl_utils_torch.h",
        "nccl/nccl_utils.h",
    ],
    deps = torch_deps() + [
        ":cuda_utils",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nvtx",
    srcs = [
        "nvtx/nvtx_utils.cc",
        "nvtx/kernel_profiler.cc"
    ],
    hdrs = [
        "nvtx/nvtx_utils.h",
        "nvtx/kernel_profiler.h"
    ],
    deps = [
        ":cuda_utils",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "memory_utils",
    srcs = [
        "memory_utils.cu",

    ],
    hdrs = [
        "memory_utils.h",

    ],
    deps = [
        "//src/fastertransformer/cutlass:cutlass_interface",
        ":cuda_utils",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/core:Tensor",
        "//src/fastertransformer/core:allocator",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_utils_cu",
    srcs = [
        "cuda_fp8_utils.cu"
    ],
    hdrs = [
        "cuda_fp8_utils.h",
        "cuda_type_utils.cuh",
        "cuda_bf16_fallbacks.cuh",
    ],
    deps = [
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)


cc_library(
    name = "cuda_utils",
    srcs = [
        "cuda_utils.cc",
    ],
    hdrs = [
        "cuda_utils.h",
    ],
    deps = [
        ":cuda_utils_cu",
        "//src/fastertransformer/utils:utils",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda",
    srcs = ["gemm.cc"],
    hdrs = [
        "custom_ar_comm.h",
        "gemm.h"],
    deps = [
        ":nccl",
        ":nvtx",
        ":cublas",
        ":cuda_utils",
        ":memory_utils",
        "//src/fastertransformer/kernels:kernels",
        "//src/fastertransformer/utils:utils",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "allocator_cuda",
    srcs = [
        "allocator_cuda.cc"
    ],
    hdrs = [
        "allocator_cuda.h"
    ],
    copts = cuda_copts(),
    deps = [
        "//src/fastertransformer/core:allocator",
        ":cuda_utils",
    ],
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "allocator_torch",
    srcs = [
        "allocator_torch.cc"
    ],
    hdrs = [
        "allocator_torch.h"
    ],
    copts = copts(),
    deps = torch_deps() + [
        ":allocator_cuda",
    ],
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tensor_utils",
    srcs = [
        "tensor_utils.cc"
    ],
    hdrs = [
        "tensor_utils.h"
    ],
    copts = copts(),
    deps = torch_deps() + [
        "//src/fastertransformer/core:Tensor",
    ],
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gemm_test_utils",
    srcs = [
        "gemm_test/encoder_igemm_func.cc",
        "gemm_test/encoder_gemm_func.cc",
        "gemm_test/gpt_gemm_func.cc",
        "gemm_test/gemm_func.cc",
    ],
    hdrs = [
        "gemm_test/encoder_igemm_func.h",
        "gemm_test/encoder_gemm_func.h",
        "gemm_test/gpt_gemm_func.h",
        "gemm_test/gemm_func.h",
    ],
    deps = [
        ":cuda_utils",
        ":cublas",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)