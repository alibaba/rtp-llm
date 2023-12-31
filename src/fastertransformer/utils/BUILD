load("//:def.bzl", "rpm_library", "torch_deps", "copts", "cuda_copts",)

cc_library(
    name = "utils_for_3rdparty",
    srcs = [
        "cuda_utils.cc",
        "logger.cc",
    ],
    hdrs = [
        "cuda_bf16_wrapper.h",
        "cuda_utils.h",
        "cuda_fp8_utils.h",
        "logger.h",
        "string_utils.h",
    ],
    deps = [
        "//3rdparty:ini_reader",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "memory_utils_cu",
    srcs = [
        "memory_utils.cu"
    ],
    hdrs = [
        "memory_utils.h"
    ],
    deps = [
        "//src/fastertransformer/cutlass:cutlass_interface",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
        ":utils_cu",
        ":utils_for_3rdparty"
    ],
    copts = cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "utils_cu",
    srcs = glob([
        "*.cu",
    ], exclude = [ # fp8 codes
        "cublasFP8MMWrapper.cu",
        "memory_utils.cu",
    ]),
    hdrs = glob([
        "*.h",
        "*.cuh",
    ], exclude = [
        "allocator_impl.h",
        "cublasFP8MMWrapper.h",
        "nccl_utils.h",
        "nccl_utils_torch.h",
        "serialize_utils.h",
        "memory_utils.h",
    ]),
    deps = [
        "//3rdparty:cub",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "utils",
    srcs = glob([
        "*.cc",
    ], exclude=[
        "logger.cc",
        "cuda_utils.cc",
        "custom_ar_comm.cc",
        "nccl_utils.cc",
        "nccl_utils_torch.cc",
        "serialize_utils.cc",
    ]),
    deps = [
        ":memory_utils_cu",
        ":utils_cu",
        ":utils_for_3rdparty",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nccl_utils",
    srcs = [
	    "nccl_utils.cc",
        "nccl_utils_torch.cc"
    ],
    hdrs = [
    	"nccl_utils.h",
        "nccl_utils_torch.h",
    ],
    deps = torch_deps() + [
        ":utils",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "torch_utils",
    srcs = [
    ],
    hdrs = [
        "allocator_impl.h",
    ],
    deps = [
        ":nccl_utils",
        ":utils",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    alwayslink = True,
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
        ":utils",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "serialize_utils",
    srcs = [
        "serialize_utils.cc",
    ],
    hdrs = [
        "serialize_utils.h",
    ],
    deps = [
        "//src/fastertransformer/kernels:kernels_cu",
    ],
    copts = copts(),
    alwayslink = True,
    visibility = ["//visibility:public"],
)
