load("//:def.bzl", "copts", "cuda_copts")

cc_library(
    name = "cutlass_kernels_common",
    srcs = glob([
        "cutlass_kernels/*.cc",
    ]),
    hdrs = glob([
        "cutlass_kernels/*.h",
    ]),
    deps = [
        "//src/fastertransformer/cutlass/cutlass_extensions:cutlass_extensions",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/cuda:cuda",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    # include_prefix = "src",
    copts = copts(),
    visibility = ["//visibility:public"],
    alwayslink = True,
)

cc_library(
    name = "weight_only_gemm_cu",
    srcs = glob([
            "cutlass_kernels/weightOnlyBatchedGemv/weightOnlyBatchedGemvInt8b.cu",
            "cutlass_kernels/weightOnlyBatchedGemv/kernelLauncher.cu",
            "cutlass_kernels/weightOnlyBatchedGemv/enabled.cc"
        ]),
    hdrs = glob([
            "cutlass_kernels/weightOnlyBatchedGemv/*.h",         
            "interface.h"
        ]),
    deps = [
    	":cutlass_kernels_common",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
    alwayslink = True,
)

cc_library(
    name = "fpA_intB_cu",
    srcs = glob([
        "cutlass_kernels/fpA_intB_gemm/*.cu",
        "cutlass_kernels/fpA_intB_gemm/*.h",
    ]),
    hdrs = glob([
        "cutlass_kernels/fpA_intB_gemm/*.h",
    ]),
    deps = [
    	":cutlass_kernels_common",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    alwayslink = True,
)

cc_library(
    name = "moe_cu",
    srcs = glob([
        "cutlass_kernels/moe_gemm/*.cu",
        "cutlass_kernels/moe_gemm/*.h",
    ]),
    hdrs = glob([
        "cutlass_kernels/moe_gemm/moe_gemm_kernels_template.h",
        "cutlass_kernels/moe_gemm/moe_gemm_kernels.h",
        "cutlass_kernels/moe_gemm/moe_kernels.h",
    ]),
    include_prefix = "src",
    deps = [
    	":cutlass_kernels_common",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    alwayslink = True,
)


cc_library(
    name = "group_cu",
    srcs = glob([
        "cutlass_kernels/group_gemm/*.cu",
        "cutlass_kernels/group_gemm/*.h",
    ]),
    hdrs = glob([
        "cutlass_kernels/group_gemm/group_gemm_template.h",
        "cutlass_kernels/group_gemm/group_gemm.h",
    ]),
    include_prefix = "src",
    deps = [
    	":cutlass_kernels_common",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    alwayslink = True,
)

cc_library(
    name = "cutlass_headers",
    deps = [
        "@cutlass//:cutlass",
        "@cutlass//:cutlass_utils",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cutlass_kernels_impl",
    deps = [
        ":fpA_intB_cu",
        ":weight_only_gemm_cu",
        ":group_cu",
        ":moe_cu"
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cutlass_interface",
    hdrs = [
        "interface.h",
        "cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h",
        "cutlass_kernels/group_gemm/group_gemm.h",
        "cutlass_kernels/moe_gemm/moe_gemm_kernels.h",
        "cutlass_kernels/moe_gemm/moe_kernels.h",
        "cutlass_kernels/cutlass_preprocessors.h",
        "cutlass_kernels/weight_only_quant_op.h",
        "cutlass_kernels/gemm_configs.h",
        "cutlass_kernels/weightOnlyBatchedGemv/kernelLauncher.h",
        "cutlass_kernels/weightOnlyBatchedGemv/common.h"
    ],
    visibility = ["//visibility:public"],
)