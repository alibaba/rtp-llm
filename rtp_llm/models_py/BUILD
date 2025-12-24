load("//bazel:arch_select.bzl", "requirement")

py_library(
    name = "utils",
    srcs = glob([
        "utils/*.py",
    ]),
)

# flashinfer-python is only available for CUDA12
flashinfer = ["flashinfer-python"]
deep_gemm = ["deep_gemm"]
requirement(flashinfer)
requirement(deep_gemm)


filegroup(
    name = "cutlass_moe_config",
    srcs = glob(["kernels/cuda/fp8_kernel/cutlass_groupgemm/*"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "modules",
    srcs = glob([
        "modules/*.py",
        "modules/**/*.py",
    ]),
    data = [":cutlass_moe_config"],
    deps = [
        ":utils",
        "//rtp_llm/models_py/triton_kernels:triton_kernels",
        "//rtp_llm/models_py/distributed:deepep_wrapper",
        "//rtp_llm/models_py/distributed:collective_torch",
    ] + select({
        "@//:using_cuda12": flashinfer + deep_gemm,
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

py_library(
    name = "modules_cuda",
    srcs = glob([
        "modules/cuda/*.py",
        "modules/cuda/**/*.py",
    ]),
    visibility = [":__subpackages__"],
)

py_library(
    name = "modules_rocm",
    srcs = glob([
        "modules/rocm/*.py",
        "modules/rocm/**/*.py",
    ]),
    visibility = [":__subpackages__"],
)

py_library(
    name = "kernels",
    srcs = glob([
        "kernels/*.py",
        "kernels/**/*.py",
    ])
)

py_library(
    name = "configs",
    srcs = glob([
        "configs/*.py",
    ]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "models",
    srcs = glob([
        "model_desc/*.py",
        "model_desc/cuda_graph/*.py"
    ]),
    deps = [
        ":utils",
        ":modules",
        ":kernels",
        ":configs",
        "//rtp_llm/model_loader:loader",
        "@flash-linear-attention//:fla_lib_py",
    ] + select({
        "@//:using_cuda12": [
            ":modules_cuda",
        ],
        "@//:using_rocm": [
            ":modules_rocm"
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)
