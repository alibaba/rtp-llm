load("//:def.bzl", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "cutlass_kernels_interface")

cutlass_kernels_interface()

cc_library(
    name = "trt_plugins",
    srcs = glob([
        "common/checkMacrosPlugin.cpp",
        "weightOnlyQuantMatmulPlugin/*.cpp",
        "weightOnlyGroupwiseQuantMatmulPlugin/*.cpp",
        "mixtureOfExperts/*.cpp",
    ]), 
    hdrs = glob([
        "*.h",
        "common/checkMacrosPlugin.h",
        "weightOnlyQuantMatmulPlugin/*.h",
        "weightOnlyGroupwiseQuantMatmulPlugin/*.h",
        "mixtureOfExperts/*.h",
    ]),
    deps = [
        "//src/fastertransformer/cutlass:cutlass_interface",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/kernels:kernels",
        "cutlass_kernels_interface",
        "cutlass_headers_interface",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

