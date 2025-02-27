load("//:def.bzl", "copts")


cc_library(
    name = "trt_plugins",
    srcs = glob([
        "common/checkMacrosPlugin.cpp",
        "weightOnlyQuantMatmulPlugin/*.cpp",
        "weightOnlyGroupwiseQuantMatmulPlugin/*.cpp",
        "mixtureOfExperts/*.cpp",
        "smoothQuantGemmPlugin/*.cpp",
        "GroupGemmPlugin/*.cpp",
    ]),
    hdrs = glob([
        "*.h",
        "common/checkMacrosPlugin.h",
        "common/trtPluginsInterface.h",
        "weightOnlyQuantMatmulPlugin/*.h",
        "weightOnlyGroupwiseQuantMatmulPlugin/*.h",
        "mixtureOfExperts/*.h",
        "smoothQuantGemmPlugin/*.h",
        "GroupGemmPlugin/*.h",
    ]),
    deps = [
        "//src/fastertransformer/cutlass:cutlass_interface",
        "//src/fastertransformer/utils",
        "//src/fastertransformer/kernels:kernels",
        "//src/fastertransformer/cutlass:cutlass_kernels_impl",
        "//src/fastertransformer/cutlass:cutlass_headers",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    include_prefix = "trt_plugins",
    visibility = ["//visibility:public"],
)


cc_test(
    name = "moe_test",
    srcs = [
        "test/moe_test.cpp",
    ],
    copts = copts(),
    deps = [
        "@com_google_googletest//:gtest",
        "//src/fastertransformer/core:buffer_torch",
        "//src/fastertransformer/devices/testing:device_test_utils",
        "//src/fastertransformer/devices/cuda_impl:gpu_base",
        "//src/fastertransformer/devices/cuda_impl:cuda_impl",
        ":trt_plugins",
    ],
    tags = ["manual"],
    exec_properties = {'gpu':'A10'},
)
