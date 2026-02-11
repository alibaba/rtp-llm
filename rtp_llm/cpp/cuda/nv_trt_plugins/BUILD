load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper


cc_library(
    name = "nv_trt_plugins",
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
        "//rtp_llm/cpp/cuda/cutlass:cutlass_interface",
        "//rtp_llm/cpp/cuda:trt_utils",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/kernels:kernels_cu",
        "//rtp_llm/cpp/cuda/cutlass:cutlass_kernels_impl",
        "//rtp_llm/cpp/cuda/cutlass:cutlass_headers",
        "//rtp_llm/cpp/model_utils:model_utils",
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
        "//rtp_llm/cpp/core:buffer_torch",
        "//rtp_llm/cpp/devices/testing:device_test_utils",
        "//rtp_llm/cpp/devices/cuda_impl:gpu_base",
        "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
        ":trt_plugins",
    ],
    tags = ["manual"],
    exec_properties = {'gpu':'A10'},
)
