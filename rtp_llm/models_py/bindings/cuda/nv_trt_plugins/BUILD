load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper


cc_library(
    name = "nv_trt_plugins",
    srcs = glob([
        "mixtureOfExperts/*.cpp",
    ]),
    hdrs = glob([
        "mixtureOfExperts/*.h",
    ]),
    deps = [
        "//rtp_llm/models_py/bindings/cuda/cutlass:cutlass_interface",
        "//rtp_llm/models_py/bindings/cuda:trt_utils",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings/common/kernels:kernels_cu",
        "//rtp_llm/models_py/bindings/cuda/cutlass:cutlass_kernels_impl",
        "//rtp_llm/models_py/bindings/cuda/cutlass:cutlass_headers",
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
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        ":trt_plugins",
    ],
    tags = ["manual"],
    exec_properties = {'gpu':'H20'},
)
