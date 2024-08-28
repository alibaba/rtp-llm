load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "trt_plugins")

trt_plugins()

cc_library(
    name = "cuda_impl",
    hdrs = glob([
        "*.h",
    ]) + [
    ],
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        "//src/fastertransformer/devices:devices_base",
        "//src/fastertransformer/devices:device_utils",
        "//src/fastertransformer/devices:devices_base_impl",
        "//src/fastertransformer/cuda:cuda",
        "//src/fastertransformer/cuda:allocator_cuda",
        "//src/fastertransformer/cuda:torch_cuda_allocator",
        "//src/fastertransformer/kernels:kernels",
        ":trt_plugins",
        "@havenask//aios/autil:string_helper",
        "@local_config_cuda//cuda:cuda_headers",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
)
