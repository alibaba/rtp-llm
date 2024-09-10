load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "cuda_base", "trt_plugins")

cuda_base()
trt_plugins()

cc_library(
    name = "gpu_base",
    hdrs = glob([
        "CudaDevice.h",
    ]),
    srcs = glob([
        "CudaDevice.cc",
        "CudaAttentionOp.cc",
        "CudaFfnLayer.cc",
        "CudaGemmOp.cc",
        "CudaGroupGemmOp.cc",
        "CudaLoraLinearWithActOp.cc",
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
    include_prefix = "src/fastertransformer/devices/",
)
cc_library(
    name = "cuda_impl",
    srcs = glob([
        "CudaActOp.cc",
        "CudaAddBiasOp.cc",
        "CudaEmbeddingLookup.cc",
        "CudaLayernorm.cc",
        "CudaOps.cc",
        "CudaQuantizeOp.cc",
        "CudaSampleOp.cc",
        "CudaSoftmaxOp.cc",
    ]),
    deps = [
        ":cuda_base",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
)
