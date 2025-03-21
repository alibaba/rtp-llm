load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "cuda_register", "trt_plugins", "flashinfer_deps", "flashmla_deps")

cuda_register()
trt_plugins()
flashinfer_deps()
flashmla_deps()

cc_library(
    name = "gpu_base",
    hdrs = glob([
        "CudaDevice.h",
    ]),
    srcs = glob([
        "CudaDevice.cc",
        "CudaWeights.cc",
        "CudaAttentionOp.cc",
        "CudaFfnLayer.cc",
        "CudaMlaAttentionOp.cc",
        "CudaGemmOp.cc",
        "CudaPrefillAttention.cc",
        "CudaGroupGemmOp.cc",
        "CudaNvtxOp.cc",
        "CudaLoraLinearWithActOp.cc",
    ]),
    deps = [
        "//src/fastertransformer/devices:devices_base",
        "//src/fastertransformer/devices:device_utils",
        "//src/fastertransformer/devices:devices_base_impl",
        "//src/fastertransformer/cuda:cuda",
        "//src/fastertransformer/cuda:allocator_cuda",
        "//src/fastertransformer/core:torch_cuda_allocator",
        "//src/fastertransformer/kernels:kernels",
        "//src/fastertransformer/utils",
        "//:th_utils",
        ":trt_plugins",
        "@havenask//aios/autil:string_helper",
        "@local_config_cuda//cuda:cuda_headers",
        "//maga_transformer/cpp/disaggregate/cache_store:cache_store",
        ":flashinfer",
        "//3rdparty/flashinfer:flashinfer",
        ":flashmla",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
)
cc_library(
    name = "gpu_register",
    srcs = glob([
        "CudaDeviceRegister.cc",
    ]),
    deps = [
        ":gpu_base",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
)
cc_library(
    name = "cuda_impl",
    srcs = glob([
        "CudaActOp.cc",
        "CudaAddBiasOp.cc",
        "CudaEmbeddingLookup.cc",
        "CudaMlaQKVGemm.cc",
        "CudaMlaContextAttention.cc",
        "CudaLayernorm.cc",
        "CudaOps.cc",
        "CudaQuantizeOp.cc",
        "CudaSampleOp.cc",
        "CudaSoftmaxOp.cc",
        "CudaBeamSearchOp.cc",
    ]),
    deps = [
        ":cuda_register",
        "//src/fastertransformer/devices/torch_impl:torch_beam_search_op_impl",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
)
