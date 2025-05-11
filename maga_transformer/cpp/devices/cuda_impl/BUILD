load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "cuda_register", "trt_plugins", "flashinfer_deps", "flashmla_deps")

cuda_register()
trt_plugins()
flashinfer_deps()
flashmla_deps()

config_setting(
    name = "enable_deep_ep",
    values = {"copt": "-DENABLE_DEEP_EP=1"},
)

config_setting(
    name = "use_accl_ep",
    values = {
        "copt": "-DUSE_ACCL_EP=1",
    },
)

cc_library(
    name = "deep_ep_buffer",
    hdrs = [
        "DeepEPDefs.h",
        "DeepEPBuffer.h",
    ],
    srcs = [
        "DeepEPBuffer.cc",
    ],
    deps = [
        "//maga_transformer/cpp/devices:devices_base",
        "//maga_transformer/cpp/devices:device_utils",
        "//maga_transformer/cpp/core:torch_cuda_allocator",
        "//maga_transformer/cpp/core:torch_event",
        "//:th_utils",
    ] + select({
        ":use_accl_ep": [
            "//3rdparty/accl_ep:accl_ep",
        ],
        "//conditions:default": ["@deep_ep//:deep_ep",],
    }),
)

cc_library(
    name = "gpu_base",
    hdrs = [
        "CudaDevice.h",
        "CudaFlashInfer.h",
        "CudaXqa.h",
    ],
    srcs = [
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
        "CudaLoraLinear.cc",
        "CudaDeepEPLLFfnLayer.cc",
        "CudaDeepEPFfnLayer.cc",
        "CudaFlashInfer.cc",
        "CudaXqa.cc",
    ],
    deps = [
        "//maga_transformer/cpp/devices:devices_base",
        "//maga_transformer/cpp/devices:device_utils",
        "//maga_transformer/cpp/devices:devices_base_impl",
        "//maga_transformer/cpp/cuda:cuda",
        "//maga_transformer/cpp/cuda:allocator_cuda",
        "//maga_transformer/cpp/core:torch_cuda_allocator",
        "//maga_transformer/cpp/core:torch_event",
        "//maga_transformer/cpp/kernels:kernels",
        "//maga_transformer/cpp:utils",
        "//maga_transformer/cpp/deep_gemm:deep_gemm_plugins_impl",
        "//:th_utils",
        ":trt_plugins",
        "@havenask//aios/autil:string_helper",
        "@local_config_cuda//cuda:cuda_headers",
        "//maga_transformer/cpp/disaggregate/cache_store:cache_store",
        ":flashinfer",
        "//3rdparty/flashinfer:flashinfer",
        ":flashmla",
    ] + select({
        ":enable_deep_ep": [
            ":deep_ep_buffer",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
)

cc_library(
    name = "gpu_register",
    srcs = [
        "CudaDeviceRegister.cc",
    ],
    deps = [
        ":gpu_base",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
)

cc_library(
    name = "cuda_impl",
    srcs = [
        "CudaActOp.cc",
        "CudaAddBiasOp.cc",
        "CudaEmbeddingLookup.cc",
        "CudaMlaQKVGemm.cc",
        "CudaLayernorm.cc",
        "CudaOps.cc",
        "CudaQuantizeOp.cc",
        "CudaSampleOp.cc",
        "CudaSoftmaxOp.cc",
        "CudaBeamSearchOp.cc",
        "CudaFP8Moe.cc",
    ],
    deps = [
        ":cuda_register",
        "//maga_transformer/cpp/devices/torch_impl:torch_beam_search_op_impl",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
)
