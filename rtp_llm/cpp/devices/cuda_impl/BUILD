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
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/devices:device_utils",
        "//rtp_llm/cpp/core:torch_cuda_allocator",
        "//rtp_llm/cpp/core:torch_event",
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
    ] + select({
        "@//:using_cuda12": [
            "CudaXqa.h",
        ],
        "//conditions:default": [],
    }),
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
    ] + select({
        "@//:using_cuda12": [
            "CudaXqa.cc",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/devices:device_utils",
        "//rtp_llm/cpp/devices:devices_base_impl",
        "//rtp_llm/cpp/cuda:cuda",
        "//rtp_llm/cpp/cuda:allocator_cuda",
        "//rtp_llm/cpp/core:torch_cuda_allocator",
        "//rtp_llm/cpp/core:torch_event",
        "//rtp_llm/cpp/kernels:kernels",
        "//rtp_llm/cpp:utils",
        "//rtp_llm/cpp/deep_gemm:deep_gemm_plugins_impl",
        "//:th_utils",
        ":trt_plugins",
        "@havenask//aios/autil:string_helper",
        "@local_config_cuda//cuda:cuda_headers",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store",
        ":flashinfer",
        "//3rdparty/flashinfer:flashinfer",
        ":flashmla",
    ] + select({
        ":enable_deep_ep": [
            ":deep_ep_buffer",
        ],
        "//conditions:default": [],
    }) + select({
        "@//:using_cuda12": [
            "//3rdparty/xqa:xqa",
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
        "//rtp_llm/cpp/devices/torch_impl:torch_beam_search_op_impl",
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
    alwayslink = 1,
)
