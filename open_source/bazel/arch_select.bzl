# to wrapper target relate with different system config
load("@pip_cpu_torch//:requirements.bzl", requirement_cpu="requirement")
load("@pip_arm_torch//:requirements.bzl", requirement_arm="requirement")
load("@pip_gpu_torch//:requirements.bzl", requirement_gpu="requirement")
load("@pip_gpu_cuda12_torch//:requirements.bzl", requirement_gpu_cuda12="requirement")
load("@pip_gpu_rocm_torch//:requirements.bzl", requirement_gpu_rocm="requirement")

def requirement(names):
    for name in names:
        native.py_library(
            name = name,
            deps = select({
                "@//:using_cuda12": [requirement_gpu_cuda12(name)],
                "@//:using_cuda11": [requirement_gpu(name)],
                "@//:using_rocm": [requirement_gpu_rocm(name)],
                "@//:using_arm": [requirement_arm(name)],
                "//conditions:default": [requirement_cpu(name)],
            }),
            visibility = ["//visibility:public"],
        )

def cache_store_deps():
    native.alias(
        name = "cache_store_arch_select_impl",
        actual = "//maga_transformer/cpp/disaggregate/cache_store:cache_store_base_impl"
    )

def th_transformer_so():
    native.alias(
        name = "th_transformer",
        actual = "//:th_transformer",
    )

def embedding_arpc_deps():
    native.alias(
        name = "embedding_arpc_deps",
        actual = "//maga_transformer/cpp/embedding_engine:embedding_engine_arpc_server_impl"
    )

def subscribe_deps():
    native.alias(
        name = "subscribe_deps",
        actual = "//maga_transformer/cpp/disaggregate/load_balancer/subscribe:subscribe_service_impl"
    )

def whl_deps():
    return select({
        "@//:using_cuda12": ["torch==2.1.2+cu121"],
        "@//:using_cuda11": ["torch==2.1.2+cu118"],
        "@//:using_rocm": ["torch==2.1.2"],
        "//conditions:default": ["torch==2.1.2"],
    })

def platform_deps():
    return select({
        "@//:using_arm": [],
        "//conditions:default": ["decord==0.6.0"],
    })

def torch_deps():
    deps = select({
        "@//:using_rocm": [
            "@torch_rocm//:torch_api",
            "@torch_rocm//:torch",
            "@torch_rocm//:torch_libs",
        ],
        "@//:using_arm": [
            "@torch_2.3_py310_cpu_aarch64//:torch_api",
            "@torch_2.3_py310_cpu_aarch64//:torch",
            "@torch_2.3_py310_cpu_aarch64//:torch_libs",
        ],
        "@//:using_cuda": [
            "@torch_2.1_py310_cuda//:torch_api",
            "@torch_2.1_py310_cuda//:torch",
            "@torch_2.1_py310_cuda//:torch_libs",
        ],
        "//conditions:default": [
            "@torch_2.1_py310_cpu//:torch_api",
            "@torch_2.1_py310_cpu//:torch",
            "@torch_2.1_py310_cpu//:torch_libs",
        ]
    })
    return deps

def cutlass_kernels_interface():
    native.alias(
        name = "cutlass_kernels_interface",
        actual = "//src/fastertransformer/cutlass:cutlass_kernels_impl",
    )

    native.alias(
        name = "cutlass_headers_interface",
        actual = "//src/fastertransformer/cutlass:cutlass_headers",
    )

def fa_deps():
    native.alias(
        name = "fa",
        actual = "@flash_attention//:fa"
    )

    native.alias(
        name = "fa_hdrs",
        actual = "@flash_attention//:fa_hdrs",
    )

def kernel_so_deps():
    return select({
        "@//:using_cuda": [":libmmha1_so", ":libmmha2_so", ":libdmmha_so", ":libfa_so", ":libfpA_intB_so", ":libint8_gemm_so", ":libmoe_so", ":libflashinfer_0_so", ":libflashinfer_1_so", ":libflashinfer_2_so"],
        "@//:using_rocm": [":libmmha1_so", ":libmmha2_so", ":libdmmha_so"],
        "//conditions:default":[],
    })

def arpc_deps():
    native.cc_library(
        name = ""
    )

def trt_plugins():
    native.alias(
        name = "trt_plugins",
        actual = select({
            "@//:using_cuda12": "//src/fastertransformer/trt_plugins:trt_plugins",
            "//conditions:default": "//src/fastertransformer/trt_plugins:trt_plugins",
        })
    )

def cuda_register():
    native.alias(
        name = "cuda_register",
        actual = select({
            "//conditions:default": "//src/fastertransformer/devices/cuda_impl:gpu_register",
        })
    )

def libacext_so():
    native.filegroup(
        name = "libacext_so",
        srcs = [],
        visibility = ["//visibility:public"],
    )

def triton_deps(names):
    return select({
        "//conditions:default": [],
    })
