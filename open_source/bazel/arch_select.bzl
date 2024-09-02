# to wrapper target relate with different system config
load("@pip_cpu_torch//:requirements.bzl", requirement_cpu="requirement")
load("@pip_gpu_torch//:requirements.bzl", requirement_gpu="requirement")
load("@pip_gpu_cuda12_torch//:requirements.bzl", requirement_gpu_cuda12="requirement")
load("@pip_gpu_rocm_torch//:requirements.bzl", requirement_gpu_rocm="requirement")
load("@pip_cpu_arm_torch//:requirements.bzl", requirement_cpu_arm="requirement")

def requirement(names):
    for name in names:
        native.py_library(
            name = name,
            deps = select({
                "//:using_cuda12": [requirement_gpu_cuda12(name)],
                "//:using_cuda11": [requirement_gpu(name)],
                "//:using_rocm": [requirement_gpu_rocm(name)],
                "//:using_arm": [requirement_cpu_arm(name)],
                "//conditions:default": [requirement_cpu(name)],
            }),
            visibility = ["//visibility:public"],
        )

def th_transformer_so():
    native.alias(
        name = "th_transformer_so",
        actual = "//:th_transformer",
    )

def embedding_arpc_deps():
    native.alias(
        name = "embedding_arpc_deps",
        actual = "//maga_transformer/cpp/embedding_engine:embedding_engine_arpc_server_impl"
    )

def whl_deps():
    return select({
        "//:using_cuda12": ["torch==2.1.2+cu121"],
        "//:using_cuda11": ["torch==2.1.2+cu118"],
        "//:using_rocm": ["torch==2.1.2"],
        "//conditions:default": ["torch==2.1.2"],
    })

def torch_deps():
    deps = select({
        "@//:using_rocm": [
            "@torch_2.1_py310_rocm//:torch_api",
            "@torch_2.1_py310_rocm//:torch",
            "@torch_2.1_py310_rocm//:torch_libs",
        ],
        "//:using_arm": [
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
        "//:using_cuda": [":libmmha_so", ":libfa_so", ":libfpA_intB_so", ":libint8_gemm_so", ":libmoe_so"],
        "//:using_rocm": [":libmmha_so"],
        "//conditions:default":[],
    })
