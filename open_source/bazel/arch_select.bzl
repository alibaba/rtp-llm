# to wrapper target relate with different system config
load("@pip_gpu_torch//:requirements.bzl", requirement_gpu="requirement")
load("@pip_gpu_cuda12_torch//:requirements.bzl", requirement_gpu_cuda12="requirement")

def requirement(names):
    for name in names:
        native.py_library(
            name = name,
            deps = select({
                "//:use_cuda12": [requirement_gpu_cuda12(name)],
                "//conditions:default": [requirement_gpu(name)],
            }),
            visibility = ["//visibility:public"],
        )

def th_transformer_so():
    native.alias(
        name = "th_transformer_so",
        actual = select({
            "//:use_cuda12": "//:th_transformer",
            "//conditions:default": "//:th_transformer"
        })
    )

    native.genrule(
        name = "libth_transformer_so",
        srcs = [":th_transformer_so"],
        outs = [
            "libth_transformer.so",
        ],
        cmd = " && ".join(["cp $(SRCS) $(@D)"])
    )

def whl_deps():
    return select({
        "//:use_cuda12": ["torch==2.1.0+cu121"],
        "//conditions:default": ["torch==2.1.0+cu118"],
    })

def cutlass_kernels_interface():
    native.alias(
        name = "cutlass_kernels_interface",
        actual = select({
            "//:use_cuda12": "//src/fastertransformer/cutlass:cutlass_kernels_impl",
            "//conditions:default": "//src/fastertransformer/cutlass:cutlass_kernels_impl",
        })
    )

    native.alias(
        name = "cutlass_headers_interface",
        actual = select({
            "//:use_cuda12": "//src/fastertransformer/cutlass:cutlass_headers",
            "//conditions:default": "//src/fastertransformer/cutlass:cutlass_headers",
        })
    )

