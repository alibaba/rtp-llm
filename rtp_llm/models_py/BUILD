load("@arch_config//:arch_select.bzl", "requirement")

py_library(
    name = "utils",
    srcs = glob([
        "utils/*.py",
    ]),
    visibility = ["//visibility:public"],
)

# flashinfer-python is only available for CUDA12
flashinfer = ["flashinfer-python"]
deep = ["deep_gemm", "deep_ep"]
flashmla = ["fast-hadamard-transform", "flash-mla", "tilelang"]
rtp_kernel = ["rtp-kernel"]
flashattn = ["flash_attn","flash-attn-3"]
requirement(flashinfer)
requirement(deep)
requirement(flashmla)
requirement(rtp_kernel)
requirement(flashattn)


filegroup(
    name = "cutlass_moe_config",
    srcs = glob(["kernels/cuda/fp8_kernel/cutlass_groupgemm/*"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "rocm_fp8_ptpc_hipb_solutions",
    srcs = [
        "modules/factory/linear/impl/rocm/data/fp8_ptpc_hipb_solutions.json",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "modules",
    srcs = glob([
        "modules/*.py",
        "modules/**/*.py",
    ]),
    data = [
        ":cutlass_moe_config",
        ":rocm_fp8_ptpc_hipb_solutions",
    ],
    deps = [
        ":utils",
        "//rtp_llm/models_py/triton_kernels:triton_kernels",
        "//rtp_llm/models_py/distributed:deepep_wrapper",
        "//rtp_llm/models_py/distributed:moriep_wrapper",
        "//rtp_llm/models_py/distributed:collective_torch",
        "//rtp_llm/models_py/distributed:user_buffers",
    ] + select({
        "@//:using_cuda12": flashinfer + deep,
        "@//:using_ppu": deep,
        "@//:using_rocm": [
            "//rtp_llm/models_py/modules/base/rocm:rocm_base",
            "//rtp_llm/models_py/modules/base/rocm:trt_allreduce",
        ],
        "//conditions:default": [],
    }) + select({
        "@//:using_cuda12_9_x86": flashmla,
        "//conditions:default": [],
    }) + select({
        "@//:using_cuda12_9_x86": rtp_kernel,
        "@//:using_cuda12_arm": rtp_kernel,
        "//conditions:default": [],
    }) + select({
        "@//:using_cuda12_9_x86": flashattn,
        "@//:using_cuda12_arm": flashattn,
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

# Export sitecustomize.py for test targets to fix nvidia-cutlass-dsl import
exports_files(["modules/factory/sitecustomize.py"])

py_library(
    name = "modules_cuda",
    srcs = glob([
        "modules/cuda/*.py",
        "modules/cuda/**/*.py",
    ]),
    visibility = [":__subpackages__"],
)

py_library(
    name = "modules_rocm",
    srcs = glob([
        "modules/rocm/*.py",
        "modules/rocm/**/*.py",
    ]),
    visibility = [":__subpackages__"],
)

py_library(
    name = "kernels",
    srcs = glob([
        "kernels/*.py",
        "kernels/**/*.py",
    ])
)

py_library(
    name = "configs",
    srcs = glob([
        "configs/*.py",
    ]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "new_loader_core",
    srcs = [
        "__init__.py",
        "model_loader.py",
        "module_base.py",
        "registry.py",
        "weight_mapper.py",
    ],
    deps = [
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "new_weight_loader",
    srcs = [
        "model_desc/module_base.py",
    ] + glob([
        "layers/*.py",
        "new_models/*.py",
        "new_models/qwen2/*.py",
        "new_models/qwen2_moe/*.py",
        "new_models/qwen2_vl/*.py",
        "new_models/qwen3/*.py",
        "new_models/qwen3_moe/*.py",
        "new_models/qwen3_next/*.py",
        "quant_methods/*.py",
    ]),
    deps = [
        ":new_loader_core",
        ":kernels",
        ":modules",
        "//rtp_llm:_ft_pickler",
        "//rtp_llm:config",
        "//rtp_llm:device",
        "//rtp_llm:ops",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm:utils",
        "//rtp_llm/models_py/distributed:collective_torch",
        "//rtp_llm/models_py/triton_kernels:triton_kernels",
    ],
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer",
        "//:th_transformer_config",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "qwen2_vl_vision_loader",
    srcs = [
        "new_models/__init__.py",
        "new_models/qwen2_vl/__init__.py",
        "new_models/qwen2_vl/vision.py",
    ],
    deps = [
        ":new_loader_core",
        "//rtp_llm:torch",
        "//rtp_llm:utils",
    ] + select({
        "@//:using_cuda12_9_x86": flashattn,
        "@//:using_cuda12_arm": flashattn,
        "//conditions:default": [],
    }),
    visibility = ["//rtp_llm:__subpackages__"],
)

py_library(
    name = "models",
    srcs = glob([
        "model_desc/*.py",
        "model_desc/cuda_graph/*.py"
    ]),
    deps = [
        ":utils",
        ":modules",
        ":kernels",
        ":configs",
        ":new_weight_loader",
        "//rtp_llm/model_loader:loader",
    ] + select({
        "@//:using_cuda12": [
            ":modules_cuda",
        ],
        "@//:using_rocm": [
            ":modules_rocm"
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)
