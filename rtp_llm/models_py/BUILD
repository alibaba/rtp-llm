load("@arch_config//:arch_select.bzl", "requirement")

py_library(
    name = "utils",
    srcs = glob([
        "utils/*.py",
    ]),
    visibility = ["//visibility:public"],
)

# flashinfer-python ships in the cuda12 and cuda13 locks
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
    name = "tile_kernels_mhc",
    srcs = glob([
        "3rdparty/__init__.py",
        "3rdparty/tile_kernels/*.py",
        "3rdparty/tile_kernels/**/*.py",
    ]),
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
        ":tile_kernels_mhc",
        "//rtp_llm:warmup",
        "//rtp_llm/models_py/triton_kernels:triton_kernels",
        "//rtp_llm/models_py/distributed:deepep_wrapper",
        "//rtp_llm/models_py/distributed:moriep_wrapper",
        "//rtp_llm/models_py/distributed:collective_torch",
        "//rtp_llm/models_py/distributed:user_buffers",
    ] + select({
        # cuda13 locks ship flashinfer-python; deep-ep only in the x86 lock.
        "@//:using_cuda13_x86": flashinfer + deep,
        "@//:using_cuda13_arm": flashinfer + ["deep_gemm"],
        "@//:using_cuda12": flashinfer + deep,
        "@//:using_ppu": deep,
        "@//:using_rocm": [
            "//rtp_llm/models_py/modules/base/rocm:rocm_base",
            "//rtp_llm/models_py/modules/base/rocm:trt_allreduce",
        ],
        "//conditions:default": [],
    }) + select({
        # cuda13 locks currently ship flash-mla (+tilelang on arm; no fast-hadamard).
        # cuda12_arm (kept from main) has no branch here: its lock ships none of these.
        "@//:using_cuda13_x86": ["flash-mla"],
        "@//:using_cuda13_arm": ["flash-mla", "tilelang"],
        "@//:using_cuda12_9_x86": flashmla,
        "//conditions:default": [],
    }) + select({
        "@//:using_cuda13_x86": rtp_kernel,
        "@//:using_cuda13_arm": rtp_kernel,
        "@//:using_cuda12_9_x86": rtp_kernel,
        "@//:using_cuda12_arm": rtp_kernel,
        "//conditions:default": [],
    }) + select({
        # flash_attn / flash_attn_3: deferred rebuild for torch 2.11+cu130.
        "@//:using_cuda13_x86": [],
        "@//:using_cuda13_arm": [],
        "@//:using_cuda12_9_x86": flashattn,
        "@//:using_cuda12_arm": flashattn,
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

py_library(
    name = "dsv4_fused_kernel_sources",
    srcs = [
        "modules/dsv4/_fused_inv_rope_fp8_quant_triton.py",
        "modules/dsv4/_fused_rmsnorm_rope_triton.py",
        "modules/dsv4/_rope_only_triton.py",
        "modules/dsv4/moe/_mega_input_pack_triton.py",
        "modules/dsv4/moe/_mega_se_gate_pack_triton.py",
        "modules/dsv4/moe/_mega_se_input_pack_triton.py",
    ],
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
    name = "new_weight_loader",
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
    name = "models",
    srcs = glob([
        "model_desc/*.py",
        "model_desc/cuda_graph/*.py",
        "speculative/*.py",
    ]),
    deps = [
        ":utils",
        ":modules",
        ":kernels",
        ":configs",
        "//rtp_llm:warmup",
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
