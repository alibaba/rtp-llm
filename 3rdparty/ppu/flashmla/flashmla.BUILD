load("@//:def.bzl", "copts", "cuda_copts")
load("@rules_cc//examples:experimental_cc_shared_library.bzl", "cc_shared_library")
load("@arch_config//:arch_select.bzl", "torch_deps")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

common_opts = [
    "-DFLASH_MLA_STANDALONE_BUILD",
]

flash_mla_cuda_copts = common_opts + copts() + cuda_copts() + if_cuda([
    "-D_USE_MATH_DEFINES",
    "-Wno-deprecated-declarations",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-nvcc_options=expt-relaxed-constexpr",
    "-nvcc_options=expt-extended-lambda",
    "-nvcc_options=use_fast_math",
    # "-nvcc_options=ptxas-options=-v,--register-usage-level=10",
    "-nvcc_options=ptxas-options=--register-usage-level=10", # disable the verbose flag of ptxas for clear compile log
    "-mllvm",
    "-ppu-max-vreg-count=256",
    "-mllvm",
    "-ppu-sink-matrix-addr=true",
    "-mllvm",
    "-ppu-max-alloca-byte-size=320",
    "-mllvm",
    "-ppu-sink-async-addr=true",
    "-mllvm",
    "-ppu-sink-load-addr=true",
    "-mllvm",
    "-ppu-sink-store-addr=true",
    "-mllvm",
    "-ppu-alloca-half-ldst-simplify=true",
    "-DUSE_PPU",
    "-DUSE_AIU=1",
    "-DACOMPUTE_VERSION=10000",
])

flash_mla_headers = glob([
    "csrc/*.h",
    "csrc/*.hpp",
])

cc_library(
    name = "flashmla_hdrs",
    hdrs = flash_mla_headers,
    deps = [
        "@cutlass3_ppu_flashmla//:headers",
        "@cutlass3_ppu_flashmla//:cutlass_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    includes = ["csrc"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "flashmla_interface",
    hdrs = [
        "csrc/flashmla.h",
    ],
    strip_include_prefix = "csrc",
    include_prefix = "flashmla",
    visibility = ["//visibility:public"],
)

# PPU: flashmla is header-only. The CUDA kernels (flash_fwd_*.cu, previously the
# flashmla_cu target) and flash_api.cpp torch-op registrations are intentionally
# NOT compiled: PPU MLA runs through the flash_mla pip wheel (Python: `from
# flash_mla import flash_mla_with_kvcache, get_mla_metadata`), no RTP-LLM C++
# calls the kernels, and flash_fwd_split_*.cu require an arch newer than ppu0015.
# Keeping only the interface headers satisfies the gpu_base link dep without
# compiling dead code.
cc_library(
    name = "flashmla",
    deps = [
        ":flashmla_interface",
    ],
    copts = copts() + common_opts,
    visibility = ["//visibility:public"],
)
