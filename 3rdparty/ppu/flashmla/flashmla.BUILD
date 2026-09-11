load("@//:def.bzl", "copts")

common_opts = ["-DFLASH_MLA_STANDALONE_BUILD"]

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
