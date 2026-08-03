# CUDA 13 variant of flashinfer.BUILD. It uses the CUDA 13 CUTLASS repository,
# applies the CUDA 13 compatibility patches, and builds only the utility
# kernels still registered by RTP-LLM. Keep shared target names aligned with
# flashinfer.BUILD so arch_select can switch repositories transparently.
load("@//:def.bzl", "cuda_copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

common_copts = [
    "-DFLASHINFER_ENABLE_BF16",
    "-DFLASHINFER_ENABLE_F16",
    "-DFLASHINFER_ENABLE_FP8_E4M3",
]

cc_library(
    name = "dispatch",
    hdrs = ["dispatch.inc"],
    include_prefix = "generated",
)

cc_library(
    name = "flashinfer_hdrs",
    hdrs = glob([
        "include/flashinfer/**/*.cuh",
        "include/flashinfer/**/*.h",
    ]) + [
        ":dispatch",
    ],
    deps = [
        "@cutlass3.6_cu13//:cutlass",
        "@cutlass3.6_cu13//:cutlass_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "@local_config_cuda//cuda:cublas_headers",
        "@local_config_cuda//cuda:cublas",
        "@local_config_cuda//cuda:cublasLt",
    ] + torch_deps(),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)

py_library(
    name = "aot_build_utils",
    srcs = [
        "aot_build_utils/__init__.py",
        "aot_build_utils/literal_map.py",
    ],
)

py_library(
    name = "dispatch_generate_py",
    srcs = ["aot_build_utils/generate_dispatch_inc.py"],
    deps = [":aot_build_utils"],
)

genrule(
    name = "generate_dispatch",
    outs = ["dispatch.inc"],
    # This interpreter path and dispatch command intentionally mirror the
    # generate_dispatch rule in flashinfer.BUILD for the current CUDA images.
    # Change both rules together until a shared Python-toolchain macro replaces
    # the duplicated genrules.
    cmd = "loc=$(locations @flashinfer_cpp_cu13//:dispatch_generate_py); loc=$${loc%/*}; loc=$${loc%/*}; PYTHONPATH=$$loc /opt/conda310/bin/python -m aot_build_utils.generate_dispatch_inc --use_fp16_qk_reductions false --mask_modes 1 --path $(RULEDIR)/dispatch.inc --head_dims_sm90 64,64 128,128 --head_dims 64 128 256 --pos_encoding_modes 0",
    tags = ["local"],
    tools = [":dispatch_generate_py"],
)

# C++ FlashInfer attention was replaced by the Python backend in 5312895a25.
# Keep only the utility kernels still registered by the CUDA bindings.
# At the pinned upstream commit these four sources do not consume csrc/*.inc or
# aot_default_additional_params.h, so the CUDA 13 source/header set is smaller.
cc_library(
    name = "flashinfer",
    srcs = [
        "csrc/norm.cu",
        "csrc/sampling.cu",
        "csrc/renorm.cu",
        "csrc/activation.cu",
    ] + glob(["csrc/*.h"]),
    implementation_deps = [
        ":dispatch",
        ":flashinfer_hdrs",
    ],
    copts = cuda_copts() + common_copts,
    visibility = ["//visibility:public"],
)
