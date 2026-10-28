load("@//:def.bzl", "cuda_copts")
load("@rtp_llm//bazel:defs.bzl", "torch_deps")

common_copts = [
    '-DFLASHINFER_ENABLE_BF16',
    '-DFLASHINFER_ENABLE_F16',
    '-DFLASHINFER_ENABLE_FP8_E4M3',
]

cc_library(
    name = "dispatch",
    hdrs = ["dispatch.inc"],
    include_prefix = "generated",
)

cc_library(
    name = "aot_default_additional_params",
    hdrs = ["aot_default_additional_params.h"],
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
        "@cutlass3.6//:cutlass",
        "@cutlass3.6//:cutlass_utils",
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
    srcs = glob(['aot_build_utils/*.py']),
)

py_library(
    name = "aot_build_utils_generate",
    srcs = ['aot_build_utils/generate.py'],
    deps = [":aot_build_utils"],
)

py_library(
    name = "dispatch_generate_py",
    srcs = ['aot_build_utils/generate_dispatch_inc.py'],
    deps = [":aot_build_utils"],
)

genrule(
    name = "generate_dispatch",
    tools = [":dispatch_generate_py"],
    cmd = "loc=$(locations @flashinfer_cpp//:dispatch_generate_py); loc=$${loc%/*};loc=$${loc%/*}; PYTHONPATH=$$loc /opt/conda310/bin/python -m aot_build_utils.generate_dispatch_inc --use_fp16_qk_reductions false --mask_modes 1 --path $(RULEDIR)/dispatch.inc --head_dims_sm90 64,64 128,128 --head_dims 64 128 256 --pos_encoding_modes 0",
    outs =[
        "dispatch.inc",
    ],
    tags=["local"],
)

genrule(
    name = "generated",
    tools = [":aot_build_utils_generate"],
    cmd = "loc=$(locations @flashinfer_cpp//:aot_build_utils_generate); loc=$${loc%/*};loc=$${loc%/*}; PYTHONPATH=$$loc /opt/conda310/bin/python -m aot_build_utils.generate --enable_fp8_e4m3 true --enable_fp8_e5m2 true --enable_f16 true --use_fp16_qk_reductions false --enable_bf16 true --mask_modes 1 --path $(RULEDIR) --head_dims 64 128 256 --pos_encoding_modes 0 ",
    outs = ['aot_default_additional_params.h'],
    tags=["local"],
)

cc_library(
    name = "flashinfer",
    srcs = [
        "csrc/norm.cu",
        "csrc/sampling.cu",
        "csrc/renorm.cu",
        "csrc/activation.cu",
    ] + glob([
        "csrc/*.h",
        "csrc/*.inc",
    ]),
    implementation_deps = [
        ":dispatch",
        ":flashinfer_hdrs",
        ":aot_default_additional_params",
    ],
    copts = cuda_copts() + common_copts,
    visibility = ["//visibility:public"],
)
