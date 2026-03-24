load("@//:def.bzl", "copts", "cuda_copts")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch", "if_cuda")
load("@//bazel:arch_select.bzl", "torch_deps")
load("@//3rdparty/flashinfer:def.bzl", "sub_lib")

common_copts = [
    '-DFLASHINFER_ENABLE_BF16',
    '-DFLASHINFER_ENABLE_F16',
    '-DFLASHINFER_ENABLE_FP8_E4M3',
]

sm90_cuda_copts = copts() + cuda_default_copts_without_arch() + if_cuda(["-nvcc_options=objdir-as-tempdir"]) + [
    '--cuda-include-ptx=sm_90', '--cuda-gpu-arch=sm_90',
    '--cuda-include-ptx=sm_90a', '--cuda-gpu-arch=sm_90a',
] + common_copts

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
    name = "aot_build_utils_generate_sm90",
    srcs = ['aot_build_utils/generate_sm90.py'],
    deps = [":aot_build_utils"],
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
    name = "generated_sm90",
    tools = [":aot_build_utils_generate_sm90"],
    cmd = "loc=$(locations @flashinfer_cpp//:aot_build_utils_generate_sm90); loc=$${loc%/*};loc=$${loc%/*}; PYTHONPATH=$$loc /opt/conda310/bin/python -m aot_build_utils.generate_sm90 --enable_f16 true --use_fp16_qk_reductions false --enable_bf16 true --mask_modes 1 --path $(RULEDIR) --head_dims 64,64 128,128 --pos_encoding_modes 0",
    outs = [
        'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32_sm90.cu',
        'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32_sm90.cu',
        'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32_sm90.cu',
        'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32_sm90.cu',
        'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32_sm90.cu',
        'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32_sm90.cu',
        'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32_sm90.cu',
        'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32_sm90.cu',
        'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_sm90.cu',
        'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_sm90.cu',
        'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_sm90.cu',
        'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_sm90.cu',
    ],
    tags=["local"],
)

genrule(
    name = "generated",
    tools = [":aot_build_utils_generate"],
    cmd = "loc=$(locations @flashinfer_cpp//:aot_build_utils_generate); loc=$${loc%/*};loc=$${loc%/*}; PYTHONPATH=$$loc /opt/conda310/bin/python -m aot_build_utils.generate --enable_fp8_e4m3 true --enable_fp8_e5m2 true --enable_f16 true --use_fp16_qk_reductions false --enable_bf16 true --mask_modes 1 --path $(RULEDIR) --head_dims 64 128 256 --pos_encoding_modes 0 ",
    outs = ['batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_e4m3_dtypekv_e4m3_dtypeout_e4m3_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_e5m2_dtypekv_e5m2_dtypeout_e5m2_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_e4m3_dtypekv_e4m3_dtypeout_e4m3_idtype_i32.cu', 'batch_paged_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_e5m2_dtypekv_e5m2_dtypeout_e5m2_idtype_i32.cu', 'batch_paged_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_e4m3_dtypekv_e4m3_dtypeout_e4m3_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_e5m2_dtypekv_e5m2_dtypeout_e5m2_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu',
    'aot_default_additional_params.h',
    'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_e4m3_dtypekv_e4m3_dtypeout_e4m3.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_e5m2_dtypekv_e5m2_dtypeout_e5m2.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_e4m3_dtypeout_f16.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_e5m2_dtypeout_f16.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_f16_dtypeout_f16.cu', 'single_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16.cu', 'single_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16.cu', 'single_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16.cu', 'single_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_e4m3_dtypekv_e4m3_dtypeout_e4m3.cu', 'single_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_e5m2_dtypekv_e5m2_dtypeout_e5m2.cu', 'single_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_f16_dtypekv_e4m3_dtypeout_f16.cu', 'single_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_f16_dtypekv_e5m2_dtypeout_f16.cu', 'single_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_f16_dtypekv_f16_dtypeout_f16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_e4m3_dtypekv_e4m3_dtypeout_e4m3.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_e5m2_dtypekv_e5m2_dtypeout_e5m2.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_e4m3_dtypeout_f16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_e5m2_dtypeout_f16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_f16_dtypeout_f16.cu', 'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16.cu', 'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16.cu', 'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16.cu', 'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16.cu', 'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16.cu', 'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16.cu', 'single_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16.cu', 'single_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16.cu', 'single_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16.cu', 'single_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16.cu', 'single_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16.cu', 'single_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16.cu'],
    tags=["local"],
)

batch_paged_decode = ['batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu','batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_e4m3_dtypekv_e4m3_dtypeout_e4m3_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_e5m2_dtypekv_e5m2_dtypeout_e5m2_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_e4m3_dtypekv_e4m3_dtypeout_e4m3_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_e5m2_dtypekv_e5m2_dtypeout_e5m2_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu']

batch_paged_decode_256 = ['batch_paged_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_e4m3_dtypekv_e4m3_dtypeout_e4m3_idtype_i32.cu', 'batch_paged_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_e5m2_dtypekv_e5m2_dtypeout_e5m2_idtype_i32.cu', 'batch_paged_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu']

batch_paged_prefill = ['batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu']

batch_paged_prefill_256 = ['batch_paged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu']

batch_ragged_prefill = ['batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu']

batch_ragged_prefill_256 = ['batch_ragged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu']

single_decode = ['single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_e4m3_dtypekv_e4m3_dtypeout_e4m3.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_e5m2_dtypekv_e5m2_dtypeout_e5m2.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_e4m3_dtypeout_f16.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_e5m2_dtypeout_f16.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_f16_dtypeout_f16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_e4m3_dtypekv_e4m3_dtypeout_e4m3.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_e5m2_dtypekv_e5m2_dtypeout_e5m2.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_e4m3_dtypeout_f16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_e5m2_dtypeout_f16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_f16_dtypeout_f16.cu']

single_decode_256 = ['single_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16.cu', 'single_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16.cu', 'single_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16.cu', 'single_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_e4m3_dtypekv_e4m3_dtypeout_e4m3.cu', 'single_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_e5m2_dtypekv_e5m2_dtypeout_e5m2.cu', 'single_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_f16_dtypekv_e4m3_dtypeout_f16.cu', 'single_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_f16_dtypekv_e5m2_dtypeout_f16.cu', 'single_decode_head_qk_256_head_vo_256_posenc_0_dtypeq_f16_dtypekv_f16_dtypeout_f16.cu']

single_prefill = ['single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16.cu', 'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16.cu', 'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16.cu', 'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16.cu', 'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16.cu', 'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16.cu']

single_prefill_256 = ['single_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16.cu', 'single_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16.cu', 'single_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16.cu', 'single_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16.cu', 'single_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16.cu', 'single_prefill_head_qk_256_head_vo_256_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16.cu']

sub_lib('flashinfer_batch_paged_prefill', batch_paged_prefill, cuda_copts() + common_copts)
sub_lib('flashinfer_batch_paged_prefill_256', batch_paged_prefill_256, cuda_copts() + common_copts)
sub_lib('flashinfer_batch_paged_decode', batch_paged_decode, cuda_copts() + common_copts)
sub_lib('flashinfer_batch_paged_decode_256', batch_paged_decode_256, cuda_copts() + common_copts)
sub_lib('flashinfer_batch_ragged_prefill', batch_ragged_prefill, cuda_copts() + common_copts)
sub_lib('flashinfer_batch_ragged_prefill_256', batch_ragged_prefill_256, cuda_copts() + common_copts)
sub_lib('flashinfer_single_decode', single_prefill, cuda_copts() + common_copts)
sub_lib('flashinfer_single_decode_256', single_prefill_256, cuda_copts() + common_copts)
sub_lib('flashinfer_single_prefill', single_decode, cuda_copts() + common_copts)
sub_lib('flashinfer_single_prefill_256', single_decode_256, cuda_copts() + common_copts)
sub_lib('flashinfer_sm90', [":generated_sm90"], sm90_cuda_copts)

cc_library(
    name = "flashinfer_mla",
    srcs = [
        "csrc/batch_mla_plan.cu",
        "csrc/batch_mla_run.cu",
    ] + glob([
        "csrc/*.h",
        "csrc/*.inc",
    ]) + [
        "flashinfer_single_decode",
        "flashinfer_single_decode_256",
        "flashinfer_single_prefill",
        "flashinfer_single_prefill_256",
        "flashinfer_batch_paged_prefill",
        "flashinfer_batch_paged_prefill_256",
        "flashinfer_batch_paged_decode",
        "flashinfer_batch_paged_decode_256",
        "flashinfer_batch_ragged_prefill",
        "flashinfer_batch_ragged_prefill_256",
        "flashinfer_sm90"
    ],
    implementation_deps = [
        ":dispatch",
        ":flashinfer_hdrs",
        ":aot_default_additional_params",
    ],
    # mla not support fp8 on current commit
    copts = cuda_copts() + [
        '-DFLASHINFER_ENABLE_BF16',
        '-DFLASHINFER_ENABLE_F16',
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "flashinfer",
    srcs = [
        "csrc/bmm_fp8.cu",
        "csrc/cascade.cu",
        "csrc/group_gemm.cu",
        "csrc/norm.cu",
        "csrc/page.cu",
        "csrc/quantization.cu",
        "csrc/rope.cu",
        "csrc/sampling.cu",
        "csrc/renorm.cu",
        "csrc/activation.cu",
        "csrc/batch_decode.cu",
        "csrc/batch_prefill.cu",
        "csrc/single_decode.cu",
        "csrc/single_prefill.cu",
        "csrc/group_gemm_sm90.cu",
        "csrc/single_prefill_sm90.cu",
        "csrc/batch_prefill_sm90.cu",
    ] + glob([
        "csrc/*.h",
        "csrc/*.inc",
    ]) + [
        "flashinfer_single_decode",
        "flashinfer_single_decode_256",
        "flashinfer_single_prefill",
        "flashinfer_single_prefill_256",
        "flashinfer_batch_paged_prefill",
        "flashinfer_batch_paged_prefill_256",
        "flashinfer_batch_paged_decode",
        "flashinfer_batch_paged_decode_256",
        "flashinfer_batch_ragged_prefill",
        "flashinfer_batch_ragged_prefill_256",
        "flashinfer_sm90"
    ],
    implementation_deps = [
        ":flashinfer_mla",
        ":dispatch",
        ":flashinfer_hdrs",
        ":aot_default_additional_params",
    ],
    copts = cuda_copts() + common_copts,
    visibility = ["//visibility:public"],
)
