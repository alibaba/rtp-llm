load("@//:def.bzl", "copts", "cuda_copts")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch", "if_cuda")
load("@arch_config//:arch_select.bzl", "torch_deps")
load("@//3rdparty/ppu/flashinfer:def.bzl", "sub_lib")

common_copts = [
    '-DFLASHINFER_ENABLE_BF16',
    '-DFLASHINFER_ENABLE_FP8',
    '-DFLASHINFER_ENABLE_F16',
    '-DUSE_SAIL',
]

cc_library(
    name = "dispatch",
    hdrs = ["dispatch.inc"],
    include_prefix = "generated",
    copts = cuda_copts() + common_copts,
)

cc_library(
    name = "aot_default_additional_params",
    hdrs = ["aot_default_additional_params.h"],
    copts = cuda_copts() + common_copts,
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
        "@cutlass3_ppu_flashinfer//:headers",
        "@cutlass3_ppu_flashinfer//:cutlass_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "@local_config_cuda//cuda:cublas_headers",
        "@local_config_cuda//cuda:cublas",
        "@local_config_cuda//cuda:cublasLt",
    ] + torch_deps(),
    strip_include_prefix = "include",
    copts = cuda_copts() + common_copts,
    visibility = ["//visibility:public"],
)

py_library(
    name = "aot_build_utils",
    srcs = glob(['aot_build_utils/*.py']),
)

py_binary(
    name = "aot_generate",
    srcs = ["@//3rdparty/ppu/flashinfer:aot_generate.py"],
    main = "@//3rdparty/ppu/flashinfer:aot_generate.py",
    deps = [":aot_build_utils"],
)

genrule(
    name = "generate_dispatch",
    tools = [":aot_generate"],
    cmd = "$(location :aot_generate) aot_build_utils.generate_dispatch_inc --use_fp16_qk_reductions false --mask_modes 1 --path $(RULEDIR)/dispatch.inc --head_dims_sm90 64,64 128,128 --head_dims 64 128 --pos_encoding_modes 0",
    outs =[
        "dispatch.inc",
    ],
    tags=["local"],
)

genrule(
    name = "generated",
    tools = [":aot_generate"],
    cmd = "$(location :aot_generate) aot_build_utils.generate --enable_fp8_e4m3 true --enable_fp8_e5m2 true --enable_f16 true --use_fp16_qk_reductions false --enable_bf16 true --mask_modes 1 --path $(RULEDIR) --head_dims 64 128 --pos_encoding_modes 0 ",
    outs = ['batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_e4m3_dtypekv_e4m3_dtypeout_e4m3_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_e5m2_dtypekv_e5m2_dtypeout_e5m2_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_e4m3_dtypekv_e4m3_dtypeout_e4m3_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_e5m2_dtypekv_e5m2_dtypeout_e5m2_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu',
    'aot_default_additional_params.h',
    'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_e4m3_dtypekv_e4m3_dtypeout_e4m3.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_e5m2_dtypekv_e5m2_dtypeout_e5m2.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_e4m3_dtypeout_f16.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_e5m2_dtypeout_f16.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_f16_dtypeout_f16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_e4m3_dtypekv_e4m3_dtypeout_e4m3.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_e5m2_dtypekv_e5m2_dtypeout_e5m2.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_e4m3_dtypeout_f16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_e5m2_dtypeout_f16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_f16_dtypeout_f16.cu', 'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16.cu', 'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16.cu', 'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16.cu', 'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16.cu', 'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16.cu', 'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16.cu'],
    tags=["local"],
)

batch_paged_decode = ['batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu','batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_e4m3_dtypekv_e4m3_dtypeout_e4m3_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_e5m2_dtypekv_e5m2_dtypeout_e5m2_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_e4m3_dtypekv_e4m3_dtypeout_e4m3_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_e5m2_dtypekv_e5m2_dtypeout_e5m2_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_paged_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu']

batch_paged_prefill = ['batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_paged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu']

batch_ragged_prefill = ['batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16_idtype_i32.cu', 'batch_ragged_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16_idtype_i32.cu']

single_decode = ['single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_e4m3_dtypekv_e4m3_dtypeout_e4m3.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_e5m2_dtypekv_e5m2_dtypeout_e5m2.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_e4m3_dtypeout_f16.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_e5m2_dtypeout_f16.cu', 'single_decode_head_qk_128_head_vo_128_posenc_0_dtypeq_f16_dtypekv_f16_dtypeout_f16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_e4m3_dtypekv_e4m3_dtypeout_e4m3.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_e5m2_dtypekv_e5m2_dtypeout_e5m2.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_e4m3_dtypeout_f16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_e5m2_dtypeout_f16.cu', 'single_decode_head_qk_64_head_vo_64_posenc_0_dtypeq_f16_dtypekv_f16_dtypeout_f16.cu']

single_prefill = ['single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16.cu', 'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16.cu', 'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16.cu', 'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16.cu', 'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16.cu', 'single_prefill_head_qk_128_head_vo_128_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_bf16_dtypeout_bf16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e4m3_dtypeout_bf16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_bf16_dtypekv_e5m2_dtypeout_bf16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e4m3_dtypeout_f16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_e5m2_dtypeout_f16.cu', 'single_prefill_head_qk_64_head_vo_64_posenc_0_fp16qkred_0_mask_1_dtypeq_f16_dtypekv_f16_dtypeout_f16.cu']

sub_lib('flashinfer_batch_paged_prefill', batch_paged_prefill, cuda_copts() + common_copts)
sub_lib('flashinfer_batch_paged_decode', batch_paged_decode, cuda_copts() + common_copts)
sub_lib('flashinfer_batch_ragged_prefill', batch_ragged_prefill, cuda_copts() + common_copts)
sub_lib('flashinfer_single_prefill', single_prefill, cuda_copts() + common_copts)
sub_lib('flashinfer_single_decode', single_decode, cuda_copts() + common_copts)
sub_lib('flashinfer_sm90', [], cuda_copts() + common_copts)

cc_library(
    name = "flashinfer",
    srcs = [
        # only kernels declared in flashinfer.h and called from RTP-LLM C++
        "csrc/norm.cu",
        "csrc/sampling.cu",
        "csrc/renorm.cu",
        "csrc/activation.cu",
    ] + glob([
        "csrc/*.h",
        "csrc/*.inc",
    ]),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
    ],
    implementation_deps = [
        ":dispatch",
        ":flashinfer_hdrs",
        ":aot_default_additional_params",
    ],
    copts = cuda_copts() + common_copts,
    visibility = ["//visibility:public"],
)
