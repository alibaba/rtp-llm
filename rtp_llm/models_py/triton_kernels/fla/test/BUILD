load("@arch_config//:arch_select.bzl", "requirement")
load("@pip_gpu_rocm_torch//:requirements.bzl", requirement_gpu_rocm = "requirement")

requirement(["triton", "torch", "numpy", "einops", "packaging"])

# flydsl is ROCm-only (only present in pip_gpu_rocm requirements).
# On non-ROCm configs we point to an empty stub to avoid analysis failures.
py_library(
    name = "flydsl",
    deps = select({
        "@rtp_llm//:using_rocm": [requirement_gpu_rocm("flydsl")],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_gdn_block_prefill",
    srcs = [
        "test_gdn_block_prefill.py"
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [":triton", ":torch", ":numpy", ":einops", ":packaging"],
    visibility = ["//visibility:public"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "test_gdn_decode",
    srcs = [
        "test_gdn_decode.py"
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [":triton", ":torch", ":numpy", ":einops", ":packaging"],
    visibility = ["//visibility:public"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "test_gdn_block_prefill_rocm",
    srcs = ["test_gdn_block_prefill.py"],
    main = "test_gdn_block_prefill.py",
    deps = ["//rtp_llm/models_py/triton_kernels:fla"] + [":triton", ":torch", ":numpy", ":einops", ":packaging"],
    visibility = ["//visibility:public"],
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test(
    name = "test_gdn_decode_rocm",
    srcs = ["test_gdn_decode.py"],
    main = "test_gdn_decode.py",
    deps = ["//rtp_llm/models_py/triton_kernels:fla"] + [":triton", ":torch", ":numpy", ":einops", ":packaging"],
    visibility = ["//visibility:public"],
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test(
    name = "test_chunk_prefill",
    srcs = [
        "test_chunk_prefill.py"
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [":triton", ":torch", ":numpy", ":einops", ":packaging"],
    visibility = ["//visibility:public"],
    exec_properties = {'gpu':'A10'},
)

# Reuse test_chunk_prefill on AMD to verify chunk-gdn Triton path on ROCm.
# NOTE: The Gluon fwd_h kernel uses CDNA4-specific instructions (e.g.
# v_mfma_f32_32x32x16_f16, ds_read_b128 with k_width=8) that are only
# available on gfx950 (MI355X). It will NOT work on gfx942 (MI308X) —
# attempting to run on gfx942 will fail with illegal-instruction errors.
# This CI target runs on MI308X and therefore only exercises the Triton
# fallback path. When CI gains gfx950 machines, the Gluon kernel will be
# auto-dispatched via _is_gluon_beneficial.
py_test(
    name = "test_chunk_prefill_amd",
    srcs = [
        "test_chunk_prefill.py"
    ],
    main = "test_chunk_prefill.py",
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [":triton", ":torch", ":numpy", ":einops", ":packaging"],
    visibility = ["//visibility:public"],
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test(
    name = "test_l2norm",
    srcs = [
        "test_l2norm.py"
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [":triton", ":torch", ":numpy", ":einops", ":packaging"],
    visibility = ["//visibility:public"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "test_l2norm_rocm",
    srcs = [
        "test_l2norm.py"
    ],
    main = "test_l2norm.py",
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [":triton", ":torch", ":numpy", ":einops", ":packaging"],
    visibility = ["//visibility:public"],
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test(
    name = "test_flydsl_chunk_gdn_shape_gate",
    srcs = [
        "test_flydsl_chunk_gdn_shape_gate.py"
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [":torch", ":triton", ":einops", ":packaging", ":flydsl"],
    visibility = ["//visibility:public"],
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

# Numeric + cache-byte equivalence for the FlyDSL Chunk-GDN direct-store path,
# pinned to ROCm (MI308X) because the FlyDSL megakernel only runs on AMD.
py_test(
    name = "test_flydsl_chunk_gdn_cache_store",
    srcs = [
        "test_flydsl_chunk_gdn_cache_store.py"
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [":triton", ":torch", ":numpy", ":einops", ":packaging", ":flydsl"],
    visibility = ["//visibility:public"],
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

# Diagnostic only: actual 397B LINEAR geometry, BF16 state feedback across tokens.
py_test(
    name = "qwen35_mtp_bf16_state_test",
    srcs = ["test_qwen35_mtp_bf16_state.py"],
    main = "test_qwen35_mtp_bf16_state.py",
    deps = [
        "//rtp_llm/models_py:utils",
        "//rtp_llm/models_py/triton_kernels:fla",
        "//rtp_llm/models_py/triton_kernels:causal_conv1d",
    ] + [":triton", ":torch", ":numpy", ":einops", ":packaging"],
    visibility = ["//visibility:public"],
    tags = ["manual"],
    exec_properties = {"gpu": "SM100_ARM"},
)

py_test(
    name = "test_exact_qk_norm",
    srcs = ["test_exact_qk_norm.py"],
    deps = ["//rtp_llm:rtp_llm_lib"],
    tags = ["manual", "cuda13", "sm10x"],
    exec_properties = {"gpu": "SM100_ARM"},
)

py_test(
    name = "test_gdn_prefill_exact",
    srcs = ["test_gdn_prefill_exact.py"],
    deps = ["//rtp_llm:rtp_llm_lib"],
    tags = ["manual", "cuda13", "sm10x"],
    exec_properties = {"gpu": "SM100_ARM"},
)

py_test(
    name = "test_gdn_prefill_optimization",
    srcs = ["test_gdn_prefill_optimization.py"],
    deps = ["//rtp_llm:rtp_llm_lib"],
    tags = ["manual", "cuda13", "sm10x"],
    exec_properties = {"gpu": "SM100_ARM"},
)

py_test(
    name = "test_flashinfer_prefill_adapter",
    srcs = ["test_flashinfer_prefill_adapter.py"],
    deps = ["//rtp_llm:rtp_llm_lib"],
    tags = ["manual", "cuda13", "sm10x"],
    exec_properties = {"gpu": "SM100_ARM"},
)

py_test(
    name = "test_gdn_prefill_dispatch",
    srcs = ["test_gdn_prefill_dispatch.py"],
    deps = ["//rtp_llm:rtp_llm_lib"],
    tags = ["manual", "cuda13", "sm10x"],
    exec_properties = {"gpu": "SM100_ARM"},
)

py_test(
    name = "test_gdn_conv_qkv_fusion",
    srcs = ["test_gdn_conv_qkv_fusion.py"],
    deps = ["//rtp_llm:rtp_llm_lib"],
    env = {"GPU_COUNT": "1"},
    tags = ["manual", "cuda13", "sm10x"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)
