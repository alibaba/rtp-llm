load("@arch_config//:arch_select.bzl", "requirement")
load("@pip_gpu_rocm_torch//:requirements.bzl", requirement_gpu_rocm = "requirement")

requirement([
    "triton",
    "torch",
    "numpy",
    "einops",
    "packaging",
])

# flydsl is ROCm-only (only present in pip_gpu_rocm requirements).
# On non-ROCm configs we point to an empty stub to avoid analysis failures.
py_library(
    name = "flydsl",
    visibility = ["//visibility:public"],
    deps = select({
        "@rtp_llm//:using_rocm": [requirement_gpu_rocm("flydsl")],
        "//conditions:default": [],
    }),
)

# pytest is only present in the ROCm lock file. Keep its repository-specific
# label out of non-ROCm dependency graphs so CUDA/CPU analysis remains valid.
py_library(
    name = "pytest_rocm",
    visibility = ["//visibility:public"],
    deps = select({
        "@rtp_llm//:using_rocm": [requirement_gpu_rocm("pytest")],
        "//conditions:default": [],
    }),
)

py_test(
    name = "test_gdn_block_prefill",
    srcs = [
        "test_gdn_block_prefill.py",
    ],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [
        ":einops",
        ":numpy",
        ":packaging",
        ":torch",
        ":triton",
    ],
)

py_test(
    name = "test_gdn_decode",
    srcs = [
        "test_gdn_decode.py",
    ],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [
        ":einops",
        ":numpy",
        ":packaging",
        ":torch",
        ":triton",
    ],
)

py_test(
    name = "test_gdn_block_prefill_rocm",
    srcs = ["test_gdn_block_prefill.py"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
    main = "test_gdn_block_prefill.py",
    tags = ["rocm"],
    visibility = ["//visibility:public"],
    deps = ["//rtp_llm/models_py/triton_kernels:fla"] + [
        ":einops",
        ":numpy",
        ":packaging",
        ":torch",
        ":triton",
    ],
)

py_test(
    name = "test_gdn_decode_rocm",
    srcs = ["test_gdn_decode.py"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
    main = "test_gdn_decode.py",
    tags = ["rocm"],
    visibility = ["//visibility:public"],
    deps = ["//rtp_llm/models_py/triton_kernels:fla"] + [
        ":einops",
        ":numpy",
        ":packaging",
        ":torch",
        ":triton",
    ],
)

py_test(
    name = "test_chunk_prefill",
    srcs = [
        "test_chunk_prefill.py",
    ],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [
        ":einops",
        ":numpy",
        ":packaging",
        ":torch",
        ":triton",
    ],
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
        "test_chunk_prefill.py",
    ],
    exec_properties = {"gpu": "MI308X-ROCM7"},
    main = "test_chunk_prefill.py",
    tags = ["rocm"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [
        ":einops",
        ":numpy",
        ":packaging",
        ":torch",
        ":triton",
    ],
)

py_test(
    name = "test_l2norm",
    srcs = [
        "test_l2norm.py",
    ],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [
        ":einops",
        ":numpy",
        ":packaging",
        ":torch",
        ":triton",
    ],
)

py_test(
    name = "test_l2norm_rocm",
    srcs = [
        "test_l2norm.py",
    ],
    exec_properties = {"gpu": "MI308X-ROCM7"},
    main = "test_l2norm.py",
    tags = ["rocm"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [
        ":einops",
        ":numpy",
        ":packaging",
        ":torch",
        ":triton",
    ],
)

py_test(
    name = "test_flydsl_chunk_gdn_shape_gate",
    srcs = [
        "test_flydsl_chunk_gdn_shape_gate.py",
    ],
    exec_properties = {"gpu": "MI308X-ROCM7"},
    tags = ["rocm"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [
        ":einops",
        ":flydsl",
        ":packaging",
        ":torch",
        ":triton",
    ],
)

# Numeric + cache-byte equivalence for the FlyDSL Chunk-GDN direct-store path,
# pinned to ROCm (MI308X) because the FlyDSL megakernel only runs on AMD.
py_test(
    name = "test_flydsl_chunk_gdn_cache_store",
    srcs = [
        "test_flydsl_chunk_gdn_cache_store.py",
    ],
    exec_properties = {"gpu": "MI308X-ROCM7"},
    tags = ["rocm"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [
        ":einops",
        ":flydsl",
        ":numpy",
        ":packaging",
        ":torch",
        ":triton",
    ],
)

py_test(
    name = "test_aiter_flydsl_gdn_prefill",
    srcs = ["test_aiter_flydsl_gdn_prefill.py"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
    main = "test_aiter_flydsl_gdn_prefill.py",
    tags = ["rocm"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [
        ":einops",
        ":flydsl",
        ":numpy",
        ":packaging",
        ":pytest_rocm",
        ":torch",
        ":triton",
    ],
)

py_test(
    name = "test_aiter_flydsl_gdn_decode",
    srcs = [
        "test_aiter_flydsl_gdn_decode.py",
    ],
    exec_properties = {"gpu": "H20_CU13"},
    tags = ["H20_CU13"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [
        ":einops",
        ":packaging",
        ":torch",
        ":triton",
    ],
)

py_test(
    name = "test_aiter_flydsl_gdn_decode_rocm",
    srcs = [
        "test_aiter_flydsl_gdn_decode.py",
    ],
    exec_properties = {"gpu": "MI308X-ROCM7"},
    main = "test_aiter_flydsl_gdn_decode.py",
    tags = ["rocm"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [
        ":einops",
        ":packaging",
        ":torch",
        ":triton",
    ],
)
