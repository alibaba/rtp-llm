load("@pip_dsv4_test//:requirements.bzl", test_requirement = "requirement")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_v41_sm90_prepare_meta",
    srcs = ["test_v41_sm90_prepare_meta.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_v41_sm90_prepare_meta.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_v41_sm90_decode_candidate",
    srcs = ["test_v41_sm90_decode_candidate.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_v41_sm90_decode_candidate.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_v41_sm90_indexer_q",
    srcs = ["test_v41_sm90_indexer_q.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_v41_sm90_indexer_q.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_v41_sm90_topk",
    srcs = [
        "test_v41_sm90_topk.py",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_topk_v3.py",
        "//rtp_llm/test/pytest:pytest_main.py",
    ],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_v41_sm90_topk.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_v41_sm90_output_projection",
    srcs = ["test_v41_sm90_output_projection.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_v41_sm90_output_projection.py)"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:jit_cache_env",
        test_requirement("pytest"),
    ],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

py_test(
    name = "test_v41_sm90_v4_cache",
    srcs = glob(["*.py"]) + ["//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_v41_sm90_v4_cache.py)"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py:dsv41_sm90_attention_reference",
        "//rtp_llm:jit_cache_env",
        test_requirement("pytest"),
    ],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

py_test(
    name = "test_v41_sm90_paged_decode",
    srcs = ["test_v41_sm90_paged_decode.py", "test_v41_flash_mla_cache_format.py", "//rtp_llm/models_py/modules/dsv41/test:test_flash_mla_heads.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_v41_sm90_paged_decode.py)", "$(location //rtp_llm/models_py/modules/dsv41/test:test_flash_mla_heads.py)"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py:dsv41_sm90_attention_reference",
        "//rtp_llm:jit_cache_env",
        test_requirement("pytest"),
    ],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

py_test(
    name = "test_v41_sm90_attention",
    srcs = ["test_v41_sm90_attention.py", "test_v41_flash_mla_cache_format.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_v41_sm90_attention.py)"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py:dsv41_sm90_attention_reference",
        "//rtp_llm:jit_cache_env",
        test_requirement("pytest"),
    ],
    tags = ["dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

# Reuse the existing exact TopK contract on Hopper, including tie overflow,
# candidate-mask holes and repeat/graph determinism. Legacy backend benchmarks
# in that file remain in the existing SM100 target.
py_test(
    name = "test_v41_sm90_topk_contract",
    srcs = ["test_v41_prefill_topk.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = [
        "--cuda-devices", "1", "--cuda-major", "9",
        "$(location :test_v41_prefill_topk.py)::V41PrefillTopKCUDA::test_mixed_exact_boundary_matches_oracle_in_all_dispatch_paths",
        "$(location :test_v41_prefill_topk.py)::V41PrefillTopKCUDA::test_strided_short_long_and_clamped_bounds",
        "$(location :test_v41_prefill_topk.py)::V41PrefillTopKCUDA::test_nan_inf_ties_and_all_masked_rows",
        "$(location :test_v41_prefill_topk.py)::V41PrefillTopKCUDA::test_repeated_selection_is_bitwise_stable",
    ],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:jit_cache_env",
        test_requirement("pytest"),
    ],
    tags = ["dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

_TESTS = [
    "test_v41_attention_pool_warmup",
    "test_attention_v41",
    "test_attention_v41_cp_raw_groups",
    "test_attention_v41_cuda_graph",
    "test_attention_v41_prefill_producer",
    "test_attention_v41_prefill_selection",
    "test_attention_v41_request_rows",
    "test_attention_v41_sparse_selection",
    "test_v41_attention_jit_warmup",
    "test_v41_batched_producer",
    "test_v41_candidate_topk",
    "test_v41_decode_attention_fusion",
    "test_v41_decode_device_metadata",
    "test_v41_decode_global",
    "test_v41_decode_indexer",
    "test_v41_decode_topk",
    "test_v41_decode_preprocessing",
    "test_v41_decode_qkv_integration",
    "test_v41_deepselect",
    "test_v41_fp4_codec",
    "test_v41_flash_mla_cache_format",
    "test_v41_fp4_transport",
    "test_v41_global_overlap",
    "test_v41_host_swa_metadata",
    "test_v41_grouped_gemm",
    "test_v41_grouped_prefill_score",
    "test_v41_indexer_q_fusion",
    "test_v41_mega_attention",
    "test_v41_merged_qb",
    "test_v41_mega_wiring",
    "test_v41_joint_pool",
    "test_v41_owned_pool",
    "test_v41_cp_async",
    "test_v41_norm_quant",
    "test_v41_prefill_candidates",
    "test_v41_prefill_deepselect",
    "test_v41_prefill_global",
    "test_v41_prefill_index_plan",
    "test_v41_prefill_indexer",
    "test_v41_prefill_kv_workspace",
    "test_v41_prefill_meta_cache",
    "test_v41_prefill_meta_cache_cuda",
    "test_v41_prefill_metadata",
    "test_v41_prefill_pools",
    "test_v41_prefill_q_memory",
    "test_v41_prefill_topk",
    "test_v41_producer_metadata",
    "test_v41_qkv_rope_cache",
    "test_v41_selection_oracle",
    "test_v41_sparse_prefill_indexer",
    "test_v41_sparse_decode",
    "test_v41_swa_codec",
    "test_v41_swa_fresh",
    "test_v41_wo_a_quant",
]


# The selection chain supports both SM100 ARM and SM100/SM103 x86.
_X86_TESTS = [
    "test_v41_attention_jit_warmup",
    "test_v41_owned_pool",
    "test_v41_cp_async",
    "test_v41_prefill_kv_workspace",
    "test_v41_host_swa_metadata",
    "test_v41_sparse_decode",
    "test_v41_decode_topk",
    "test_v41_decode_indexer",
    "test_v41_prefill_deepselect",
    "test_v41_deepselect",
    "test_v41_decode_attention_fusion",
    "test_v41_decode_qkv_integration",
]
py_test(
    name = "test_v41_sm90_indexer_multiquery",
    srcs = ["test_v41_sm90_indexer_multiquery.py", "test_v41_sm90_indexer_tuning.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_v41_sm90_indexer_multiquery.py)"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:jit_cache_env",
        test_requirement("pytest"),
    ],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1", "PYTHONNOUSERSITE": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

py_test(
    name = "test_v41_sm90_indexer_tuning",
    srcs = ["test_v41_sm90_indexer_tuning.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_v41_sm90_indexer_tuning.py)"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:jit_cache_env",
        test_requirement("pytest"),
    ],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1", "PYTHONNOUSERSITE": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

py_test(
    name = "test_v41_sm90_indexer",
    srcs = ["test_v41_sm90_indexer.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_v41_sm90_indexer.py)"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:jit_cache_env",
        test_requirement("pytest"),
    ],
    tags = ["dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

[
    py_test(
        name = name,
        srcs = glob(["*.py"]) + ["//rtp_llm/test/pytest:pytest_main.py", "//rtp_llm/models_py/modules/dsv4/fp8/test:test_prefill_common_metadata_cuda.py"],
        main = "//rtp_llm/test/pytest:pytest_main.py",
        args = ["--cuda-devices", "1", "$(location :" + name + ".py)"],
        deps = [
            "//rtp_llm:testlib",
            "//rtp_llm/models_py:modules",
            "//rtp_llm/models_py:tile_kernels_mhc",
            "//rtp_llm:jit_cache_env",
            test_requirement("pytest"),
        ],
        data = [
            "//rtp_llm/models_py/bindings/cuda/kernels:topk_v3.cu",
            "//rtp_llm/models_py/bindings/cuda/kernels:topk_v3.cuh",
        ] if name == "test_v41_attention_jit_warmup" else [],
        tags = ["dsv41_cuda13", "SM100_ARM_CU13"] + (["L20_CU13"] if name in _X86_TESTS else []),
        exec_properties = select({
            "@//:using_cuda13_x86": {"gpu": "L20_CU13", "gpu_count": "1"},
            "//conditions:default": {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
        }),
        target_compatible_with = select({
            "@//:using_cuda13_arm": [],
            "@//:using_cuda13_x86": [] if name in _X86_TESTS else ["@platforms//:incompatible"],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
        env = {"RTP_V41_PREFILL_DEEPSELECT_GPU_TEST": "1"} if name == "test_v41_prefill_deepselect" else {},
        legacy_create_init = 0,
    )
    for name in _TESTS
]

# CPU dispatch contracts compiled against the CUDA13 x86 module closure.
[
    py_test(
        name = name + "_sm90",
        srcs = [name + ".py", "//rtp_llm/test/pytest:pytest_main.py"],
        main = "//rtp_llm/test/pytest:pytest_main.py",
        args = ["--cuda-devices", "0", "$(location :" + name + ".py)::" + case],
        deps = [
            "//rtp_llm:testlib",
            "//rtp_llm/models_py:modules",
            "//rtp_llm:jit_cache_env",
            test_requirement("pytest"),
        ],
        tags = ["dsv41_cuda13", "H20_CU13"],
        env = {"GPU_COUNT": "0", "PYTHONNOUSERSITE": "1"},
        target_compatible_with = select({
            "@//:using_cuda13_x86": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
        legacy_create_init = 0,
    )
    for name, case in [
        ("test_attention_v41_sparse_selection", "SparseSelectionIntegrationCPU"),
        ("test_v41_prefill_indexer", "V41PrefillIndexerCPU"),
        ("test_v41_decode_global", "V41DecodeGlobalCPU"),
    ]
]
