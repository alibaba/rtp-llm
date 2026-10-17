load(":defs.bzl", "dsv4_py_test", "dsv4_sm100_py_test")

package(default_visibility = ["//visibility:public"])

_DSV4_TEST_DEPS = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
    "//rtp_llm:testlib",
]

_MODEL_TEST_DEPS = [
    "//rtp_llm/models_py:modules",
    "//rtp_llm:testlib",
]

# CPU math and mocked transport tests still need the CUDA13 Python import
# closure. Run them alongside GPU tests, with one isolated process per file.
_CP_HELPER_TESTS = [
    "test_cp_context_build",
    "test_cp_topk_idxs_align",
    "test_cp_pool_block_gather",
    "test_pool_reader",
    "test_cp_kv_allgather_restore",
    "test_cp_slot_mapping",
    "test_indexer_cp_assembler",
    "test_cp_workspace_combine_topk",
    "test_cp_attention_merge",
    "test_cp_attention_shard",
    "test_cp_async",
    "test_prefill_workspace",
]

_MODEL_CONTRACT_TESTS = [
    "test_mtp_hidden_buffer",
    "test_dspark_cuda_graph_contract",
    "test_dspark_runtime_config",
    "test_aux_hidden_capture",
    "test_prefill_fast_path",
    "test_hc_impl",
    "test_dsv4_kernel_jit_warmup",
    "test_layer_profiler_ranges",
]

_GPU_TESTS = [
    "test_cp_metadata_triton",
    "test_indexer_topk_backend",
    "test_fused_inv_rope_fp8_quant",
    "test_fused_rmsnorm_rope",
    "test_fused_rmsnorm_fp8_quant",
    "test_rope_only",
]

[
    dsv4_py_test(name = name, deps = _DSV4_TEST_DEPS)
    for name in _CP_HELPER_TESTS
]

[
    dsv4_py_test(
        name = name,
        deps = _MODEL_TEST_DEPS + ([
            "//rtp_llm/models_py:tile_kernels_mhc",
        ] if name == "test_hc_impl" else []),
    )
    for name in _MODEL_CONTRACT_TESTS + _GPU_TESTS
]

dsv4_sm100_py_test(
    name = "test_prefill_fast_path_cuda",
    extra_srcs = ["test_prefill_fast_path.py"],
    deps = _MODEL_TEST_DEPS,
)

_PERF_TESTS = [
    "test_fused_inv_rope_fp8_quant_perf",
    "test_fused_rmsnorm_rope_perf",
    "test_rope_only_perf",
    "test_cp_restore_perf",
    "test_dsv4_tc_simt_overlap_perf",
]

# These historical fixtures compare duplicated local formulas instead of
# calling the production builders; they are not regression gate evidence.
[
    dsv4_py_test(
        name = name,
        tags = [
            "legacy_fixture",
            "manual",
        ],
        deps = _DSV4_TEST_DEPS,
    )
    for name in [
        "test_cp_via_concat_meta",
        "test_cp_indexer_seq_total",
    ]
]

[
    dsv4_py_test(
        name = name,
        tags = [
            "manual",
            "perf",
        ],
        deps = _MODEL_TEST_DEPS + [":dsv4_kernel_perf_utils"],
    )
    for name in _PERF_TESTS
]

py_library(
    name = "dsv4_test_utils",
    srcs = ["dsv4_test_utils.py"],
    deps = _DSV4_TEST_DEPS,
)

py_library(
    name = "dsv4_kernel_perf_utils",
    srcs = ["dsv4_kernel_perf_utils.py"],
)

# MoE coverage follows the production modules moved into the shared factory.
_MOVED_CONTRACT_TESTS = [
    "//rtp_llm/models_py/modules/factory/fused_moe/tests:test_strategy_select",
    "//rtp_llm/models_py/modules/factory/fused_moe/utils/test:test_chunked_moe",
]

_PAIRED_GPU_TESTS = [
    ":test_prefill_fast_path_cuda",
    "//rtp_llm/models_py/modules/factory/fused_moe/utils/test:test_shared_expert_executor",
    "//rtp_llm/models_py/modules/factory/fused_moe/utils/test:test_mega_moe_input_packer",
    "//rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/executors/test:test_grouped_fp4_strategy_cuda_graph",
    "//rtp_llm/models_py/triton_kernels/moe/test:test_mega_moe_se_input_pack",
    "//rtp_llm/models_py/triton_kernels/moe/test:test_mega_moe_gate_pack_cuda13",
]

_DSV4_SM120_ONLY = select({
    "@//:using_cuda13_sm120": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

filegroup(
    name = "cpu_test_support",
    srcs = ["cpu_test_utils.py"],
)

py_test(
    name = "test_cpu_test_utils",
    srcs = ["test_cpu_test_utils.py"],
    deps = ["//rtp_llm:torch"],
    data = [":cpu_test_support"],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

# Source-level CPU contracts for the SM120/CP prefill path. They load the
# production functions from source and never import GPU modules.
_SOURCE_CPU_TESTS = [
    "test_cp_deferred_verify_cpu",
    "test_cp_host_staging_cpu",
    "test_head_host_mirror_scalars_cpu",
    "test_prefill_async_head_cpu",
    "test_suffix_gather_max_host_cpu",
    "test_mhc_prenorm_dg_cpu",
    "test_prefill_diagnostics_cpu",
]

[
    py_test(
        name = name,
        srcs = [name + ".py"],
        deps = ["//rtp_llm:torch"],
        data = [
            "//rtp_llm/models_py:prefill_optimization_source_data",
            ":cpu_test_support",
        ] + (["//rtp_llm:device_type_source"] if name == "test_mhc_prenorm_dg_cpu" else []),
        env = {"CUDA_VISIBLE_DEVICES": ""},
    )
    for name in _SOURCE_CPU_TESTS
]

py_test(
    name = "test_prefill_integration_policy",
    srcs = ["test_prefill_integration_policy.py"],
    data = ["//rtp_llm/models_py:prefill_policy_source_data"],
)

py_test(
    name = "test_prefill_async_head_stream",
    srcs = ["test_prefill_async_head_stream.py"],
    deps = ["//rtp_llm:torch"],
    data = ["//rtp_llm/models_py:prefill_optimization_source_data"],
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
    target_compatible_with = _DSV4_SM120_ONLY,
)

# Exact-SM120 hardware tests (RTX PRO 5000); built only by cuda13_sm120.
[
    py_test(
        name = name,
        srcs = [name + ".py"],
        deps = _MODEL_TEST_DEPS,
        tags = ["RTX_5000_PRO_CU13"],
        exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
        target_compatible_with = _DSV4_SM120_ONLY,
    )
    for name in [
        "test_sm120_small_batch_projections",
        "test_sm120_mhc_columns",
        "test_fp4_indexer_contract",
        "test_fp4_speculative_state",
    ]
]

py_test(
    name = "test_fp4_native_indexer",
    srcs = ["test_fp4_native_indexer.py", "test_fp4_indexer_contract.py"],
    main = "test_fp4_native_indexer.py",
    deps = _MODEL_TEST_DEPS,
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
    target_compatible_with = _DSV4_SM120_ONLY,
)

_SM120_CPU_CONTRACT_TESTS = [":test_cpu_test_utils", ":test_prefill_integration_policy"] + [":" + name for name in _SOURCE_CPU_TESTS]

test_suite(
    name = "dsv4_cuda13_unit",
    tests = [":" + name for name in _CP_HELPER_TESTS + _MODEL_CONTRACT_TESTS + _GPU_TESTS] + _MOVED_CONTRACT_TESTS + _PAIRED_GPU_TESTS + _SM120_CPU_CONTRACT_TESTS,
)

test_suite(
    name = "dsv4_sm120_unit",
    tests = _SM120_CPU_CONTRACT_TESTS + [
        ":test_prefill_async_head_stream",
        ":test_sm120_small_batch_projections",
        ":test_sm120_mhc_columns",
        ":test_fp4_indexer_contract",
        ":test_fp4_native_indexer",
        ":test_fp4_speculative_state",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_indexer_forward_with_pending_nested_sm120",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_sm120_prefill_indices_sm120",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_sm120_sparse_mla_sm120",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_sm120_cp_compressor_projection",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_sm120_sparse_mla_hardware",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_sm120_indexer_hardware",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_sm120_prefill_indices_hardware",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_sm120_fused_indexer_gather",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_sm120_prefill_indices_producer_matrix",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_sm120_prefill_split",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_dsv4_persistent_topk_sm120",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_compact_cp_contract_cpu",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_compact_cp_runtime_cpu",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_compact_cp_assert_fusion_cpu",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_attention_prefix_host_mirror_cpu",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_compact_cp_plan_cache_cpu",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_indexer_tail_stream_cpu",
        "//rtp_llm/models_py/modules/factory/fused_moe/utils/test:fp8_fp4_layer_test",
        "//rtp_llm/models_py/modules/factory/fused_moe/utils/test:test_sm120_dispatch_quant_pack",
    ],
    target_compatible_with = _DSV4_SM120_ONLY,
)

# The ARM lane covers every runnable ordinary target. x86 additionally checks
# kernels, capture, profiling, and JIT specialization on its CUDA13 runtime.
test_suite(
    name = "dsv4_cuda13_cross_arch",
    tests = [":" + name + "_x86" for name in _GPU_TESTS] + [
        ":test_dsv4_kernel_jit_warmup_x86",
        ":test_hc_impl_x86",
        ":test_layer_profiler_ranges_x86",
    ] + [name + "_x86" for name in _PAIRED_GPU_TESTS],
)

dsv4_sm100_py_test(
    name = "test_mega_inv_rope_fp8_quant",
    deps = _DSV4_TEST_DEPS,
)
