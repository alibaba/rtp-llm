load(":defs.bzl", "dsv4_py_test", "dsv4_sm100_py_test", "dsv4_sm120_py_test")

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

# test_prefill_workspace and test_mtp_hidden_buffer hung silently in CI for
# 1.5-2h+ while holding four cards each, starving the pool; their normal
# runtimes are 11.5s and 16.6s. They stay in the lists above so the test_suites
# keep covering them, but are generated outside the loops below so they can
# carry an explicit "moderate" (300s) bound. The gpu_lock wrapper's runtime
# watchdog is the independent enforcement, and it also reports the node and
# cards and exits with a distinctive code.
_KNOWN_HANG_TESTS = [
    "test_prefill_workspace",
    "test_mtp_hidden_buffer",
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
    if name not in _KNOWN_HANG_TESTS
]

dsv4_py_test(
    name = "test_prefill_workspace",
    deps = _DSV4_TEST_DEPS,
    timeout = "moderate",
    env = {"RTP_TEST_WATCHDOG_SECONDS": "300"},
)

# Cache-clear recompile coverage: clearing the Triton device caches forces
# full JIT recompilation of all three CP metadata kernels, which dominates
# wall-clock on the ARM lane. Kept out of the default/PR lanes via the
# manual tag; run explicitly when touching the warmup/specialization
# contract. `heavy` documents the reason for the exclusion.
dsv4_py_test(
    name = "test_cp_metadata_triton_recompile",
    deps = _DSV4_TEST_DEPS,
    tags = [
        "heavy",
        "manual",
    ],
)

[
    dsv4_py_test(
        name = name,
        deps = _MODEL_TEST_DEPS + ([
            "//rtp_llm/models_py:tile_kernels_mhc",
        ] if name == "test_hc_impl" else []) + ([
            "//rtp_llm/models_py:models",
        ] if name == "test_prefill_fast_path" else []),
    )
    for name in _MODEL_CONTRACT_TESTS + _GPU_TESTS
    if name not in _KNOWN_HANG_TESTS
]

dsv4_py_test(
    name = "test_mtp_hidden_buffer",
    deps = _MODEL_TEST_DEPS,
    timeout = "moderate",
    env = {"RTP_TEST_WATCHDOG_SECONDS": "300"},
)

dsv4_sm100_py_test(
    name = "test_prefill_fast_path_cuda",
    extra_srcs = ["test_prefill_fast_path.py"],
    deps = _MODEL_TEST_DEPS + ["//rtp_llm/models_py:models"],
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
    "@//:using_cuda13_sm120_dsv4": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

# Matches plain cuda13 and the stricter cuda13_sm120_dsv4 (superset), so these
# targets run as blocking tests in the fat-build SM120 lanes.
_DSV4_CUDA13_X86 = select({
    "@//:using_cuda13_x86": [],
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
# production functions from source without executing GPU kernels.
_SOURCE_CPU_TESTS = [
    "test_cp_host_staging_cpu",
    "test_head_host_mirror_scalars_cpu",
    "test_prefill_async_head_cpu",
    "test_suffix_gather_max_host_cpu",
    "test_mhc_prenorm_dg_cpu",
]

# These tests evaluate CPU tensors but import the production Triton module.
# Use the Linux CUDA13 import closure while keeping GPUs hidden.
_TRITON_SOURCE_CPU_TESTS = [
    "test_cp_host_staging_cpu",
    "test_head_host_mirror_scalars_cpu",
    "test_prefill_async_head_cpu",
]

[
    py_test(
        name = name,
        srcs = [name + ".py"],
        deps = ["//rtp_llm:torch"] + (
            ["//rtp_llm:numpy", "//rtp_llm:triton"] if name in _TRITON_SOURCE_CPU_TESTS else []
        ),
        data = [
            "//rtp_llm/models_py:prefill_optimization_source_data",
            "//rtp_llm:device_type_source",
            ":cpu_test_support",
        ] + (
            ["test_prefill_async_head_cpu.py"]
            if name == "test_head_host_mirror_scalars_cpu"
            else []
        ),
        env = {"CUDA_VISIBLE_DEVICES": ""},
        target_compatible_with = _DSV4_CUDA13_X86 if name in _TRITON_SOURCE_CPU_TESTS else [],
    )
    for name in _SOURCE_CPU_TESTS
]

py_test(
    name = "test_prefill_integration_policy",
    srcs = ["test_prefill_integration_policy.py"],
    deps = ["//rtp_llm:torch"],
    data = ["//rtp_llm/models_py:prefill_policy_source_data"],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)


# SM120 hardware contracts; the fat cuda13 build runs them on the SM120 pool too.
[
    dsv4_sm120_py_test(
        name = name,
        tags = ["cuda13_sm120_dsv4"],
        deps = _MODEL_TEST_DEPS,
    )
    for name in [
        "test_sm120_small_batch_projections",
        "test_fp4_indexer_contract",
    ]
]

dsv4_sm120_py_test(
    name = "test_sm120_ordered_reduce_scatter",
    gpu_count = 8,
    args = ["-k", "test_producer_and_subgroup_reduction_8rank"],
    tags = ["multi_device"],  # Optional eight-rank extension; excluded from broad UT lanes.
    deps = _MODEL_TEST_DEPS,
    dsv4_profile_only = True,  # Runs under the DSv4 profile multi-GPU job.
)

# Required CP4/EP4 PREFILL baseline (the supported prefill topology).
# The dedicated four-GPU CI job selects this target explicitly; multi_device
# keeps it out of broad UT lanes that exclude multi-device tests.
dsv4_sm120_py_test(
    name = "test_sm120_ordered_reduce_scatter_4rank",
    src = "test_sm120_ordered_reduce_scatter.py",
    gpu_count = 4,
    args = ["-k", "test_producer_and_subgroup_reduction_4rank"],
    tags = ["multi_device"],
    deps = _MODEL_TEST_DEPS,
    dsv4_profile_only = True,  # Runs under the DSv4 profile multi-GPU job.
)

_SM120_CPU_CONTRACT_TESTS = [":test_cpu_test_utils", ":test_prefill_integration_policy"] + [":" + name for name in _SOURCE_CPU_TESTS]

dsv4_sm120_py_test(
    name = "test_aux_hidden_capture_sm120",
    src = "test_aux_hidden_capture.py",
    deps = _MODEL_TEST_DEPS,
)


# SM120 cases split out of the pre-existing main test files; the originals
# are left unchanged.
[
    dsv4_sm120_py_test(
        name = name,
        deps = _MODEL_TEST_DEPS,
    )
    for name in [
        "test_prefill_fast_path_sm120",
        "test_prefill_workspace_sm120",
        "test_aux_hidden_capture_sm120_cases",
        "test_dsv4_kernel_jit_warmup_sm120_cases",
        "test_indexer_topk_backend_sm120_cases",
    ]
]

test_suite(
    name = "dsv4_cuda13_unit",
    tests = [":" + name for name in _CP_HELPER_TESTS + _MODEL_CONTRACT_TESTS + _GPU_TESTS] + _MOVED_CONTRACT_TESTS + _PAIRED_GPU_TESTS,
)

test_suite(
    name = "dsv4_sm120_unit",
    tests = _SM120_CPU_CONTRACT_TESTS + [
        "//rtp_llm/test:start_server_warmup_test_sm120",
        "//rtp_llm/models_py/kernels/cuda/test:ue8m0_layout_policy_test_sm120",
        "//rtp_llm/models_py/kernels/cuda/test:sm120_fp8_quant_invariance_test",
        "//rtp_llm/models_py/modules/factory/fused_moe/tests:test_sm120_strategy_select_sm120",
        "//rtp_llm/cpp/normal_engine/test:dsv4_dp_prefill_policy_test",
        ":test_aux_hidden_capture_sm120",
        ":test_sm120_small_batch_projections",
        ":test_sm120_batch_invariant_linear",
        ":test_sm120_ordered_reduce_scatter",
        ":test_sm120_ordered_reduce_scatter_4rank",
        ":test_fp4_indexer_contract",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_indexer_forward_with_pending_nested_sm120",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_serialized_indexer_tail_cpu",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_sm120_prefill_indices_sm120",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_sm120_sparse_mla_sm120",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_pack_slots_to_paged_sm120",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_sm120_cp_compressor_projection",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_sm120_sparse_mla_hardware",
        "//rtp_llm/models_py/modules/factory/linear/impl/cuda/test:fp8_blockwise_sm120_hardware_test",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_indexer_topk_v3_routing_sm120",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_sm120_indexer_hardware",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_sm120_prefill_indices_hardware",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_sm120_prefill_indices_producer_matrix",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_dsv4_persistent_topk_sm120",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_attention_packed_prefix_cpu",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_backend_contracts",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_dspark_sparse_backend",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_grouped_output_weights",
        "//rtp_llm/models_py/modules/factory/fused_moe/utils/test:fp8_fp4_layer_test",
        "//rtp_llm/models_py/modules/factory/fused_moe/utils/test:test_sm120_decode_combine",
        "//rtp_llm/models_py/modules/factory/fused_moe/utils/test:test_sm120_moe_numerics",
        "//rtp_llm/models_py/modules/factory/fused_moe/utils/test:test_sm120_decode_transport",
        "//rtp_llm/models_py/modules/factory/fused_moe/utils/test:test_sm120_decode_transport_4rank",
        # Sibling coverage (main test sources are left unchanged).
        ":test_prefill_fast_path_sm120",
        ":test_prefill_workspace_sm120",
        ":test_aux_hidden_capture_sm120_cases",
        ":test_dsv4_kernel_jit_warmup_sm120_cases",
        ":test_indexer_topk_backend_sm120_cases",
        "//rtp_llm/test:start_server_warmup_sm120_cases_test",
        "//rtp_llm/test:auto_configure_deepep_sm120_test",
        "//rtp_llm/test:sm120_dsv4_worker_profile_test",
        "//rtp_llm/models_py/utils/test:arch_sm120_test",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_swa_dequant_large_stride_sm120",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_swa_prefill_suffix_pool",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_swa_slot_mapping_sm120",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_attention_cp_prefill_paths_sm120",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_fp8_sparse_attn_decode_op_sm120_cases",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_sparse_decode_backend_sm120",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_indexer_topk_v3_routing_sm120_cases",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_swa_fp8_kv_roundtrip_sm120_cases",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_prefill_mixed_layer_cache_store_sm120",
        "//rtp_llm/models_py/modules/dsv4/fp8/test:test_indexer_fp8_score_sm120",
        "//rtp_llm/models_py/modules/factory/fused_moe/utils/test:fp8_fp4_layer_sm120_test",
        "//rtp_llm/models_py/kernels/cuda/test/sm120:cutlass_fp4_scaled_mm_sm120_test",
        "//rtp_llm/model_loader/test:test_inline_fp8_quant_sm120",
        "@rtp_deps//:update_cuda13_locks_sm120_test",
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

# Batch-invariant SM120 linear projection kernel
# (rtp_llm/models_py/triton_kernels/sm120_batch_invariant_linear.py); the fat
# cuda13 build runs it on the SM120 pool too.
dsv4_sm120_py_test(
    name = "test_sm120_batch_invariant_linear",
    tags = ["cuda13_sm120_dsv4"],
    deps = _MODEL_TEST_DEPS,
)
