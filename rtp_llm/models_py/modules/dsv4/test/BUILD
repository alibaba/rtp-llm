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
        deps = _DSV4_TEST_DEPS,
        tags = ["manual", "legacy_fixture"],
    )
    for name in ["test_cp_via_concat_meta", "test_cp_indexer_seq_total"]
]

[
    dsv4_py_test(
        name = name,
        deps = _MODEL_TEST_DEPS + [":dsv4_kernel_perf_utils"],
        tags = ["manual", "perf"],
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

test_suite(
    name = "dsv4_cuda13_unit",
    tests = [":" + name for name in _CP_HELPER_TESTS + _MODEL_CONTRACT_TESTS + _GPU_TESTS] + _MOVED_CONTRACT_TESTS + _PAIRED_GPU_TESTS,
)

# The ARM lane covers every runnable ordinary target. x86 additionally checks
# kernels, capture, profiling, and JIT specialization on its CUDA13 runtime.
test_suite(
    name = "dsv4_cuda13_cross_arch",
    tests = [":" + name for name in _GPU_TESTS] + [
        ":test_dsv4_kernel_jit_warmup",
        ":test_layer_profiler_ranges",
        ":test_hc_impl",
    ] + [name + "_x86" for name in _PAIRED_GPU_TESTS],
)
