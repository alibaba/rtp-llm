load("//rtp_llm/models_py/modules/dsv4/test:defs.bzl", "DSV4_CUDA13_ONLY", "dsv4_py_test")
load("@arch_config//:arch_select.bzl", "cuda13_test_exec_properties")
load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

_DSV4_FP8_TEST_DEPS = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
    "//rtp_llm:testlib",
]

# Numerical kernels and the associated metadata/pool boundary regressions.
_GPU_TESTS = [
    "test_bf16_gemm_fp32",
    "test_compressor_disable_raw_path",
    "test_compressor_forward_flat_input",
    "test_compressor_fp8_per_token",
    "test_compressor_prepare_metadata_varlen",
    "test_compressor_raw_dispatch",
    "test_compressor_state_pool_cpu",
    "test_compressor_varlen_raw_path",
    "test_decode_swa_write",
    "test_dsv4_persistent_topk",
    "test_topk_v3",
    "test_indexer_topk_v3_routing",
    "test_dsv4_top_k_per_row_prefill",
    "test_fused_compressor_meta_triton",
    "test_ring_write_mask",
    "test_fp8_mqa_prefill_indexer_score",
    "test_fp8_paged_indexer_score",
    "test_indexer_compute_q_flat_input",
    "test_indexer_fp8_prefill_chain",
    "test_indexer_fp8_prefill_meta",
    "test_indexer_fp8_quant_triton",
    "test_indexer_fp8_score",
    "test_indexer_q_fp8_quant",
    "test_swa_combine_topk",
    "test_swa_cp_packed_gather_triton",
    "test_swa_dequant_large_stride",
    "test_swa_fp8_kv_roundtrip",
    "test_swa_prefill_gather_lens",
    "test_swa_prefill_varlen",
    "test_swa_slot_mapping",
    "test_swa_topk_meta_triton",
    "test_workspace_prefill_varlen",
    "test_no_recompile",
    "test_decode_attn_metadata_start_pos",
    "test_flash_mla_decode_kernel_layout",
    "test_fused_decode_meta_comprehensive",
    "test_compressor_start_finish_prefill",
    "test_pool_slot_mapping_split",
    "test_indexer_cp_padded_gather_triton",
    "test_prefill_common_metadata_cuda",
]

# Mock only non-target transport/kernel boundaries; these verify scheduling,
# lifetime, and metadata contracts without loading a full DSv4 checkpoint.
_CONTRACT_TESTS = [
    "test_cp_pool_reader_owned_lens",
    "test_attention_cp_prefill_paths",
    "test_attention_csa_overlap",
    "test_attention_hca_overlap",
    "test_cache_lifecycle",
    "test_output_projection_all_reduce",
    "test_decode_topk_length",
    "test_indexer_forward_with_pending_nested",
    "test_prefill_mixed_layer_cache_store",
    "test_fp8_sparse_attn_decode_op",
    "test_compressor_cp_merged_gather",
]

[
    dsv4_py_test(
        name = name,
        deps = _DSV4_FP8_TEST_DEPS,
        gpu_count = 2 if name == "test_topk_v3" else 1,
        tags = ["multi_device"] if name == "test_topk_v3" else [],
        # Do not let the local integer-arithmetic assertion hide a skipped
        # large-stride CUDA regression when the worker lacks memory.
        args = ["-k", "SwaDequantLargeStrideTest"] if name == "test_swa_dequant_large_stride" else [],
    )
    for name in _GPU_TESTS + _CONTRACT_TESTS
]

dsv4_py_test(
    name = "test_indexer_cp_padded_gather_distributed",
    deps = _DSV4_FP8_TEST_DEPS,
    gpu_count = 2,
    tags = ["multi_device"],
)

# Neither target provides a runnable model oracle in a CI sandbox: one
# hardcodes an external official model.py, the other has an empty fixture.
[
    dsv4_py_test(
        name = name,
        deps = _DSV4_FP8_TEST_DEPS,
        tags = ["manual", "external_fixture"],
    )
    for name in ["test_dsv4_kv_cache", "test_varlen_prefill_oracle"]
]

# Compares locally constructed F.linear/cat expressions, not CompressorFP8.
dsv4_py_test(
    name = "test_compressor_gemm_merge",
    deps = _DSV4_FP8_TEST_DEPS,
    tags = ["manual", "legacy_fixture"],
)

# Preserve the combined correctness/benchmark script for explicit measurements.
# The ordinary target above collects only its correctness test.
py_test(
    name = "test_fused_decode_meta_perf",
    srcs = ["test_fused_decode_meta_comprehensive.py"],
    main = "test_fused_decode_meta_comprehensive.py",
    deps = _DSV4_FP8_TEST_DEPS,
    tags = ["manual", "perf"],
    exec_properties = cuda13_test_exec_properties(),
    target_compatible_with = DSV4_CUDA13_ONLY,
)

# Explicit entrypoints retain the scripts' strict torch.topk speed comparisons
# separately from the ordinary correctness gate.
[
    py_test(
        name = name + "_perf",
        srcs = [name + ".py"],
        main = name + ".py",
        deps = _DSV4_FP8_TEST_DEPS,
        tags = ["manual", "perf"] + (["multi_device"] if gpu_count == 2 else []),
        env = {"GPU_COUNT": str(gpu_count)},
        exec_properties = cuda13_test_exec_properties(gpu_count),
        target_compatible_with = DSV4_CUDA13_ONLY,
        legacy_create_init = 0,
    )
    for name, gpu_count in [("test_topk_v3", 2), ("test_dsv4_persistent_topk", 1)]
]

test_suite(
    name = "dsv4_fp8_cuda13_unit",
    tests = [":" + name for name in _GPU_TESTS + _CONTRACT_TESTS] + [
        ":test_indexer_cp_padded_gather_distributed",
    ],
)

# x86 repeats layout/capture/compiler boundaries and a real two-rank NCCL
# transfer. Host mathematics and orchestration are covered by the ARM lane.
test_suite(
    name = "dsv4_fp8_cuda13_cross_arch",
    tests = [
        ":test_indexer_cp_padded_gather_distributed",
        ":test_indexer_cp_padded_gather_triton",
        ":test_prefill_common_metadata_cuda",
        ":test_fused_compressor_meta_triton",
        ":test_swa_cp_packed_gather_triton",
        ":test_swa_fp8_kv_roundtrip",
        ":test_indexer_fp8_score",
        ":test_flash_mla_decode_kernel_layout",
        ":test_no_recompile",
        ":test_fused_decode_meta_comprehensive",
    ],
)
