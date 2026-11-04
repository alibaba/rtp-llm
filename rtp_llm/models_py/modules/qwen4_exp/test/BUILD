load(":defs.bzl", "qwen4_test")

package(default_visibility = ["//visibility:public"])

exports_files(["test_main.py"])

# Keep the standalone grouped-FP8 kernel test free of the native-engine testlib.
py_library(
    name = "sparse_paged_reference",
    srcs = ["reference/sparse_paged.py"],
    deps = ["//rtp_llm:torch"],
)

py_library(
    name = "math_references",
    srcs = glob(["reference/*.py"]),
    visibility = ["//visibility:public"],
    deps = ["//rtp_llm/models_py:modules"],
)

py_library(
    name = "test_support",
    srcs = glob(
        ["support/*.py"],
        exclude = ["support/test_runner.py"],
    ),
    visibility = ["//visibility:public"],
    deps = [
        ":math_references",
        "//rtp_llm:testlib",
    ],
)

py_library(
    name = "test_runner",
    srcs = ["support/test_runner.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "upstream_fixture",
    srcs = ["parity/upstream_fixture.py"],
    data = [
        "reference/upstream/manifest.json",
        "reference/upstream/modeling_qwen4_exp.py",
    ],
)

qwen4_test(
    name = "ple_test",
    src = "ple/ple_test.py",
    gpu = "L20_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "ple_gather_triton_test",
    src = "ple/ple_gather_triton_test.py",
    gpu = "L20_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "ple_prefetch_test",
    src = "ple/ple_prefetch_test.py",
    gpu = "L20_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "ple_inline_uva_graph_test",
    src = "ple/ple_inline_uva_graph_test.py",
    gpu = "L20_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "ple_decode_test",
    src = "ple/ple_decode_test.py",
    gpu = "L20_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "upstream_parity_test",
    src = "parity/upstream_parity_test.py",
    env = {"QWEN4_EXP_REQUIRE_PARITY": "1"},
    deps = [
        ":math_references",
        ":test_support",
        ":upstream_fixture",
    ],
)

qwen4_test(
    name = "gdn_upstream_parity_test",
    src = "parity/gdn_upstream_parity_test.py",
    gpu = "L20_CU13",
    deps = [
        ":math_references",
        ":test_support",
        ":upstream_fixture",
    ],
)

qwen4_test(
    name = "indexer_compressor_test",
    src = "qsa/indexer_compressor_test.py",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "indexer_decode_triton_test",
    src = "qsa/indexer_decode_triton_test.py",
    gpu = "L20_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "indexer_prefill_triton_test",
    src = "qsa/indexer_prefill_triton_test.py",
    gpu = "L20_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "moe_align_triton_test",
    src = "moe/moe_align_triton_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "moe_topk_triton_test",
    src = "moe/moe_topk_triton_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "marlin_nvfp4_test",
    src = "moe/marlin_nvfp4_test.py",
    args = [
        "MarlinNvfp4Test.test_scale_range_zero_expert_and_fp32_weight_reference",
        "MarlinNvfp4Test.test_ep_global_ids_and_all_nonlocal_routes",
        "MarlinNvfp4Test.test_graph_refreshes_values_ids_weights",
        "MarlinNvfp4Test.test_alignment_stable_order_padding_and_nonlocal_routes",
        "MarlinNvfp4Test.test_concurrent_stream_workspaces",
        "MarlinNvfp4Test.test_repack_reuses_original_storage_and_executor_contract",
    ],
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "marlin_nvfp4_hopper_test",
    src = "moe/marlin_nvfp4_test.py",
    args = ["MarlinNvfp4Test.test_hopper_strategy_uses_prepared_weights"],
    gpu = "H20_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "target_writer_fusion_test",
    src = "qsa/target_writer_fusion_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "target_metadata_test",
    src = "qsa/target_metadata_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "indexer_rope_triton_test",
    src = "qsa/indexer_rope_triton_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
        "//rtp_llm/models_py/model_desc/test/qwen4_exp:qsa_fixtures",
    ],
)

qwen4_test(
    name = "qsa_topk_triton_test",
    src = "qsa/qsa_topk_triton_test.py",
    gpu = "L20_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "qsa_decode_graph_test",
    src = "qsa/qsa_decode_graph_test.py",
    gpu = "L20_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "indexer_paged_score_test",
    src = "qsa/indexer_paged_score_test.py",
    gpu = "L20_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "sparse_fmha_test",
    src = "qsa/sparse_fmha_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "qsa_tma_prefill_test",
    src = "qsa/tma_prefill_test.py",
    gpu = "RTX_5000_PRO_CU13",
)

qwen4_test(
    name = "sparse_gqa_impl_test",
    src = "qsa/sparse_gqa_impl_test.py",
    gpu = "L20_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "sparse_paged_fmha_test",
    src = "qsa/sparse_paged_fmha_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "draft_prefill_graph_test",
    src = "qsa/draft_prefill_graph_test.py",
    gpu = "L20_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "draft_metadata_test",
    src = "qsa/draft_metadata_test.py",
    gpu = "L20_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "prefill_silu_quant_test",
    src = "qsa/prefill_silu_quant_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "sparse_prefill_bitmap_test",
    src = "qsa/sparse_prefill_bitmap_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "indexer_prefill_score_test",
    src = "qsa/indexer_prefill_score_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "small_moe_gemv_test",
    src = "moe/small_moe_gemv_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "pd_cache_store_test",
    src = "qsa/pd_cache_store_test.py",
    gpu = "L20_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "gdn_composition_test",
    src = "gdn/gdn_composition_test.py",
    gpu = "L20_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "fp8_kv_test",
    src = "qsa/fp8_kv_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "indexer_tensorcore_test",
    src = "qsa/indexer_tensorcore_test.py",
    gpu = "RTX_5000_PRO_CU13",
    testlib = "//rtp_llm/models_py/standalone:auto_model",
)

qwen4_test(
    name = "prefill_preflight_test",
    src = "qsa/prefill_preflight_test.py",
    gpu = "RTX_5000_PRO_CU13",
    testlib = "//rtp_llm/models_py/standalone:auto_model",
)

qwen4_test(
    name = "compute_weight_view_test",
    src = "parity/compute_weight_view_test.py",
    testlib = "//rtp_llm/models_py/standalone:auto_model",
)

qwen4_test(
    name = "marlin_large_decode_test",
    src = "moe/marlin_large_decode_test.py",
    gpu = "RTX_5000_PRO_CU13",
    testlib = "//rtp_llm/models_py/standalone:auto_model",
)

qwen4_test(
    name = "gated_residual_bf16_test",
    src = "parity/gated_residual_bf16_test.py",
    gpu = "RTX_5000_PRO_CU13",
    testlib = "//rtp_llm/models_py/standalone:auto_model",
)

qwen4_test(
    name = "histogram_topk_test",
    src = "qsa/histogram_topk_test.py",
    gpu = "RTX_5000_PRO_CU13",
    testlib = "//rtp_llm/models_py/standalone:auto_model",
)

qwen4_test(
    name = "hc_prefill_mix_test",
    src = "parity/hc_prefill_mix_test.py",
    gpu = "RTX_5000_PRO_CU13",
    testlib = "//rtp_llm/models_py/standalone:auto_model",
)

qwen4_test(
    name = "chunk_paged_gqa_test",
    src = "qsa/chunk_paged_gqa_test.py",
    gpu = "RTX_5000_PRO_CU13",
    testlib = "//rtp_llm/models_py/standalone:auto_model",
    deps = [":sparse_paged_reference"],
)

qwen4_test(
    name = "grouped_fp8_test",
    src = "qsa/grouped_fp8_test.py",
    gpu = "RTX_5000_PRO_CU13",
    testlib = "//rtp_llm/models_py/standalone:auto_model",
    deps = [":sparse_paged_reference"],
)

qwen4_test(
    name = "upstream_parity_qualification",
    src = "parity/real_weight_parity_test.py",
    env = {"QWEN4_EXP_REQUIRE_PARITY": "1"},
    gpu = "RTX_5000_PRO_CU13",
    tags = ["manual"],
    deps = [
        ":math_references",
        ":test_support",
        ":upstream_fixture",
    ],
)

qwen4_test(
    name = "gdn_upstream_parity_qualification",
    src = "parity/gdn_upstream_parity_test.py",
    env = {"QWEN4_EXP_REQUIRE_PARITY": "1"},
    gpu = "RTX_5000_PRO_CU13",
    tags = ["manual"],
    deps = [
        ":math_references",
        ":test_support",
        ":upstream_fixture",
    ],
)

qwen4_test(
    name = "ple_outer_residual_test",
    src = "ple/ple_outer_residual_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "ple_commit_metadata_test",
    src = "ple/ple_commit_metadata_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "ple_commit_prepare_test",
    src = "ple/ple_commit_prepare_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "ple_commit_test",
    src = "ple/ple_commit_test.py",
    gpu = "L20_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "ple_hash_triton_test",
    src = "ple/ple_hash_triton_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "gdn_prefill_norm_test",
    src = "gdn/gdn_prefill_norm_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "ple_gate_test",
    src = "ple/ple_gate_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "gdn_prefill_adapter_test",
    src = "gdn/gdn_prefill_adapter_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "indexer_norm_rope_test",
    src = "qsa/indexer_norm_rope_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "qsa_packed_count_test",
    src = "qsa/qsa_packed_count_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "ple_read_blocks_test",
    src = "ple/ple_read_blocks_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "main_qk_norm_rope_test",
    src = "qsa/main_qk_norm_rope_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "qsa_projection_pair_test",
    src = "qsa/qsa_projection_pair_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "mtp_embedding_norm_test",
    src = "qsa/mtp_embedding_norm_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "qsa_cub_topk_test",
    src = "qsa/qsa_cub_topk_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "qsa_target_prepare_test",
    src = "qsa/qsa_target_prepare_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "tp_registered_input_test",
    src = "distributed/tp_registered_input_test.py",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "prefetch_order_test",
    src = "qsa/prefetch_order_test.py",
    gpu = "L20_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "decode_snapshot_test",
    src = "qsa/decode_snapshot_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "main_qk_cache_test",
    src = "qsa/main_qk_cache_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "qsa_main_metadata_test",
    src = "qsa/qsa_main_metadata_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "decode_rope_stride_test",
    src = "qsa/decode_rope_stride_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "rope_cache_policy_test",
    src = "qsa/rope_cache_policy_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "target_head_graph_test",
    src = "qsa/target_head_graph_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "draft_head_graph_test",
    src = "qsa/draft_head_graph_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "gdn_norm_quant_test",
    src = "gdn/gdn_norm_quant_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "shared_expert_overlap_test",
    src = "moe/shared_expert_overlap_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "head_ipc_gather_test",
    src = "distributed/head_ipc_gather_test.py",
    args = ["HeadIpcGatherTest"],
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "head_ipc_gather_integration_test",
    src = "distributed/head_ipc_gather_test.py",
    args = ["HeadIpcGatherIntegrationTest"],
    gpu = "RTX_5000_PRO_CU13",
    gpus = 2,
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "draft_qsa_fused_test",
    src = "qsa/draft_qsa_fused_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "draft_qsa_layout_test",
    src = "qsa/draft_qsa_layout_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "draft_paged_fmha_test",
    src = "qsa/draft_paged_fmha_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "draft_qsa_metadata_test",
    src = "qsa/draft_qsa_metadata_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "cpu_target_status_test",
    src = "distributed/cpu_target_status_test.py",
    gpu = "L20_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "eager_validation_test",
    src = "qsa/eager_validation_test.py",
)

qwen4_test(
    name = "indexer_cpu_contract_test",
    src = "qsa/indexer_cpu_contract_test.py",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

qwen4_test(
    name = "indexer_cuda_contract_test",
    src = "qsa/indexer_cuda_contract_test.py",
    gpu = "L20_CU13",
    deps = [
        ":math_references",
        ":test_support",
    ],
)

# Preserve the existing CI entrypoint while sharing component parity coverage.
test_suite(
    name = "upstream_fixture_test",
    tests = [":upstream_parity_test"],
)

test_suite(
    name = "qwen4_parity",
    tests = [
        ":gdn_upstream_parity_test",
        ":hc_prefill_mix_test",
        ":gated_residual_bf16_test",
        ":compute_weight_view_test",
        ":histogram_topk_test",
        ":upstream_parity_test",
    ],
)

test_suite(
    name = "qwen4_ple",
    tests = [
        ":ple_commit_metadata_test",
        ":ple_commit_prepare_test",
        ":ple_commit_test",
        ":ple_decode_test",
        ":ple_gate_test",
        ":ple_gather_triton_test",
        ":ple_hash_triton_test",
        ":ple_inline_uva_graph_test",
        ":ple_outer_residual_test",
        ":ple_prefetch_test",
        ":ple_read_blocks_test",
        ":ple_test",
    ],
)

test_suite(
    name = "qwen4_qsa",
    tests = [
        ":decode_rope_stride_test",
        ":decode_snapshot_test",
        ":draft_head_graph_test",
        ":draft_metadata_test",
        ":draft_paged_fmha_test",
        ":draft_prefill_graph_test",
        ":draft_qsa_fused_test",
        ":draft_qsa_layout_test",
        ":draft_qsa_metadata_test",
        ":fp8_kv_test",
        ":chunk_paged_gqa_test",
        ":indexer_tensorcore_test",
        ":prefill_preflight_test",
        ":grouped_fp8_test",
        ":indexer_compressor_test",
        ":indexer_cpu_contract_test",
        ":indexer_cuda_contract_test",
        ":indexer_decode_triton_test",
        ":indexer_norm_rope_test",
        ":indexer_paged_score_test",
        ":indexer_prefill_score_test",
        ":indexer_prefill_triton_test",
        ":indexer_rope_triton_test",
        ":main_qk_cache_test",
        ":main_qk_norm_rope_test",
        ":mtp_embedding_norm_test",
        ":pd_cache_store_test",
        ":prefetch_order_test",
        ":prefill_silu_quant_test",
        ":qsa_cub_topk_test",
        ":qsa_decode_graph_test",
        ":qsa_main_metadata_test",
        ":qsa_packed_count_test",
        ":qsa_projection_pair_test",
        ":qsa_target_prepare_test",
        ":qsa_tma_prefill_test",
        ":qsa_topk_triton_test",
        ":rope_cache_policy_test",
        ":sparse_fmha_test",
        ":sparse_gqa_impl_test",
        ":sparse_paged_fmha_test",
        ":sparse_prefill_bitmap_test",
        ":target_head_graph_test",
        ":target_metadata_test",
        ":target_writer_fusion_test",
    ],
)

test_suite(
    name = "qwen4_moe",
    tests = [
        ":marlin_large_decode_test",
        ":marlin_nvfp4_hopper_test",
        ":marlin_nvfp4_test",
        ":moe_align_triton_test",
        ":moe_topk_triton_test",
        ":shared_expert_overlap_test",
        ":small_moe_gemv_test",
    ],
)

test_suite(
    name = "qwen4_gdn",
    tests = [
        ":gdn_composition_test",
        ":gdn_norm_quant_test",
        ":gdn_prefill_adapter_test",
        ":gdn_prefill_norm_test",
    ],
)

test_suite(
    name = "qwen4_distributed",
    tests = [
        ":cpu_target_status_test",
        ":head_ipc_gather_integration_test",
        ":head_ipc_gather_test",
        ":tp_registered_input_test",
    ],
)

test_suite(
    name = "qwen4_components",
    tests = [
        ":qwen4_gdn",
        ":qwen4_moe",
        ":qwen4_parity",
        ":qwen4_ple",
        ":qwen4_qsa",
    ],
)
