load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")
load("//rtp_llm/models_py/test:defs.bzl", "models_py_test")

package(default_visibility = ["//visibility:public"])

models_py_test(
    name = "hc_defer_policy_test",
    src = "hc_defer_policy_test.py",
)

models_py_test(
    name = "indexer_prefix_reuse_snapshot_fusion_test",
    src = "indexer_prefix_reuse_snapshot_fusion_test.py",
    gpu = "H20_CU13",
)

models_py_test(
    name = "qsa_indexer_buffer_sm90_test",
    src = "qsa_indexer_buffer_sm90_test.py",
    gpu = "H20_CU13",
)

models_py_test(
    name = "qsa_mrope_fused_sm90_test",
    src = "qsa_mrope_fused_sm90_test.py",
    gpu = "H20_CU13",
)

py_test(
    name = "qsa_writer_geometry_test",
    srcs = [
        "qsa_writer_geometry_test.py",
        "qsa_draft_tail_vector_test.py",
        "qsa/indexer_prefill_triton_test.py",
        "indexer_prefix_reuse_triton_test.py",
    ],
    deps = ["//rtp_llm:testlib", ":math_references"],
    env = {"GPU_COUNT": "1"},
    tags = ["H20_CU13"],
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
)

models_py_test(
    name = "qsa_cu_seqlens_host_test",
    src = "qsa_cu_seqlens_host_test.py",
    gpu = "H20_CU13",
)

models_py_test(
    name = "qsa_draft_tail_vector_test",
    src = "qsa_draft_tail_vector_test.py",
    gpu = "H20_CU13",
)

models_py_test(
    name = "qsa_grid_prep_fusion_test",
    src = "qsa_grid_prep_fusion_test.py",
    gpu = "H20_CU13",
)

models_py_test(
    name = "qsa_qk_projection_fusion_test",
    src = "qsa_qk_projection_fusion_test.py",
    gpu = "H20_CU13",
)

models_py_test(
    name = "qsa_rope_fused_sm90_test",
    src = "qsa_rope_fused_sm90_test.py",
    gpu = "H20_CU13",
)

models_py_test(
    name = "ple_metadata_sync_test",
    src = "ple_metadata_sync_test.py",
    gpu = "H20_CU13",
)

models_py_test(
    name = "ple_page_projection_jit_test",
    src = "ple_page_projection_jit_test.py",
    gpu = "H20_CU13",
)

models_py_test(
    name = "hc_norm_fp8_scale_jit_test",
    src = "hc_norm_fp8_scale_jit_test.py",
    gpu = "H20_CU13",
)

models_py_test(
    name = "hc_gdn_moe_prepare_once_test",
    src = "hc_gdn_moe_prepare_once_test.py",
    gpu = "H20_CU13",
)

models_py_test(
    name = "qsa_prepare_once_validation_test",
    src = "qsa_prepare_once_validation_test.py",
    gpu = "H20_CU13",
)

models_py_test(
    name = "sparse_paged_slim_test",
    src = "sparse_paged_slim_test.py",
    gpu = "H20_CU13",
)

models_py_test(
    name = "sparse_paged_trust_test",
    src = "sparse_paged_trust_test.py",
    gpu = "H20_CU13",
)

models_py_test(
    name = "target_reduce_test",
    src = "target_reduce_test.py",
)

py_test(
    name = "qsa_triton_recompile_regression_test",
    srcs = [
        "qsa_triton_recompile_regression_test.py",
        "indexer_prefix_reuse_triton_test.py",
        "ple_sm90_test_utils.py",
    ],
    deps = ["//rtp_llm:testlib", ":math_references"],
    env = {"GPU_COUNT": "1"},
    tags = ["H20_CU13"],
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
)

_REFERENCE_DEPS = [
    ":math_references",
    ":test_support",
]

models_py_test(
    name = "state_cache_inputs_test",
    src = "state_cache_inputs_test.py",
    gpu = "L20_CU13",
)

# Keep the standalone grouped-FP8 kernel test free of the native-engine testlib.

py_library(
    name = "math_references",
    srcs = glob(["reference/*.py"]),
    visibility = ["//visibility:public"],
    deps = ["//rtp_llm/models_py:modules", "//rtp_llm/models_py/kernels/cuda/qsa/test/reference:math_references"],
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

models_py_test(
    name = "qsa_step_plan_test",
    src = "qsa_step_plan_test.py",
)

models_py_test(
    name = "indexer_prefix_reuse_triton_test",
    src = "indexer_prefix_reuse_triton_test.py",
    gpu = "H20_CU13",
)

models_py_test(
    name = "qwen4_profiler_ranges_test",
    src = "qwen4_profiler_ranges_test.py",
    gpus = 0,
)

py_library(
    name = "upstream_fixture",
    srcs = ["parity/upstream_fixture.py"],
    data = [
        "reference/upstream/manifest.json",
        "reference/upstream/modeling_qwen4_exp.py",
    ],
)

models_py_test(
    name = "ple_test",
    src = "ple/ple_test.py",
    gpu = "L20_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "ple_prefetch_test",
    src = "ple/ple_prefetch_test.py",
    gpu = "L20_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "ple_inline_uva_graph_test",
    src = "ple/ple_inline_uva_graph_test.py",
    gpu = "L20_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "ple_decode_test",
    src = "ple/ple_decode_test.py",
    gpu = "L20_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "upstream_parity_test",
    src = "parity/upstream_parity_test.py",
    env = {"QWEN4_EXP_REQUIRE_PARITY": "1"},
    deps = [
        ":math_references",
        ":test_support",
        ":upstream_fixture",
    ],
)

models_py_test(
    name = "gdn_upstream_parity_test",
    src = "parity/gdn_upstream_parity_test.py",
    gpu = "L20_CU13",
    deps = [
        ":math_references",
        ":test_support",
        ":upstream_fixture",
    ],
)

models_py_test(
    name = "indexer_compressor_test",
    src = "qsa/indexer_compressor_test.py",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "indexer_decode_triton_test",
    src = "qsa/indexer_decode_triton_test.py",
    gpu = "L20_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "indexer_prefill_triton_test",
    src = "qsa/indexer_prefill_triton_test.py",
    gpu = "L20_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "moe_topk_triton_test",
    src = "moe/moe_topk_triton_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "marlin_admission_test",
    src = "moe/marlin_admission_test.py",
    data = [
        "//rtp_llm/models_py/kernels/cuda/marlin_nvfp4/test:marlin_large_decode_test.py",
        "//rtp_llm/models_py/kernels/cuda/marlin_nvfp4/test:marlin_nvfp4_test.py",
    ],
    deps = ["//rtp_llm/models_py/kernels/cuda/marlin_nvfp4/test:marlin_test_support"],
)

models_py_test(
    name = "marlin_nvfp4_test",
    src = "moe/marlin_nvfp4_test.py",
    args = [
        "MarlinNvfp4Test.test_repack_rebinds_final_storage_and_executor_contract",
        "MarlinNvfp4Test.test_resident_source_mtp_restart",
        "MarlinNvfp4Test.test_checkpoint_experts_against_dequantized_reference",
    ],
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS + ["//rtp_llm/models_py/kernels/cuda/marlin_nvfp4/test:marlin_test_support"],
)

models_py_test(
    name = "marlin_nvfp4_hopper_test",
    src = "moe/marlin_nvfp4_test.py",
    args = [
        "MarlinNvfp4Test.test_explicit_strategy_uses_prepared_weights",
        "MarlinNvfp4Test.test_resident_source_mtp_restart",
        "MarlinNvfp4Test.test_checkpoint_experts_against_dequantized_reference",
    ],
    gpu = "H20_CU13",
    deps = _REFERENCE_DEPS + ["//rtp_llm/models_py/kernels/cuda/marlin_nvfp4/test:marlin_test_support"],
)

models_py_test(
    name = "target_writer_fusion_test",
    src = "qsa/target_writer_fusion_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "indexer_rope_triton_test",
    src = "qsa/indexer_rope_triton_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":math_references",
        ":test_support",
        "//rtp_llm/models_py/model_desc/test/qwen4_exp:qsa_fixtures",
    ],
)

models_py_test(
    name = "qsa_topk_triton_test",
    src = "qsa/qsa_topk_triton_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "qsa_decode_graph_test",
    src = "qsa/qsa_decode_graph_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "sparse_gqa_impl_test",
    src = "qsa/sparse_gqa_impl_test.py",
    gpu = "L20_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "sparse_paged_fmha_test",
    src = "qsa/sparse_paged_fmha_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "draft_prefill_graph_test",
    src = "qsa/draft_prefill_graph_test.py",
    gpu = "L20_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "prefill_silu_quant_test",
    src = "qsa/prefill_silu_quant_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "pd_cache_store_test",
    src = "qsa/pd_cache_store_test.py",
    gpu = "L20_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "gdn_composition_test",
    src = "gdn/gdn_composition_test.py",
    gpu = "L20_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "fp8_kv_test",
    src = "qsa/fp8_kv_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS + ["//rtp_llm/models_py/kernels/cuda/qsa/test:fp8_kv_test_support"],
)

models_py_test(
    name = "prefill_preflight_test",
    src = "qsa/prefill_preflight_test.py",
    gpu = "RTX_5000_PRO_CU13",
    testlib = "//rtp_llm/models_py/standalone:auto_model",
)

models_py_test(
    name = "compute_weight_view_test",
    src = "parity/compute_weight_view_test.py",
    testlib = "//rtp_llm/models_py/standalone:auto_model",
)

models_py_test(
    name = "gated_residual_bf16_test",
    src = "parity/gated_residual_bf16_test.py",
    gpu = "RTX_5000_PRO_CU13",
    testlib = "//rtp_llm/models_py/standalone:auto_model",
)

models_py_test(
    name = "hc_prefill_mix_test",
    src = "parity/hc_prefill_mix_test.py",
    gpu = "RTX_5000_PRO_CU13",
    testlib = "//rtp_llm/models_py/standalone:auto_model",
)

models_py_test(
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

models_py_test(
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

models_py_test(
    name = "ple_commit_metadata_test",
    src = "ple/ple_commit_metadata_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "ple_commit_prepare_test",
    src = "ple/ple_commit_prepare_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "ple_commit_test",
    src = "ple/ple_commit_test.py",
    gpu = "L20_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "gdn_prefill_norm_test",
    src = "gdn/gdn_prefill_norm_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "ple_gate_test",
    src = "ple/ple_gate_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "qsa_projection_pair_test",
    src = "qsa/qsa_projection_pair_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "mtp_embedding_norm_test",
    src = "qsa/mtp_embedding_norm_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "qsa_target_prepare_test",
    src = "qsa/qsa_target_prepare_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "tp_registered_input_test",
    src = "distributed/tp_registered_input_test.py",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "prefetch_order_test",
    src = "qsa/prefetch_order_test.py",
    gpu = "L20_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "decode_snapshot_test",
    src = "qsa/decode_snapshot_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "main_qk_cache_test",
    src = "qsa/main_qk_cache_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "qsa_main_metadata_test",
    src = "qsa/qsa_main_metadata_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "decode_rope_stride_test",
    src = "qsa/decode_rope_stride_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS,
)

py_test(
    name = "decode_rope_block64_repro_test",
    srcs = ["decode_rope_block64_repro_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["H20_CU13"],
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
)

models_py_test(
    name = "prefill_rope_padding_tail_replay_repro_test",
    src = "prefill_rope_padding_tail_replay_repro_test.py",
    gpu = "H20_CU13",
)

models_py_test(
    name = "rope_cache_policy_test",
    src = "qsa/rope_cache_policy_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "gdn_norm_quant_test",
    src = "gdn/gdn_norm_quant_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "shared_expert_overlap_test",
    src = "moe/shared_expert_overlap_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "draft_qsa_fused_test",
    src = "qsa/draft_qsa_fused_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "cpu_target_status_test",
    src = "distributed/cpu_target_status_test.py",
    # Native CPU bindings share a CUDA-linked library; no CUDA device is needed.
    gpu = "L20_CU13",
    gpus = 0,
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "eager_validation_test",
    src = "qsa/eager_validation_test.py",
)

models_py_test(
    name = "indexer_cpu_contract_test",
    src = "qsa/indexer_cpu_contract_test.py",
    deps = _REFERENCE_DEPS,
)

models_py_test(
    name = "indexer_cuda_contract_test",
    src = "qsa/indexer_cuda_contract_test.py",
    gpu = "L20_CU13",
    deps = _REFERENCE_DEPS,
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
        "//rtp_llm/models_py/kernels/cuda/qsa/test:histogram_topk_test",
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
        "//rtp_llm/models_py/kernels/cuda/qwen4_ple/test:gate_test",
        "//rtp_llm/models_py/kernels/cuda/qwen4_ple/test:ple_gather_triton_test",
        "//rtp_llm/models_py/kernels/cuda/qwen4_ple/test:ple_hash_triton_test",
        ":ple_inline_uva_graph_test",
        "//rtp_llm/models_py/kernels/cuda/qwen4_ple/test:ple_outer_residual_test",
        ":ple_prefetch_test",
        "//rtp_llm/models_py/kernels/cuda/qwen4_ple/test:ple_read_blocks_test",
        ":ple_test",
    ],
)

test_suite(
    name = "qwen4_qsa",
    tests = [
        "//rtp_llm/models_py/kernels/cuda/qsa/test:sparse_paged_fmha_test",
        "//rtp_llm/models_py/kernels/cuda/qsa/test:fp8_kv_test",
        "//rtp_llm/models_py/kernels/cuda/qsa/test:topk_test",
        "//rtp_llm/models_py/triton_kernels/common/test:silu_fp8_quant_test",
        ":decode_rope_stride_test",
        ":decode_snapshot_test",
        "//rtp_llm/models_py/kernels/cuda/qsa/test:draft_metadata_test",
        "//rtp_llm/models_py/kernels/cuda/qsa/test:draft_paged_fmha_test",
        ":draft_prefill_graph_test",
        ":draft_qsa_fused_test",
        "//rtp_llm/models_py/kernels/cuda/qsa/test:draft_qsa_layout_test",
        "//rtp_llm/models_py/kernels/cuda/qsa/test:draft_qsa_metadata_test",
        ":fp8_kv_test",
        "//rtp_llm/models_py/kernels/cuda/qsa/test:chunk_paged_gqa_test",
        "//rtp_llm/models_py/kernels/cuda/qsa/test:indexer_tensorcore_test",
        ":prefill_preflight_test",
        "//rtp_llm/models_py/kernels/cuda/qsa/test:grouped_fp8_test",
        ":indexer_compressor_test",
        ":indexer_cpu_contract_test",
        ":indexer_cuda_contract_test",
        ":indexer_decode_triton_test",
        "//rtp_llm/models_py/kernels/cuda/qsa/test:indexer_norm_rope_test",
        "//rtp_llm/models_py/kernels/cuda/qsa/test:indexer_paged_score_test",
        "//rtp_llm/models_py/kernels/cuda/qsa/test:indexer_prefill_score_test",
        ":indexer_prefill_triton_test",
        ":indexer_rope_triton_test",
        ":main_qk_cache_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:main_qk_norm_rope_test",
        ":mtp_embedding_norm_test",
        "//rtp_llm/models_py/kernels/cuda/test:qwen4_mtp_norm_test",
        ":pd_cache_store_test",
        ":prefetch_order_test",
        ":prefill_silu_quant_test",
        "//rtp_llm/models_py/kernels/cuda/qsa/test:qsa_cub_topk_test",
        ":qsa_decode_graph_test",
        ":qsa_main_metadata_test",
        "//rtp_llm/models_py/kernels/cuda/qsa/test:qsa_packed_count_test",
        ":qsa_projection_pair_test",
        ":qsa_target_prepare_test",
        "//rtp_llm/models_py/kernels/cuda/qsa/test:qsa_tma_prefill_test",
        ":qsa_topk_triton_test",
        ":rope_cache_policy_test",
        "//rtp_llm/models_py/kernels/cuda/qsa/test:sparse_fmha_test",
        ":sparse_gqa_impl_test",
        ":sparse_paged_fmha_test",
        "//rtp_llm/models_py/kernels/cuda/qsa/test:sparse_prefill_bitmap_test",
        "//rtp_llm/models_py/kernels/cuda/qsa/test:target_metadata_test",
        ":target_writer_fusion_test",
    ],
)

test_suite(
    name = "qwen4_moe",
    tests = [
        "//rtp_llm/models_py/kernels/cuda/marlin_nvfp4/test:marlin_nvfp4_hopper_test",
        "//rtp_llm/models_py/kernels/cuda/marlin_nvfp4/test:marlin_nvfp4_test",
        "//rtp_llm/models_py/triton_kernels/moe/test:decode_alignment_test",
        "//rtp_llm/models_py/triton_kernels/moe/test:small_moe_gemv_test",
        ":factory_isolation_test",
        "//rtp_llm/models_py/kernels/cuda/marlin_nvfp4/test:marlin_large_decode_test",
        ":marlin_nvfp4_hopper_test",
        ":marlin_nvfp4_test",
        "//rtp_llm/models_py/kernels/cuda/marlin_nvfp4/test:marlin_prefill_test",
        ":moe_topk_triton_test",
        "//rtp_llm/models_py/triton_kernels/moe/test:qwen4_topk_test",
        ":shared_expert_overlap_test",
        "//rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/executors/test:small_moe_gemv_test",
    ],
)

test_suite(
    name = "qwen4_gdn",
    tests = [
        ":gdn_composition_test",
        ":gdn_norm_quant_test",
        "//rtp_llm/models_py/kernels/cuda/gdn_sm120/test:norm_quant_test",
        "//rtp_llm/models_py/kernels/cuda/gdn_sm120/test:norm_test",
        "//rtp_llm/models_py/kernels/cuda/gdn_sm120/test:prefill_test",
        ":gdn_prefill_norm_test",
    ],
)

test_suite(
    name = "qwen4_distributed",
    tests = [
        ":cpu_target_status_test",
        ":tp_registered_input_test",
    ],
)

test_suite(
    name = "qwen4_components",
    tests = [
        "//rtp_llm/models_py/modules/hybrid/test:residual_up_mix_test",
        "//rtp_llm/models_py/modules/hybrid/test:residual_norm_fp8_test",
        "//rtp_llm/models_py/modules/hybrid/test:residual_prefill_up_mix_test",
        "//rtp_llm/models_py/modules/hybrid/test:residual_prefill_inject_norm_test",
        "//rtp_llm/models_py/modules/hybrid/test:residual_inject_projection_test",
        "//rtp_llm/models_py/modules/hybrid/test:residual_splitk_inject_test",
        "//rtp_llm/models_py/modules/base/common/test:fp8_quantization_contract_test",
        "//rtp_llm/models_py/modules/base/common/test:fp8_dense_gemv_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:main_qk_norm_rope_test",
        "//rtp_llm/models_py/triton_kernels/gated_residual/test:epilogue_mix_test",
        "//rtp_llm/models_py/triton_kernels/common/test:vocab_byte_gather_test",
        "//rtp_llm/models_py/triton_kernels/common/test:zero_centered_compile_test",
        "//rtp_llm/models_py/triton_kernels/common/test:grouped_qk_norm_test",
        "//rtp_llm/models_py/triton_kernels/common/test:vocab_lookup_test",
        ":qwen4_gdn",
        ":qwen4_moe",
        ":qwen4_parity",
        ":qwen4_ple",
        ":qwen4_qsa",
    ],
)

models_py_test(
    name = "factory_isolation_test",
    src = "moe/factory_isolation_test.py",
    gpu = "RTX_5000_PRO_CU13",
)

models_py_test(
    name = "main_qk_norm_rope_test",
    src = "qsa/main_qk_norm_rope_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [":test_support"],
)
py_test(
    name = "rocm_qsa_rope_cache_test",
    srcs = ["rocm_qsa_rope_cache_test.py"],
    deps = ["//rtp_llm:testlib"],
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

# Kernel-only prefill-score targets: deps pull the real production scorer
# through //rtp_llm/models_py:qsa_prefill_score_testlib (torch+triton only),
# not the //rtp_llm:testlib engine closure or :math_references.
py_test(
    name = "indexer_prefill_score_integration_test",
    srcs = [
        "indexer_prefill_score_integration_test.py",
        "indexer_prefill_score_preintegration.py",
        "indexer_prefill_score_test_utils.py",
    ],
    main = "indexer_prefill_score_integration_test.py",
    deps = ["//rtp_llm/models_py:qsa_prefill_score_testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["H20_CU13"],
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
)

models_py_test(
    name = "sm90_prefill_indexer_test",
    src = "sm90_prefill_indexer_test.py",
    gpu = "H20_CU13",
)

py_test(
    name = "ple_sm90_components_test",
    srcs = ["ple_sm90_components_test.py", "ple_sm90_test_utils.py"],
    main = "ple_sm90_components_test.py",
    deps = ["//rtp_llm/models_py:ple_component_testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["H20_CU13"],
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
)


cc_binary(
    name = "ple_native_quant_test_ops.so",
    testonly = True,
    srcs = ["ple_native_quant_test_ops.cc"],
    defines = ["ENABLE_FP8"],
    copts = copts(),
    linkshared = True,
    deps = ["//rtp_llm/models_py/bindings/cuda/kernels:scaled_fp8_quant"] + torch_deps(),
)

py_test(
    name = "ple_sm90_pages_test",
    srcs = ["ple_sm90_pages_test.py", "ple_sm90_test_utils.py", "ple_native_quant_adapter.py"],
    main = "ple_sm90_pages_test.py",
    data = [":ple_native_quant_test_ops.so"],
    deps = ["//rtp_llm/models_py:ple_component_testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["H20_CU13"],
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
)

py_test(
    name = "ple_sm90_pages_benchmark",
    srcs = ["ple_sm90_pages_benchmark.py", "ple_sm90_pages_test.py", "ple_sm90_test_utils.py", "ple_native_quant_adapter.py"],
    main = "ple_sm90_pages_benchmark.py",
    data = [":ple_native_quant_test_ops.so"],
    deps = ["//rtp_llm/models_py:ple_component_testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["H20_CU13", "manual"],
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
)

py_test(
    name = "sparse_prefill_sm90_test",
    srcs = ["sparse_prefill_sm90_test.py", "sparse_prefill_sm90_test_utils.py"],
    main = "sparse_prefill_sm90_test.py",
    deps = ["//rtp_llm/models_py:qsa_sparse_prefill_testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["H20_CU13"],
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
)
