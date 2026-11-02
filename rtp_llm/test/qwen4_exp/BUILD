# Semantic ownership is kept in component packages. These explicit resource
# suites are the developer/CI entrypoints; hardware tags still select each pool.

test_suite(
    name = "qwen4_cpu",
    tests = [
        "//rtp_llm/model_loader/test:nvfp4_cache_test",
        "//rtp_llm/models/qwen4_exp/test:qwen4_config",
        "//rtp_llm/models_py/modules/qwen4_exp/test:cpu_target_status_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:indexer_compressor_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:indexer_cpu_contract_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:tp_registered_input_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:upstream_parity_test",
        "//rtp_llm/test:qwen4_request_admission_test",
    ],
)

test_suite(
    name = "qwen4_single_gpu",
    tests = [
        "//rtp_llm/models/qwen4_exp/test:full_fp8_test",
        "//rtp_llm/models_py/model_desc/test/qwen4_exp:qwen4_runtime",
        "//rtp_llm/models_py/modules/qwen4_exp/test:decode_rope_stride_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:decode_snapshot_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:draft_head_graph_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:draft_metadata_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:draft_paged_fmha_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:draft_prefill_graph_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:draft_qsa_fused_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:draft_qsa_layout_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:draft_qsa_metadata_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:fp8_kv_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:gdn_upstream_parity_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:grouped_fp8_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:head_ipc_gather_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:indexer_cuda_contract_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:indexer_decode_triton_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:indexer_norm_rope_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:indexer_paged_score_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:indexer_prefill_score_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:indexer_prefill_triton_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:indexer_rope_triton_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:main_qk_cache_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:main_qk_norm_rope_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:mtp_embedding_norm_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:pd_cache_store_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:prefetch_order_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:prefill_silu_quant_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:qsa_cub_topk_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:qsa_decode_graph_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:qsa_main_metadata_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:qsa_packed_count_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:qsa_projection_pair_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:qsa_target_prepare_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:qsa_topk_triton_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:qwen4_ple",
        "//rtp_llm/models_py/modules/qwen4_exp/test:qwen4_moe",
        "//rtp_llm/models_py/modules/qwen4_exp/test:qwen4_gdn",
        "//rtp_llm/models_py/modules/qwen4_exp/test:rope_cache_policy_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:sparse_fmha_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:sparse_gqa_impl_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:sparse_paged_fmha_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:sparse_prefill_bitmap_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:target_head_graph_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:target_metadata_test",
        "//rtp_llm/models_py/modules/qwen4_exp/test:target_writer_fusion_test",
    ],
)

test_suite(
    name = "qwen4_distributed",
    tests = ["//rtp_llm/models_py/modules/qwen4_exp/test:head_ipc_gather_integration_test"],
)

test_suite(
    name = "qwen4_smoke",
    tests = ["//rtp_llm/test/smoke:smoke_sm120_qwen4_exp"],
)

test_suite(
    name = "qwen4_qualification",
    tags = ["manual"],
    tests = [
        "//rtp_llm/models_py/modules/qwen4_exp/test:gdn_upstream_parity_qualification",
        "//rtp_llm/models_py/modules/qwen4_exp/test:upstream_parity_qualification",
    ],
)

test_suite(
    name = "qwen4_regression",
    tests = [
        ":qwen4_cpu",
        ":qwen4_distributed",
        ":qwen4_single_gpu",
    ],
)
