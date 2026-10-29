py_library(
    name = "math_references",
    srcs = glob(["reference/*.py"]),
    deps = ["//rtp_llm/models_py:modules"],
)

py_test(
    name = "upstream_fixture_test",
    srcs = ["upstream_fixture_test.py", "upstream_fixture.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "residual_stream_test",
    srcs = ["residual_stream_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "ple_test",
    srcs = ["ple_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "ple_gather_triton_test",
    srcs = ["ple_gather_triton_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "ple_prefetch_test",
    srcs = ["ple_prefetch_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "ple_inline_uva_graph_test",
    srcs = ["ple_inline_uva_graph_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "ple_decode_test",
    srcs = ["ple_decode_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

# Skips unless QWEN4_EXP_UPSTREAM_SRC points at transformers'
# modeling_qwen4_exp.py; see the module docstring.
py_test(
    name = "upstream_parity_test",
    srcs = ["upstream_parity_test.py", "upstream_fixture.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

# GDN (linear attention) parity on a tiny K != V config; also skips unless
# QWEN4_EXP_UPSTREAM_SRC is set.
py_test(
    name = "gdn_upstream_parity_test",
    srcs = ["gdn_upstream_parity_test.py", "upstream_fixture.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "indexer_score_test",
    srcs = ["indexer_score_test.py", "reference/indexer_score.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "indexer_compressor_test",
    srcs = ["indexer_compressor_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
)

py_test(
    name = "indexer_decode_triton_test",
    srcs = ["indexer_decode_triton_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "indexer_prefill_triton_test",
    srcs = ["indexer_prefill_triton_test.py", "scalar_cache_reference.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "moe_align_triton_test",
    srcs = ["moe_align_triton_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "moe_topk_triton_test",
    srcs = ["moe_topk_triton_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "indexer_runtime_contract_test",
    srcs = ["indexer_runtime_contract_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
)

py_test(
    name = "target_writer_fusion_test",
    srcs = ["target_writer_fusion_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "target_metadata_test",
    srcs = ["target_metadata_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "residual_up_mix_test",
    srcs = ["residual_up_mix_test.py", "residual_norm_fp8_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "residual_norm_fp8_test",
    srcs = ["residual_norm_fp8_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "indexer_rope_triton_test",
    srcs = ["indexer_rope_triton_test.py", "qsa_runtime_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "qsa_runtime_test",
    srcs = ["qsa_runtime_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "qsa_topk_triton_test",
    srcs = ["qsa_topk_triton_test.py", "topk_reference.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "qsa_decode_graph_test",
    srcs = ["qsa_decode_graph_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "indexer_paged_score_test",
    srcs = ["indexer_paged_score_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "sparse_attention_test",
    srcs = ["sparse_attention_test.py", "reference/sparse_attention.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "sparse_fmha_test",
    srcs = ["sparse_fmha_test.py", "reference/sparse_fmha.py", "reference/sparse_attention.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "sparse_gqa_impl_test",
    srcs = ["sparse_gqa_impl_test.py", "reference/sparse_fmha.py", "reference/sparse_attention.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "sparse_paged_fmha_test",
    srcs = ["sparse_paged_fmha_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "draft_prefill_graph_test",
    srcs = ["draft_prefill_graph_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "draft_metadata_test",
    srcs = ["draft_metadata_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "fp8_small_quant_test",
    srcs = ["fp8_small_quant_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "sparse_prefill_bitmap_test",
    srcs = ["sparse_prefill_bitmap_test.py", "reference/sparse_fmha.py", "reference/sparse_attention.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "indexer_prefill_score_test",
    srcs = ["indexer_prefill_score_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "small_qk_norm_test",
    srcs = ["small_qk_norm_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "small_moe_gemv_test",
    srcs = ["small_moe_gemv_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "pd_cache_store_test",
    srcs = ["pd_cache_store_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
)

py_test(
    name = "gdn_composition_test",
    srcs = ["gdn_composition_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "fp8_kv_test",
    srcs = ["fp8_kv_test.py", "sparse_paged_fmha_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    exec_properties = {'gpu':'RTX_5000_PRO_CU13'},
)

py_test(
    name = "grouped_fp8_test",
    srcs = ["grouped_fp8_test.py", "sparse_paged_fmha_test.py"],
    deps = ["//rtp_llm/models_py/standalone:auto_model"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "vocab_embedding_test",
    srcs = ["vocab_embedding_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "upstream_parity_qualification",
    srcs = ["upstream_parity_test.py", "upstream_fixture.py"],
    main = "upstream_parity_test.py",
    deps = ["//rtp_llm:testlib", ":math_references"],
    env = {"QWEN4_EXP_REQUIRE_PARITY": "1"},
    tags = ["manual"],
    exec_properties = select({"@//:using_cuda13_x86": {"gpu": "RTX_5000_PRO_CU13"}, "//conditions:default": {"gpu": "H20"}}),
)

py_test(
    name = "gdn_upstream_parity_qualification",
    srcs = ["gdn_upstream_parity_test.py", "upstream_fixture.py"],
    main = "gdn_upstream_parity_test.py",
    deps = ["//rtp_llm:testlib", ":math_references"],
    env = {"QWEN4_EXP_REQUIRE_PARITY": "1"},
    tags = ["manual"],
    exec_properties = select({"@//:using_cuda13_x86": {"gpu": "RTX_5000_PRO_CU13"}, "//conditions:default": {"gpu": "H20"}}),
)

py_test(
    name = "fp8_dense_gemv_test",
    srcs = ["fp8_dense_gemv_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {"gpu": "SM12x"},
)

py_test(
    name = "ple_outer_residual_test",
    srcs = ["ple_outer_residual_test.py", "ple_gate_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "ple_commit_metadata_test",
    srcs = ["ple_commit_metadata_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "ple_commit_prepare_test",
    srcs = ["ple_commit_prepare_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "ple_commit_test",
    srcs = ["ple_commit_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "grouped_qk_norm_test",
    srcs = ["grouped_qk_norm_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "prefill_qk_norm_test",
    srcs = ["prefill_qk_norm_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "residual_medium_up_mix_test",
    srcs = ["residual_medium_up_mix_test.py", "residual_norm_fp8_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = 'ple_hash_triton_test',
    srcs = ['ple_hash_triton_test.py', 'ple_test.py'],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = 'gdn_prefill_norm_test',
    srcs = ['gdn_prefill_norm_test.py'],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = 'residual_prefill_inject_norm_test',
    srcs = ['residual_prefill_inject_norm_test.py', 'residual_norm_fp8_test.py'],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "ple_gate_test",
    srcs = ["ple_gate_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "residual_separate_epilogue_test",
    srcs = ["residual_separate_epilogue_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "gdn_prefill_adapter_test",
    srcs = ["gdn_prefill_adapter_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "qsa_output_gate_test",
    srcs = ["qsa_output_gate_test.py", "grouped_fp8_test.py", "sparse_paged_fmha_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "residual_pending_test",
    srcs = ["residual_pending_test.py", "residual_norm_fp8_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "indexer_norm_rope_test",
    srcs = ["indexer_norm_rope_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "residual_inject_projection_test",
    srcs = ["residual_inject_projection_test.py", "residual_norm_fp8_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = 'residual_splitk_inject_test',
    srcs = ['residual_splitk_inject_test.py', 'residual_norm_fp8_test.py'],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = 'qsa_packed_count_test',
    srcs = ['qsa_packed_count_test.py', 'grouped_fp8_test.py', 'sparse_paged_fmha_test.py'],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "ple_read_blocks_test",
    srcs = ["ple_read_blocks_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "main_qk_norm_rope_test",
    srcs = ["main_qk_norm_rope_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "qsa_projection_pair_test",
    srcs = ["qsa_projection_pair_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "residual_prefill_up_mix_test",
    srcs = ["residual_prefill_up_mix_test.py", "residual_norm_fp8_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "mtp_embedding_norm_test",
    srcs = ["mtp_embedding_norm_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "qsa_cub_topk_test",
    srcs = ["qsa_cub_topk_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "qsa_target_prepare_test",
    srcs = ["qsa_target_prepare_test.py", "target_writer_fusion_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "tp_registered_input_test",
    srcs = ["tp_registered_input_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "prefetch_order_test",
    srcs = ["prefetch_order_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "decode_snapshot_test",
    srcs = ["decode_snapshot_test.py", "target_writer_fusion_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "main_qk_cache_test",
    srcs = ["main_qk_cache_test.py", "main_qk_norm_rope_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "qsa_main_metadata_test",
    srcs = ["qsa_main_metadata_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "decode_rope_stride_test",
    srcs = ["decode_rope_stride_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "rope_cache_policy_test",
    srcs = ["rope_cache_policy_test.py", "main_qk_cache_test.py", "main_qk_norm_rope_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "target_head_graph_test",
    srcs = ["target_head_graph_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "draft_head_graph_test",
    srcs = ["draft_head_graph_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "vocab_byte_gather_test",
    srcs = ["vocab_byte_gather_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "gdn_norm_quant_test",
    srcs = ["gdn_norm_quant_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "shared_expert_overlap_test",
    srcs = ["shared_expert_overlap_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "head_ipc_gather_test",
    srcs = ["head_ipc_gather_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)


py_test(
    name = "draft_qsa_fused_test",
    srcs = ["draft_qsa_fused_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "draft_qsa_layout_test",
    srcs = ["draft_qsa_layout_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "draft_paged_fmha_test",
    srcs = ["draft_paged_fmha_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "draft_qsa_metadata_test",
    srcs = ["draft_qsa_metadata_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "cpu_target_status_test",
    srcs = ["cpu_target_status_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "RTX_5000_PRO_CU13"},
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
