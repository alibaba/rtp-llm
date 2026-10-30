py_library(
    name = "math_references",
    srcs = glob(["reference/*.py"]),
    deps = ["//rtp_llm/models_py:modules"],
)

# Byte-identical vendored copy of transformers' modeling_qwen4_exp.py plus its
# content-addressed manifest: the self-contained truth anchor for parity
# qualification (same convention as committed golden tensors for other models).
VENDORED_UPSTREAM_FIXTURE = [
    "reference/upstream/modeling_qwen4_exp.py",
    "reference/upstream/manifest.json",
]

py_test(
    name = "upstream_fixture_test",
    srcs = ["upstream_fixture_test.py", "upstream_fixture.py"],
    deps = ["//rtp_llm:testlib"],
    data = VENDORED_UPSTREAM_FIXTURE,
)

py_test(
    name = "ple_test",
    srcs = ["ple_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)

py_test(
    name = "ple_gather_triton_test",
    srcs = ["ple_gather_triton_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)

py_test(
    name = "ple_prefetch_test",
    srcs = ["ple_prefetch_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)

py_test(
    name = "ple_inline_uva_graph_test",
    srcs = ["ple_inline_uva_graph_test.py"],
    deps = ["//rtp_llm:testlib"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)

py_test(
    name = "ple_decode_test",
    srcs = ["ple_decode_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)

# Component parity uses the pinned in-repo upstream fixture by default.
py_test(
    name = "upstream_parity_test",
    srcs = ["upstream_parity_test.py", "upstream_fixture.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    data = VENDORED_UPSTREAM_FIXTURE,
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)

# GDN (linear attention) parity on a tiny K != V config.
py_test(
    name = "gdn_upstream_parity_test",
    srcs = ["gdn_upstream_parity_test.py", "upstream_fixture.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    data = VENDORED_UPSTREAM_FIXTURE,
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
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
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)

py_test(
    name = "indexer_prefill_triton_test",
    srcs = ["indexer_prefill_triton_test.py", "scalar_cache_reference.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)

py_test(
    name = "moe_align_triton_test",
    srcs = ["moe_align_triton_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13"},
)

py_test(
    name = "moe_topk_triton_test",
    srcs = ["moe_topk_triton_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13"},
)

py_test(
    name = "marlin_nvfp4_test",
    srcs = ["marlin_nvfp4_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
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
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)

py_test(
    name = "qsa_topk_triton_test",
    srcs = ["qsa_topk_triton_test.py", "topk_reference.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)

py_test(
    name = "qsa_decode_graph_test",
    srcs = ["qsa_decode_graph_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)

py_test(
    name = "indexer_paged_score_test",
    srcs = ["indexer_paged_score_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)

py_test(
    name = "sparse_attention_test",
    srcs = ["sparse_attention_test.py", "reference/sparse_attention.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)

py_test(
    name = "sparse_fmha_test",
    srcs = ["sparse_fmha_test.py", "reference/sparse_fmha.py", "reference/sparse_attention.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13"},
)

py_test(
    name = "sparse_gqa_impl_test",
    srcs = ["sparse_gqa_impl_test.py", "reference/sparse_fmha.py", "reference/sparse_attention.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)

py_test(
    name = "sparse_paged_fmha_test",
    srcs = ["sparse_paged_fmha_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13"},
)

py_test(
    name = "draft_prefill_graph_test",
    srcs = ["draft_prefill_graph_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)

py_test(
    name = "draft_metadata_test",
    srcs = ["draft_metadata_test.py"],
    deps = ["//rtp_llm:testlib"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)

py_test(
    name = "prefill_silu_quant_test",
    srcs = ["prefill_silu_quant_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "sparse_prefill_bitmap_test",
    srcs = ["sparse_prefill_bitmap_test.py", "reference/sparse_fmha.py", "reference/sparse_attention.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13"},
)

py_test(
    name = "indexer_prefill_score_test",
    srcs = ["indexer_prefill_score_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13"},
)

py_test(
    name = "small_moe_gemv_test",
    srcs = ["small_moe_gemv_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13"},
)

py_test(
    name = "pd_cache_store_test",
    srcs = ["pd_cache_store_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)

py_test(
    name = "gdn_composition_test",
    srcs = ["gdn_composition_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)

py_test(
    name = "fp8_kv_test",
    srcs = ["fp8_kv_test.py", "sparse_paged_fmha_test.py"],
    deps = ["//rtp_llm:testlib", ":math_references"],
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13"},
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
    name = "upstream_parity_qualification",
    srcs = ["upstream_parity_test.py", "upstream_fixture.py"],
    main = "upstream_parity_test.py",
    deps = ["//rtp_llm:testlib", ":math_references"],
    data = VENDORED_UPSTREAM_FIXTURE,
    env = {"QWEN4_EXP_REQUIRE_PARITY": "1"},
    tags = ["manual"],
    exec_properties = select({"@//:using_cuda13_x86": {"gpu": "RTX_5000_PRO_CU13"}, "//conditions:default": {"gpu": "H20_CU13"}}),
)

py_test(
    name = "gdn_upstream_parity_qualification",
    srcs = ["gdn_upstream_parity_test.py", "upstream_fixture.py"],
    main = "gdn_upstream_parity_test.py",
    deps = ["//rtp_llm:testlib", ":math_references"],
    data = VENDORED_UPSTREAM_FIXTURE,
    env = {"QWEN4_EXP_REQUIRE_PARITY": "1"},
    tags = ["manual"],
    exec_properties = select({"@//:using_cuda13_x86": {"gpu": "RTX_5000_PRO_CU13"}, "//conditions:default": {"gpu": "H20_CU13"}}),
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
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
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
    name = "ple_gate_test",
    srcs = ["ple_gate_test.py"],
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
    name = "indexer_norm_rope_test",
    srcs = ["indexer_norm_rope_test.py"],
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
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
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
    args = ["HeadIpcGatherTest"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "head_ipc_gather_integration_test",
    srcs = ["head_ipc_gather_test.py"],
    main = "head_ipc_gather_test.py",
    args = ["HeadIpcGatherIntegrationTest"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "2"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "2"},
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
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13"},
)
