_DSV4_TEST_DEPS = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
    "//rtp_llm:testlib",
]

py_test(
    name = "test_cp_context_build",
    srcs = ["test_cp_context_build.py"],
    deps = _DSV4_TEST_DEPS,
)

py_test(
    name = "test_cp_topk_idxs_align",
    srcs = ["test_cp_topk_idxs_align.py"],
    deps = _DSV4_TEST_DEPS,
)

py_test(
    name = "test_cp_via_concat_meta",
    srcs = ["test_cp_via_concat_meta.py"],
    deps = _DSV4_TEST_DEPS,
)

py_test(
    name = "test_cp_indexer_seq_total",
    srcs = ["test_cp_indexer_seq_total.py"],
    deps = _DSV4_TEST_DEPS,
)

py_test(
    name = "test_cp_workspace_combine_topk",
    srcs = ["test_cp_workspace_combine_topk.py"],
    deps = _DSV4_TEST_DEPS,
)

py_library(
    name = "dsv4_test_utils",
    srcs = ["dsv4_test_utils.py"],
    deps = _DSV4_TEST_DEPS,
)

py_library(
    name = "dsv4_kernel_perf_utils",
    srcs = ["dsv4_kernel_perf_utils.py"],
)

py_test(
    name = "test_dsv4_compressor",
    srcs = ["test_dsv4_compressor.py"],
    deps = [":dsv4_test_utils"] + _DSV4_TEST_DEPS,
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "test_dsv4_indexer",
    srcs = ["test_dsv4_indexer.py"],
    deps = [":dsv4_test_utils"] + _DSV4_TEST_DEPS,
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "test_dsv4_attention",
    srcs = ["test_dsv4_attention.py"],
    deps = [":dsv4_test_utils"] + _DSV4_TEST_DEPS,
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

test_suite(
    name = "test_dsv4_kv_cache",
    tests = [
        ":test_dsv4_attention",
        ":test_dsv4_compressor",
        ":test_dsv4_indexer",
    ],
)

py_test(
    name = "test_cp_async",
    srcs = ["test_cp_async.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "test_metadata_triton",
    srcs = ["test_metadata_triton.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)

py_test(
    name = "test_compressor_hot_paths",
    srcs = ["test_compressor_hot_paths.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "test_hc_impl",
    srcs = ["test_hc_impl.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py:tile_kernels_mhc",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "test_strategy_select",
    srcs = ["test_strategy_select.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "test_chunked_moe",
    srcs = ["test_chunked_moe.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "test_indexer_topk_backend",
    srcs = ["test_indexer_topk_backend.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)

py_test(
    name = "test_shared_expert_executor",
    srcs = ["test_shared_expert_executor.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)

py_test(
    name = "test_mega_moe_input_packer",
    srcs = ["test_mega_moe_input_packer.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)

py_test(
    name = "test_dsv4_kernel_jit_warmup",
    srcs = ["test_dsv4_kernel_jit_warmup.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "test_mega_moe_input_packer_perf",
    srcs = ["test_mega_moe_input_packer_perf.py"],
    deps = ["//rtp_llm/models_py:dsv4_fused_kernel_sources"],
)

py_test(
    name = "test_grouped_fp4_strategy_perf",
    srcs = ["test_grouped_fp4_strategy_perf.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)

py_test(
    name = "test_fused_inv_rope_fp8_quant",
    srcs = ["test_fused_inv_rope_fp8_quant.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "test_fused_rmsnorm_rope",
    srcs = ["test_fused_rmsnorm_rope.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "test_rope_only",
    srcs = ["test_rope_only.py"],
    deps = [
        "//rtp_llm/models_py:dsv4_fused_kernel_sources",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "test_fused_inv_rope_fp8_quant_perf",
    srcs = ["test_fused_inv_rope_fp8_quant_perf.py"],
    deps = [
        ":dsv4_kernel_perf_utils",
        "//rtp_llm/models_py:dsv4_fused_kernel_sources",
    ],
    tags = ["manual", "H20", "SM100_ARM"],
)

py_test(
    name = "test_fused_rmsnorm_rope_perf",
    srcs = ["test_fused_rmsnorm_rope_perf.py"],
    deps = [
        ":dsv4_kernel_perf_utils",
        "//rtp_llm/models_py:dsv4_fused_kernel_sources",
    ],
    tags = ["manual", "H20", "SM100_ARM"],
)

py_test(
    name = "test_fused_kv_compress_norm_rope_insert_perf",
    srcs = ["test_fused_kv_compress_norm_rope_insert_perf.py"],
    deps = [
        ":dsv4_kernel_perf_utils",
        "//rtp_llm/models_py:dsv4_fused_kernel_sources",
    ],
    tags = ["manual", "H20", "SM100_ARM"],
)

py_test(
    name = "test_rope_only_perf",
    srcs = ["test_rope_only_perf.py"],
    deps = [
        ":dsv4_kernel_perf_utils",
        "//rtp_llm/models_py:dsv4_fused_kernel_sources",
    ],
    tags = ["manual", "H20", "SM100_ARM"],
)

py_test(
    name = "test_cp_restore_perf",
    srcs = ["test_cp_restore_perf.py"],
    deps = [
        ":dsv4_kernel_perf_utils",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["manual", "H20", "SM100_ARM"],
)

py_test(
    name = "test_compressor_metadata_perf",
    srcs = ["test_compressor_metadata_perf.py"],
    deps = [
        ":dsv4_kernel_perf_utils",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["manual", "H20", "SM100_ARM"],
)

py_test(
    name = "test_dsv4_tc_simt_overlap_perf",
    srcs = ["test_dsv4_tc_simt_overlap_perf.py"],
    deps = [
        "//rtp_llm/models_py:dsv4_fused_kernel_sources",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["manual", "SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)

py_test(
    name = "test_weight_conversion_contract",
    srcs = ["test_weight_conversion_contract.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
)
