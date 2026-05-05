py_test(
    name = "mhc_test",
    srcs = ["mhc_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm/models_py:tile_kernels_mhc",
        "//rtp_llm:testlib",
    ],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "test_dsv4_kv_cache",
    srcs = ["test_dsv4_kv_cache.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

# P0 cleanup gate: v4_indexer_score is now the only path producing the
# indexer score (REF branch removed in indexer.py).  Locks down kernel
# math + perf vs eager.
py_test(
    name = "test_v4_indexer_score",
    srcs = ["test_v4_indexer_score.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

# P0 cleanup gate: v4_compressor_pool is now the only path producing the
# compressor pool output (REF branch removed in compressor.py).
py_test(
    name = "test_v4_compressor_pool",
    srcs = ["test_v4_compressor_pool.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

# PP4 wireup: Compressor → fused {pool+RMSNorm+RoPE+FP8 quant+scatter}
# round-trip test. Locks the env-gated FP8 fast path for both prefill and
# the boundary decode case against the bf16 4-launch reference.
py_test(
    name = "test_compressor_fused_wireup",
    srcs = ["test_compressor_fused_wireup.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

# P1 — vLLM-vendored persistent radix-select TopK for the indexer decode
# hot path.  Replaces torch.topk (~30us → ~5-10us at K=512/T=2048).
# Auto-skipped via pytest.mark.skipif when the .so binding is missing.
py_test(
    name = "test_dsv4_persistent_topk",
    srcs = ["test_dsv4_persistent_topk.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

# vLLM-vendored per-row TopK for the DSv4 indexer prefill (hybrid
# insertion-sort + radix-select).  Auto-skipped when the .so binding is
# missing.
py_test(
    name = "test_dsv4_top_k_per_row_prefill",
    srcs = ["test_dsv4_top_k_per_row_prefill.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

# DeepGEMM FP8 non-paged MQA logits for indexer prefill — equivalence vs
# bf16 ``v4_indexer_score`` reference. Auto-skipped when ``deep_gemm`` is
# unavailable.
py_test(
    name = "test_fp8_mqa_prefill_indexer_score",
    srcs = ["test_fp8_mqa_prefill_indexer_score.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)
