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

# IndexerFP8.prepare per-call meta builder — geometry/ke/ks/cu_kv_seqlens
# math + warmup early-return path (stub-driven, no FP8 weights needed).
py_test(
    name = "test_indexer_fp8_prefill_meta",
    srcs = ["test_indexer_fp8_prefill_meta.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

# IndexerFP8 prefill kernel chain glue: cp_gather + fp8_mqa_logits +
# dsv4_top_k_per_row_prefill end-to-end. Locks shape/dtype contracts at
# each boundary + the k_scale view(fp32).squeeze(-1) glue.
py_test(
    name = "test_indexer_fp8_prefill_chain",
    srcs = ["test_indexer_fp8_prefill_chain.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

# ── DSV4 SWA prefill Triton kernel UTs ────────────────────────────────────
# Test targets for the vendored vLLM Triton kernels under
# rtp_llm/models_py/modules/dsv4/_swa_*_triton.py. Reuse the same testlib
# the other dsv4 module-level tests use; the kernels themselves only need
# torch + triton, but the import chain
# rtp_llm.models_py.modules.dsv4 → modules → modules.base → rtp_llm.device
# pulls in transitive deps so the slim ":modules" dep is not enough.

_swa_triton_ut_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
    "//rtp_llm:testlib",
]

py_test(
    name = "test_swa_fp8_kv_roundtrip",
    srcs = ["test_swa_fp8_kv_roundtrip.py"],
    deps = _swa_triton_ut_deps,
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "test_swa_slot_mapping",
    srcs = ["test_swa_slot_mapping.py"],
    deps = _swa_triton_ut_deps,
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "test_swa_prefill_gather_lens",
    srcs = ["test_swa_prefill_gather_lens.py"],
    deps = _swa_triton_ut_deps,
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "test_swa_combine_topk",
    srcs = ["test_swa_combine_topk.py"],
    deps = _swa_triton_ut_deps,
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)
