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

# Differential test for run_fused_compress_kv_write's raw-vs-cache dispatch
# (state pool is a 2*256=512-slot cyclic ring — boundaries whose overlap
# falls in the overwritten prefix must read raw kv_flat/score_flat).
py_test(
    name = "test_compressor_raw_dispatch",
    srcs = ["test_compressor_raw_dispatch.py"],
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
    env = {
        "DG_JIT_CACHE_DIR": "/tmp/deep_gemm_jit",
        "PATH": "/usr/local/cuda/bin:/usr/bin:/bin",
    },
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

# CompressorFP8.prepare_metadata under batched (positions, b_idx) — the
# Phase-3a varlen entry point. Stub-driven slot-mapping arithmetic UT
# (no FP8 weights / kernels). Asserts B==1 bit-equality with legacy
# _build_prefill_positions and B==2 per-request routing through bt[b_idx].
py_test(
    name = "test_compressor_prepare_metadata_varlen",
    srcs = ["test_compressor_prepare_metadata_varlen.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

# Phase-3a part 2: byte-equal parity between flat 2D ``[T_total, dim]``
# and legacy 3D ``[1, S, dim]`` input to CompressorFP8.forward. Pins
# down the invariant so a future patch can't silently drift one shape
# off the other.
py_test(
    name = "test_compressor_forward_flat_input",
    srcs = ["test_compressor_forward_flat_input.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

# Phase-3a part 4c: ``run_fused_compress_kv_write`` BATCHED varlen raw
# path byte-equal vs per-request scalar-seq_start launches. Locks down
# the kernel-native varlen path so B>1 batched prefill keeps the raw
# fast path while staying bit-identical to the legacy B==1 launches.
py_test(
    name = "test_compressor_varlen_raw_path",
    srcs = ["test_compressor_varlen_raw_path.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

# Phase-3a part 4a: ``CompressorFP8._launch`` raw_path on/off byte-equal
# parity for short B==1 prefill (seqlen <= state pool capacity). Validates
# that batched prefill can safely fall back to ``seq_start=None`` while
# kernel-level varlen support (option c) is being landed.
py_test(
    name = "test_compressor_disable_raw_path",
    srcs = ["test_compressor_disable_raw_path.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

# Phase-3a part 3: IndexerFP8._compute_indexer_q under flat 2D qr
# input. Validates that wrapping the flat ``[T, H, rope]`` slice with
# a fake ``S=1`` dim before apply_rotary_emb produces byte-equal q
# output to the legacy ``[1, T, H, rope]`` path.
py_test(
    name = "test_indexer_compute_q_flat_input",
    srcs = ["test_indexer_compute_q_flat_input.py"],
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
    env = {
        "DG_JIT_CACHE_DIR": "/tmp/deep_gemm_jit",
        "PATH": "/usr/local/cuda/bin:/usr/bin:/bin",
    },
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

# Phase-0 oracle for the prefill varlen migration. Drives B>1 prefill via
# both per-request and batched paths and asserts equal output. Sub-tests
# are xfail-marked per phase (2 SWA, 3a compressor/indexer, 3b workspace)
# so each Engineer flips just their own marker as the body lands. See
# the test file docstring for the migration plan.
py_test(
    name = "test_varlen_prefill_oracle",
    srcs = ["test_varlen_prefill_oracle.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
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

# Phase-2 SWA-only prefill varlen migration UT. Stub-driven (no FP8
# weights); validates the new _get_window_topk_idxs_varlen helper, the
# varlen branch of _build_swa_prefill_meta (B==1 bit-equality + B>1
# per-request semantics), and the _attn_fp8_swa_via_concat scatter math.
py_test(
    name = "test_swa_prefill_varlen",
    srcs = ["test_swa_prefill_varlen.py"],
    deps = _swa_triton_ut_deps,
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

# Phase-3b CSA / HCA workspace prefill varlen migration UT. Stub-driven;
# validates _build_workspace_meta on (1) legacy B==1 scalar path, (2)
# warmup/unbound short-circuits, (3) varlen B>=2 per-request fields +
# N_max-padded slot_in_flat math, (4) HCA dense_cmp_topk shape/values,
# and (5) end-to-end workspace overlay scatter via index_copy_.
py_test(
    name = "test_workspace_prefill_varlen",
    srcs = ["test_workspace_prefill_varlen.py"],
    deps = _swa_triton_ut_deps,
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)
