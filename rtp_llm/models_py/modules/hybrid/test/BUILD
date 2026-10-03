test_envs = {}

flashinfer_test_envs = {
    "FLASHINFER_JIT_VERBOSE": "1",  # Enable verbose JIT compilation logs
}

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

# FlashInfer dependencies (包括 apache-tvm-ffi 用于 JIT 编译)
flashinfer_deps = [
    "//rtp_llm:apache-tvm-ffi",
    "//rtp_llm:flashinfer-python",
]

py_test(
    name = "mla_attention_test",
    srcs = [
        "mla_attention_ref.py",
        "mla_attention_test.py",
    ],
    exec_properties = {"gpu": "A10_CU13"},
    # Verified on sm86: passes with 0 skips via the FlashInfer MLA fa2 fallback.
    tags = [
        "A10_CU13",
        "open_skip",
    ],
    deps = py_test_deps + [
        "//rtp_llm:testlib",
    ] + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": [],
    }),
)

py_test(
    name = "mla_reuse_cache_test",
    srcs = [
        "mla_attention_ref.py",
        "mla_reuse_cache_test.py",
    ],
    exec_properties = {"gpu": "A10_CU13"},
    # Verified on sm86: passes with 0 skips via the FlashInfer MLA fa2 fallback.
    tags = [
        "A10_CU13",
        "open_skip",
    ],
    deps = py_test_deps + [
        "//rtp_llm:testlib",
    ] + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": [],
    }),
)

py_test(
    name = "mla_perf_test",
    srcs = [
        "mla_attention_ref.py",
        "mla_perf_test.py",
    ],
    exec_properties = {"gpu": "H20_CU13"},
    tags = [
        "H20_CU13",
        "manual",
        "open_skip",
    ],
    deps = py_test_deps + [
    ],
)

py_test(
    name = "mlp_test",
    srcs = [
        "dense_mlp_ref.py",
        "mlp_test.py",
    ],
    env = test_envs,
    # Back on H20 after measuring on sm86: 1 of 40 subTest combinations fails --
    # (num_tokens=83, hidden_size=8192, bfloat16) on allclose(atol=1e-2,
    # rtol=1e-2). Deterministic under manual_seed(0) and identical across two
    # sm86 runs, and it passes on H20, so this is fused-vs-reference bf16
    # accumulation order at that one shape, not flakiness. Staying on H20 by
    # decision rather than widening the tolerance to relocate it.
    exec_properties = {"gpu": "H20_CU13"},
    tags = [
        "H20_CU13",
        "open_skip",
    ],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm:utils",
        "//rtp_llm/models_py:models",
        "//rtp_llm/test/model_test/test_util",
    ],
)

py_test(
    name = "flashinfer_jit_test",
    srcs = ["flashinfer_jit_test.py"],
    env = flashinfer_test_envs,
    exec_properties = {"gpu": "H20_CU13"},
    # Must stay on H20: an sm86 probe showed warmup still succeeds, but
    # flashinfer_mla.py:37,56 `continue` past the fa3 backend when
    # is_sm90a_supported() is false, so 2 of the 4 warmed modules silently vanish
    # with no skip marker -- a green run with deleted coverage.
    tags = [
        "H20_CU13",
        "open_skip",
    ],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm:utils",
        "//rtp_llm/models_py:models",
        "//rtp_llm/test/model_test/test_util",
    ],
)

py_test(
    name = "indexer_test",
    srcs = [
        "indexer_ref.py",
        "indexer_test.py",
    ],
    env = test_envs,
    exec_properties = {"gpu": "H20_CU13"},
    tags = [
        "H20_CU13",
        "open_skip",
    ],
    deps = py_test_deps + [
        "//rtp_llm:testlib",
    ],
)
