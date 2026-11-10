test_envs = {}

py_test(
    name = "decoder_cache_test",
    srcs = ["decoder_cache_test.py"],
    deps = ["//rtp_llm/models_py:decoder_cache_helpers"],
    tags = ["SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)

py_test(
    name = "decoder_cache_diagnostics_test",
    srcs = ["decoder_cache_diagnostics_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)

flashinfer_test_envs = {
    "FLASHINFER_JIT_VERBOSE": "1",  # Enable verbose JIT compilation logs
}

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "test_hy4_mxfp8_qc_reuse",
    srcs = ["test_hy4_mxfp8_qc_reuse.py"],
    deps = py_test_deps + ["//rtp_llm:testlib"],
    tags = ["SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)

py_test(
    name = "test_hy4_mxfp8_qc_reuse_baseline",
    srcs = ["test_hy4_mxfp8_qc_reuse.py"],
    main = "test_hy4_mxfp8_qc_reuse.py",
    deps = py_test_deps + ["//rtp_llm:testlib"],
    env = {
        "RTP_LLM_CP_PREFILL_PIPELINE_OPT": "0",
        "RTP_GLM5_PREFILL_OPTIMIZE": "0",
    },
    tags = ["SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)

py_test(
    name = "glm5_cmp_test",
    srcs = ["glm5_cmp_test.py"],
    deps = py_test_deps + [
        "//rtp_llm:testlib",
    ],
)

# FlashInfer dependencies (包括 apache-tvm-ffi 用于 JIT 编译)
flashinfer_deps = [
    "//rtp_llm:apache-tvm-ffi",
    "//rtp_llm:flashinfer-python",
]

py_test(
    name = "mla_chunk_prefill_test",
    main = "mla_reuse_cache_test.py",
    srcs = ["mla_reuse_cache_test.py", "mla_attention_ref.py"],
    deps = py_test_deps + ["//rtp_llm:testlib"] + flashinfer_deps,
    tags = ["manual"],
)

py_test (
    name = "mla_attention_test",
    srcs = ["mla_attention_test.py", "mla_attention_ref.py"],
    deps = py_test_deps + [
        "//rtp_llm:testlib",
    ] + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": []
    }),
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "mla_reuse_cache_test",
    srcs = ["mla_reuse_cache_test.py", "mla_attention_ref.py"],
    deps = py_test_deps + [
        "//rtp_llm:testlib",
    ] + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": []
    }),
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "mla_perf_test",
    srcs = ["mla_perf_test.py", "mla_attention_ref.py"],
    deps = py_test_deps + [
        "//rtp_llm:testlib",
    ],
    tags = ["open_skip", "H20", "manual"],
    exec_properties = {'gpu':'H20'},
)

py_test (
   name = "mlp_test",
   srcs = ["mlp_test.py", "dense_mlp_ref.py"],
   deps = [
       "//rtp_llm/models_py:models",
       "//rtp_llm:config",
       "//rtp_llm:utils",
      "//rtp_llm:testlib",
       "//rtp_llm/test/model_test/test_util:test_util"
   ],
   env = test_envs,
   tags = ["open_skip", "H20"],
   exec_properties = {'gpu':'H20'},
)

py_test (
   name = "flashinfer_jit_test",
   srcs = ["flashinfer_jit_test.py"],
   deps = [
       "//rtp_llm/models_py:models",
       "//rtp_llm:config",
       "//rtp_llm:utils",
      "//rtp_llm:testlib",
       "//rtp_llm/test/model_test/test_util:test_util"
   ],
   env = flashinfer_test_envs,
   tags = ["open_skip", "H20"],
   exec_properties = {'gpu':'H20'},
)

py_test (
   name = "indexer_test",
   srcs = ["indexer_test.py", "indexer_ref.py"],
   deps = py_test_deps + [
        "//rtp_llm:testlib",
   ],
   env = test_envs,
   tags = ["open_skip", "H20"],
   exec_properties = {'gpu':'H20'},
)

py_test(
    name = "mtp_indexer_share_test",
    srcs = ["mtp_indexer_share_test.py"],
    deps = py_test_deps + [
        "//rtp_llm:testlib",
    ],
    tags = ["open_skip"],
)

py_test(
    name = "test_dense_mlp_fp8_silu_fusion",
    srcs = ["test_dense_mlp_fp8_silu_fusion.py"],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {'gpu': 'H20', 'gpu_count': '1'},
)

py_test(
    name = "mtp_mega_buf_share_test",
    srcs = ["mtp_mega_buf_share_test.py"],
    deps = py_test_deps + ["//rtp_llm:testlib"],
)

py_test(name = "indexer_cp_overlap_test", srcs = ["indexer_cp_overlap_test.py"], deps = py_test_deps + ["//rtp_llm:testlib"])
py_test(name = "indexer_cp_overlap_gpu_test", srcs = ["indexer_cp_overlap_gpu_test.py"], deps = py_test_deps + ["//rtp_llm:testlib"], tags = ["manual", "H20", "open_skip"])

py_test(
    name = "indexer_early_history_test",
    srcs = ["indexer_early_history_test.py"],
    deps = py_test_deps + ["//rtp_llm:testlib"],
    tags = ["manual", "H20", "open_skip"],
)

py_test(
    name = "indexer_prologue_prefetch_test",
    srcs = ["indexer_prologue_prefetch_test.py"],
    deps = py_test_deps + ["//rtp_llm:testlib"],
)

py_test(
    name = "mtp_frontend_precision_test",
    srcs = ["mtp_frontend_precision_test.py"],
    deps = py_test_deps + ["//rtp_llm:testlib"],
    tags = ["open_skip", "H20"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "cmp_moe_streams_test",
    srcs = ["cmp_moe_streams_test.py"],
    deps = py_test_deps + ["//rtp_llm:testlib"],
    tags = ["manual", "H20", "open_skip"],
)
