# only can run in cu129 so mark as manual
py_test(
    name = "sparse_mla_stable_decode_test",
    srcs = ["sparse_mla_stable_decode_test.py", "sparse_mla_decode_op_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["H20_CU13", "manual"],
    exec_properties = {"gpu": "H20_CU13"},
)

py_test(
    name = "sparse_mla_op_test",
    srcs = ["sparse_mla_op_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = [
        "H20_CU13",
        "manual",
    ],
    exec_properties = {"gpu": "H20_CU13"},
)

py_test(
    name = "sparse_mla_fp8_decode_op_test",
    srcs = ["sparse_mla_decode_op_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = [
        "H20_CU13",
        "manual",
    ],
    exec_properties = {"gpu": "H20_CU13"},
)

py_test(
    name = "flashmla_sparse_cp_op_test",
    srcs = ["flashmla_sparse_cp_op_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = [
        "H20_CU13",
        "manual",
    ],
    exec_properties = {"gpu": "H20_CU13"},
)

py_test(
    name = "flashmla_sparse_cp_ppu_test",
    main = "flashmla_sparse_cp_op_test.py",
    srcs = ["flashmla_sparse_cp_op_test.py"],
    args = [
        "SparseMlaFp8CPOpTest.test_ppu_large_query_chunks_match_single_query",
        "SparseMlaFp8CPOpTest.test_ppu_fresh_bf16_chunks_preserve_request_boundaries",
        "SparseMlaFp8CPOpTest.test_scheduler_keeps_graph_shape_and_handles_empty_chunk_tails",
        "SparseMlaFp8CPOpTest.test_chunk_tail_positions_and_empty_rank_collectives",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    tags = ["PPU-ZW810E", "manual"],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)
