# only can run in cu129 so mark as manual
load("//bazel:arch_select.bzl", "requirement")

requirement(["torch", "triton", "flash-mla"])

py_test(
    name = "test_capture_cache_slots",
    srcs = ["test_capture_cache_slots.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)

py_test(
    name = "cp_kv_padded_pack_test",
    srcs = ["cp_kv_padded_pack_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["manual"],
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
)

py_test(
    name = "history_remap_collective_test",
    srcs = ["history_remap_collective_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {"WORLD_SIZE": "2"},
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "2"},
)

py_test(
    name = "attention_copy_fusion_test",
    srcs = ["attention_copy_fusion_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
)

py_test(
    name = "triton_kv_scatter_test",
    srcs = ["triton_kv_scatter_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
)

py_test(
    name = "cp_topk_offsets_cpu_test",
    srcs = ["cp_topk_offsets_test.py"],
    main = "cp_topk_offsets_test.py",
    args = ["CpTopkOffsetsCpuTest"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "cp_attention_metadata_test",
    srcs = ["cp_attention_metadata_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
)

py_test(
    name = "cp_offset_plan_graph_test",
    srcs = ["cp_offset_plan_graph_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
)

py_test(
    name = "cp_topk_offsets_cuda_test",
    srcs = ["cp_topk_offsets_test.py"],
    main = "cp_topk_offsets_test.py",
    args = ["CpTopkOffsetsCudaTest"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
)

py_test(
    name = "cp_kv_overlap_test",
    srcs = ["cp_kv_overlap_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["manual"],
)

py_test(
    name = "layered_prefill_cache_sm103_test",
    srcs = ["layered_prefill_cache_test.py"],
    main = "layered_prefill_cache_test.py",
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["manual"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)

py_test(
    name = "layered_prefill_chunk_compat_sm103_test",
    srcs = ["layered_prefill_cache_test.py"],
    main = "layered_prefill_cache_test.py",
    args = [
        "PrefillStagingLivePagesTest",
        "LayeredPrefillCacheTest.test_production_chunk_staging_budget",
        "LayeredPrefillCacheTest.test_chunk_progress_preserves_history_and_future_pages",
        "LayeredPrefillCacheTest.test_rejects_overflow_and_out_of_range_before_transfer",
        "LayeredPrefillCacheTest.test_partial_pages_slot_reuse_and_recycled_ids",
    ],
    env = {"RTP_LLM_CP_PREFILL_PIPELINE_OPT": "0"},
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["manual"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)

py_test(
    name = "pinned_mla_cache_test",
    srcs = ["pinned_mla_cache_test.py"],
    deps = [
        "//rtp_llm/models_py:pinned_mla_cache",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        ":torch",
        ":triton",
        ":flash-mla",
    ],
    tags = ["manual", "SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)

py_test(
    name = "glm53_prefill_workspace_test",
    srcs = ["glm53_prefill_workspace_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "sparse_mla_op_test",
    srcs = ["sparse_mla_op_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["manual"],
    exec_properties = {'gpu': 'H20'},
)

py_test(
    name = "sparse_mla_fp8_decode_op_test",
    main = "sparse_mla_decode_op_test.py",
    srcs = ["sparse_mla_decode_op_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["manual"],
    exec_properties = {'gpu': 'H20'},
)

py_test(
    name = "flashmla_sparse_cp_op_test",
    srcs = ["flashmla_sparse_cp_op_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["manual"],
    exec_properties = {'gpu': 'H20'},
)

py_test(
    name = "test_target_verify_kernel",
    srcs = ["test_target_verify_kernel.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["manual"],
    exec_properties = {'gpu': 'H20'},
)

py_test(
    name = "test_plan_fp8_no_sync",
    srcs = ["test_plan_fp8_no_sync.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["manual"],
    exec_properties = {'gpu': 'H20'},
)

py_test(
    name = "test_prefill_cuda_graph_kernel",
    srcs = ["test_prefill_cuda_graph_kernel.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["manual"],
    exec_properties = {'gpu': 'H20'},
)

py_test(
    name = "glm53_prefill_q_selection_test",
    srcs = ["glm53_prefill_q_selection_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["manual", "SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)

py_test(
    name = "cp_kv_gather_test",
    srcs = ["cp_kv_gather_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "2"},
)

py_test(
    name = "layered_prefill_cache_test",
    srcs = ["layered_prefill_cache_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["manual"],
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
)

py_test(
    name = "prefill_rollback_test",
    srcs = ["prefill_rollback_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["manual"],
)

py_library(
    name = "cp_kv_prefetch_test_support",
    testonly = True,
    srcs = ["cp_kv_prefetch_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    visibility = ["//rtp_llm/models_py/triton_kernels/common/test:__pkg__"],
)

py_test(
    name = "cp_kv_prefetch_test",
    srcs = ["cp_kv_prefetch_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["manual"],
)

py_test(
    name = "cp_plan_upload_cuda_test",
    srcs = ["cp_plan_upload_cuda_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)

[
    py_test(
        name = "cp_profile_stages_" + mode + "_test",
        srcs = ["cp_profile_stages_test.py"],
        main = "cp_profile_stages_test.py",
        deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
        env = {"RTP_LLM_CP_PROFILE_STAGES": enabled},
        tags = ["SM103_CU13"],
        exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
    )
    for mode, enabled in [("enabled", "1"), ("disabled", "0")]
]

py_test(
    name = "cp_kv_prefetch_gpu_test",
    srcs = ["cp_kv_prefetch_gpu_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    tags = ["manual", "H20", "open_skip"],
)

py_test(
    name = "cp_kv_prefetch_sm103_test",
    srcs = ["cp_kv_prefetch_gpu_test.py"],
    main = "cp_kv_prefetch_gpu_test.py",
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    env = {
        "WORLD_SIZE": "8",
        "GPU_COUNT": "8",
        "KV_TEST_WORLD_SIZE": "8",
        "KV_TEST_PROFILE": "1",
        "RTP_GLM5_PREFILL_OPTIMIZE": "1",
        "RTP_LLM_CP_PREFILL_PIPELINE_OPT": "1",
        "RTP_LLM_CP_PROFILE_STAGES": "1",
    },
    tags = ["manual", "SM103_CU13", "open_skip"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "8"},
)
