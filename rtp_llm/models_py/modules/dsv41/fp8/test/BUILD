load("@pip_dsv4_test//:requirements.bzl", test_requirement = "requirement")

package(default_visibility = ["//visibility:public"])

_TESTS = [
    "test_v41_attention_pool_warmup",
    "test_attention_v41",
    "test_attention_v41_cp_raw_groups",
    "test_attention_v41_cuda_graph",
    "test_attention_v41_prefill_producer",
    "test_attention_v41_prefill_selection",
    "test_attention_v41_request_rows",
    "test_attention_v41_sparse_selection",
    "test_v41_attention_jit_warmup",
    "test_v41_batched_producer",
    "test_v41_candidate_topk",
    "test_v41_decode_attention_fusion",
    "test_v41_decode_global",
    "test_v41_decode_indexer",
    "test_v41_decode_preprocessing",
    "test_v41_decode_qkv_integration",
    "test_v41_deepselect",
    "test_v41_fp4_codec",
    "test_v41_flash_mla_cache_format",
    "test_v41_fp4_transport",
    "test_v41_global_overlap",
    "test_v41_grouped_gemm",
    "test_v41_grouped_prefill_score",
    "test_v41_indexer_q_fusion",
    "test_v41_joint_pool",
    "test_v41_norm_quant",
    "test_v41_prefill_candidates",
    "test_v41_prefill_global",
    "test_v41_prefill_index_plan",
    "test_v41_prefill_indexer",
    "test_v41_prefill_kv_workspace",
    "test_v41_prefill_meta_cache",
    "test_v41_prefill_meta_cache_cuda",
    "test_v41_prefill_metadata",
    "test_v41_prefill_pools",
    "test_v41_prefill_q_memory",
    "test_v41_prefill_topk",
    "test_v41_producer_metadata",
    "test_v41_qkv_rope_cache",
    "test_v41_selection_oracle",
    "test_v41_sparse_prefill_indexer",
    "test_v41_swa_codec",
    "test_v41_swa_fresh",
    "test_v41_wo_a_quant",
]

[
    py_test(
        name = name,
        srcs = glob(["*.py"]) + ["//rtp_llm/test/pytest:pytest_main.py", "//rtp_llm/models_py/modules/dsv4/fp8/test:test_prefill_common_metadata_cuda.py"],
        main = "//rtp_llm/test/pytest:pytest_main.py",
        args = ["--cuda-devices", "1", "$(location :" + name + ".py)"],
        deps = [
            "//rtp_llm:testlib",
            "//rtp_llm/models_py:modules",
            "//rtp_llm/models_py:tile_kernels_mhc",
            "//rtp_llm:jit_cache_env",
            test_requirement("pytest"),
        ],
        tags = ["dsv41_cuda13", "SM100_ARM_CU13"],
        exec_properties = {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
        target_compatible_with = select({
            "@//:using_cuda13_arm": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
        legacy_create_init = 0,
    )
    for name in _TESTS
]
