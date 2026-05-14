# DSV4 FP8 unit tests.
#
# All tests in this directory exercise the FP8 KV-cache path
# (``rtp_llm/models_py/modules/dsv4/fp8/``) and require an H20-class GPU
# for FlashMLA / DeepGEMM kernels.  BF16 path tests stay in the sibling
# ``rtp_llm/models_py/modules/dsv4/test/`` directory.

load("@rules_python//python:defs.bzl", "py_test")

_DSV4_FP8_TEST_DEPS = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
    "//rtp_llm:testlib",
]

_DSV4_FP8_TESTS = [
    "test_compressor_disable_raw_path",
    "test_compressor_forward_flat_input",
    "test_compressor_fp8_per_token",
    "test_compressor_gemm_merge",
    "test_compressor_prepare_metadata_varlen",
    "test_compressor_raw_dispatch",
    "test_compressor_varlen_raw_path",
    "test_dsv4_kv_cache",
    "test_dsv4_persistent_topk",
    "test_dsv4_top_k_per_row_prefill",
    "test_fp8_mqa_prefill_indexer_score",
    "test_fp8_paged_indexer_score",
    "test_indexer_compute_q_flat_input",
    "test_indexer_fp8_prefill_chain",
    "test_indexer_fp8_prefill_meta",
    "test_indexer_fp8_quant_triton",
    "test_indexer_fp8_score",
    "test_indexer_q_fp8_quant",
    "test_swa_combine_topk",
    "test_swa_fp8_kv_roundtrip",
    "test_swa_prefill_gather_lens",
    "test_swa_prefill_varlen",
    "test_swa_slot_mapping",
    "test_varlen_prefill_oracle",
    "test_workspace_prefill_varlen",
]

py_test(
    name = "test_decode_attn_metadata_start_pos",
    srcs = ["test_decode_attn_metadata_start_pos.py"],
    deps = _DSV4_FP8_TEST_DEPS,
)

[
    py_test(
        name = t,
        srcs = [t + ".py"],
        deps = _DSV4_FP8_TEST_DEPS,
        tags = ["H20"],
        exec_properties = {"gpu": "H20"},
    )
    for t in _DSV4_FP8_TESTS
]
