py_test(
    name = "gated_residual_test",
    srcs = ["gated_residual_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "residual_stream_test",
    srcs = ["residual_stream_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "ple_test",
    srcs = ["ple_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "ple_decode_test",
    srcs = ["ple_decode_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {'gpu':'A10'},
)

# Skips unless QWEN4_EXP_UPSTREAM_SRC points at transformers'
# modeling_qwen4_exp.py; see the module docstring.
py_test(
    name = "upstream_parity_test",
    srcs = ["upstream_parity_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {'gpu':'A10'},
)

# GDN (linear attention) parity on a tiny K != V config; also skips unless
# QWEN4_EXP_UPSTREAM_SRC is set.
py_test(
    name = "gdn_upstream_parity_test",
    srcs = ["gdn_upstream_parity_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "indexer_score_test",
    srcs = ["indexer_score_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "indexer_compressor_test",
    srcs = ["indexer_compressor_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "indexer_runtime_contract_test",
    srcs = ["indexer_runtime_contract_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "qsa_runtime_test",
    srcs = ["qsa_runtime_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "indexer_paged_score_test",
    srcs = ["indexer_paged_score_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "sparse_attention_test",
    srcs = ["sparse_attention_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "sparse_fmha_test",
    srcs = ["sparse_fmha_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "sparse_gqa_impl_test",
    srcs = ["sparse_gqa_impl_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "sparse_paged_fmha_test",
    srcs = ["sparse_paged_fmha_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {'gpu':'A10'},
)
