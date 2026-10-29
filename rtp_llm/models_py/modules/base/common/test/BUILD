py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

test_envs = {}

py_test(
    name = "vocab_parallel_logits_test",
    srcs = ["vocab_parallel_logits_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "2"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "2"},
)

py_test (
    name = "embeding_test",
    srcs = ["embeding_test.py"],
    deps = py_test_deps,
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

py_test (
    name = "multimodal_embedding_test",
    srcs = ["multimodal_embedding_test.py"],
    deps = py_test_deps,
    env = test_envs,
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

py_test(
    name = "common_components_test",
    srcs = ["common_components_test.py"],
    deps = ["//rtp_llm:testlib"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)

py_test(
    name = "zero_centered_norm_test",
    srcs = ["zero_centered_norm_test.py"],
    deps = ["//rtp_llm:testlib"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
)

py_test(
    name = "vocab_embedding_test",
    srcs = ["vocab_embedding_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13"},
)

py_test(
    name = "grouped_qk_norm_test",
    srcs = ["grouped_qk_norm_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "prefill_qk_norm_test",
    srcs = ["prefill_qk_norm_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "vocab_byte_gather_test",
    srcs = ["vocab_byte_gather_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)
