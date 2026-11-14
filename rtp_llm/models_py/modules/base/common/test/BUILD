load("//rtp_llm/models_py/test:defs.bzl", "models_py_test")

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

test_envs = {}

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

models_py_test(
    name = "common_components_test",
    src = "common_components_test.py",
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
    name = "vocab_byte_gather_test",
    srcs = ["vocab_byte_gather_test.py"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py/triton_kernels/common/test:vocab_gather_test_support"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

models_py_test(
    name = "fp8_quantization_contract_test",
    src = "fp8_quantization_contract_test.py",
    gpu = "RTX_5000_PRO_CU13",
)

models_py_test(
    name = "fp8_dense_gemv_test",
    src = "fp8_dense_gemv_test.py",
    gpu = "RTX_5000_PRO_CU13",
)
