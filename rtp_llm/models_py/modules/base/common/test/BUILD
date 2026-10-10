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

py_test (
    name = "apply_input_embeddings_test",
    srcs = ["apply_input_embeddings_test.py"],
    deps = py_test_deps,
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
)

py_test (
    name = "supports_input_embeddings_test",
    srcs = ["supports_input_embeddings_test.py"],
    deps = py_test_deps,
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
)

py_test(
    name = "apply_input_embeddings_multigpu_test",
    srcs = ["apply_input_embeddings_multigpu_test.py", "apply_input_embeddings_test.py"],
    deps = py_test_deps,
    env = {"GPU_COUNT": "2"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "2"},
    tags = ["H20_CU13"],
)
