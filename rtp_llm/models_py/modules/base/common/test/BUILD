py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

test_envs = {}

py_test(
    name = "embeding_test",
    srcs = ["embeding_test.py"],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = py_test_deps,
)

py_test(
    name = "multimodal_embedding_test",
    srcs = ["multimodal_embedding_test.py"],
    env = test_envs,
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = py_test_deps,
)
