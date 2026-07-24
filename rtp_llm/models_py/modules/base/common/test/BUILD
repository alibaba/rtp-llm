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
<<<<<<< HEAD
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)
=======
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "multimodal_embedding_prefix_cache_test",
    srcs = ["multimodal_embedding_prefix_cache_test.py"],
    deps = py_test_deps,
)
>>>>>>> df99887ad (feat(loader): add Qwen3-VL dense newloader (#1216))
