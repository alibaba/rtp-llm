
py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

test_envs = {}

py_test (
    name = "embeding_test",
    srcs = ["embeding_test.py"],
    deps = py_test_deps,
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "kvcache_store_test",
    srcs = ["kvcache_store_test.py"],
    deps = py_test_deps,
    exec_properties = {"gpu": "L20A_DEV"},
)

py_test (
    name = "multimodal_embedding_test",
    srcs = ["multimodal_embedding_test.py"],
    deps = py_test_deps,
    env = test_envs,
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "mtp_multimodal_embedding_test",
    srcs = ["mtp_multimodal_embedding_test.py"],
    deps = py_test_deps,
    exec_properties = {"gpu": "L20A_DEV"},
)
