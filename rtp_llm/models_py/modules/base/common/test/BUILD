
test_envs = {
    "DEVICE_RESERVE_MEMORY_BYTES": "512000000",  # 512MB
}

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test (
    name = "embeding_test",
    srcs = ["embeding_test.py"],
    deps = py_test_deps,
    env = test_envs,
    exec_properties = {'gpu':'A10'},
)


py_test (
    name = "multimodal_embedding_test",
    srcs = ["multimodal_embedding_test.py"],
    deps = py_test_deps,
    env = test_envs,
    exec_properties = {'gpu':'H20'},
)