
test_envs = {
    "DEVICE_RESERVE_MEMORY_BYTES": "512000000",  # 512MB
}

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "test_concat_and_cache_mla",
    srcs = ["test_concat_and_cache_mla.py"],
    deps = py_test_deps,
    env = test_envs,
    tags = ["H20"],
    exec_properties = {'gpu': 'H20'},
)