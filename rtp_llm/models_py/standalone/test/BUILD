
test_envs = {
    "DEVICE_RESERVE_MEMORY_BYTES": "512000000",  # 512MB
    "LOAD_PYTHON_MODEL": "1",
}

py_test (
    name = "qwen3_test",
    srcs = ["qwen3_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:rtp_simple_model",
    ],
    env = test_envs,
    exec_properties = {'gpu':'A10'},
)

