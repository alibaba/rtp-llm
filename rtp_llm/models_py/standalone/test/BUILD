
test_envs = {}

py_test (
    name = "qwen3_test",
    srcs = ["qwen3_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:auto_model",
    ],
    env = test_envs,
    tags = ["H20"],
    exec_properties = {'gpu':'H20'},
)

