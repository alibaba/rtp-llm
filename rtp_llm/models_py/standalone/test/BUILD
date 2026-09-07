
test_envs = {}

py_test (
    name = "qwen3_test",
    srcs = ["qwen3_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:auto_model",
    ],
    env = test_envs,
    # Verified on sm86: full auto_model load + forward passes with 0 skips.
)

