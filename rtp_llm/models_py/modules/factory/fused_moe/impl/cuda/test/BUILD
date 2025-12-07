
test_envs = {
    "DEVICE_RESERVE_MEMORY_BYTES": "512000000",  # 512MB
}

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test (
    name = "fused_moe_test",
    srcs = ["fused_moe_test.py"],
    deps = py_test_deps,
    env = test_envs,
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "fused_moe_op_test",
    srcs = ["fused_moe_op_test.py"],
    deps = py_test_deps,
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

# TODO: Fix this test
# py_test (
#     name = "fused_silu_mul_token_quant_batched_test",
#     srcs = ["fused_silu_mul_token_quant_batched_test.py"],
#     deps = py_test_deps,
#     env = test_envs,
#     tags = ["open_skip", "H20"],
#     exec_properties = {'gpu':'H20'},
# )

py_test (
    name = "moe_ep_reorder_test",
    srcs = ["moe_ep_reorder_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:utils",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util:test_util"
    ],
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

