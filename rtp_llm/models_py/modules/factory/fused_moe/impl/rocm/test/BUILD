
test_envs = {
    "DEVICE_RESERVE_MEMORY_BYTES": "512000000",  # 512MB
}

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test (
    name = "rocm_fused_moe_test",
    srcs = ["rocm_fused_moe_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util:test_util"
    ],
    timeout = "eternal",
    env = test_envs,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X'},
)

# TODO: fix this test
# py_test (
#     name = "deepep_normal_router_test",
#     srcs = ["deepep_normal_router_test.py"],
#     deps = py_test_deps,
#     timeout = "eternal",
#     env = test_envs,
#     tags = ["rocm"],
#     exec_properties = {'gpu':'MI308X'},
# )

