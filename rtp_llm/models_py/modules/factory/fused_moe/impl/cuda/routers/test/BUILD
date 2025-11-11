py_test(
    name = "deepep_normal_router_test",
    srcs = ["deepep_normal_router_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm/models_py/distributed:collective_torch"
    ],
    env = {"GPU_COUNT": "2"},
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'2'},
)

py_test (
    name = "deepep_low_latency_router_test",
    srcs = ["deepep_low_latency_router_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm/models_py/distributed:collective_torch"
    ],
    env = {"GPU_COUNT": "2"},
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'2'},
)

py_test (
    name = "fused_moe_afd_test",
    srcs = ["fused_moe_afd_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:utils",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util:test_util"
    ],
    env = test_envs,
    exec_properties = {'gpu':'H20'},
)
