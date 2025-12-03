py_test(
    name = "deepep_normal_router_test",
    srcs = ["deepep_normal_router_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm/models_py/distributed/test:process_group_state"
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
        "//rtp_llm/models_py/distributed/test:process_group_state"
    ],
    env = {"GPU_COUNT": "2"},
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'2'},
)
