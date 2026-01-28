py_library(
    name = "fused_moe_router_test_util",
    srcs = ["fused_moe_router_test_util.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ]
)

py_test(
    name = "deepep_normal_router_test",
    srcs = ["deepep_normal_router_test.py"],
    deps = [
        ":fused_moe_router_test_util",
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
        ":fused_moe_router_test_util",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm/models_py/distributed:collective_torch"
    ],
    env = {"GPU_COUNT": "2"},
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'2'},
)

py_test (
    name = "deepep_low_latency_peo_router_test",
    srcs = ["deepep_low_latency_peo_router_test.py"],
    deps = [
        ":fused_moe_router_test_util",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm/models_py/distributed:collective_torch"
    ],
    env = {"GPU_COUNT": "2"},
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'2'},
)
