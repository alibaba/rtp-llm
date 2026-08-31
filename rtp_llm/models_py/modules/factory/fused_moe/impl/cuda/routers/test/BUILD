py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "deepep_normal_router_hook_test",
    srcs = ["deepep_normal_router_hook_test.py"],
    deps = py_test_deps,
    # Imports the CUDA router but exercises the extension contract with CPU tensors.
    tags = ["H20"],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "deepep_normal_router_test",
    srcs = ["deepep_normal_router_test.py"],
    deps = py_test_deps + [
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
    env = {"GPU_COUNT": "2"},
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'2'},
)

py_test (
    name = "deepep_low_latency_router_test",
    srcs = ["deepep_low_latency_router_test.py"],
    deps = py_test_deps + [
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
    env = {"GPU_COUNT": "2"},
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'2'},
)

py_test (
    name = "pure_tp_router_skip_allreduce_test",
    srcs = ["pure_tp_router_skip_allreduce_test.py"],
    deps = py_test_deps,
    # The CUDA router imports librtp_compute_ops, so this CPU-tensor contract
    # test still runs in the CUDA image where the runtime library is present.
    tags = ["H20"],
    exec_properties = {'gpu':'H20'},
)
