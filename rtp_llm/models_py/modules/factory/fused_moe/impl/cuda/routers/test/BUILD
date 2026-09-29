py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "deepep_normal_router_hook_test",
    srcs = ["deepep_normal_router_hook_test.py"],
    exec_properties = {"gpu": "A10_CU13"},
    # Imports the CUDA router but exercises the extension contract with CPU tensors.
    tags = ["A10_CU13"],
    deps = py_test_deps,
)

py_test(
    name = "deepep_normal_router_test",
    srcs = ["deepep_normal_router_test.py"],
    env = {"GPU_COUNT": "2"},
    exec_properties = {
        "gpu": "H20_CU13",
        "gpu_count": "2",
    },
    tags = [
        "H20_CU13",
        "open_skip",
    ],
    deps = py_test_deps + [
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
)

py_test(
    name = "deepep_low_latency_router_test",
    srcs = ["deepep_low_latency_router_test.py"],
    env = {"GPU_COUNT": "2"},
    exec_properties = {
        "gpu": "H20_CU13",
        "gpu_count": "2",
    },
    tags = [
        "H20_CU13",
        "open_skip",
    ],
    deps = py_test_deps + [
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
)

py_test(
    name = "pure_tp_router_skip_allreduce_test",
    srcs = ["pure_tp_router_skip_allreduce_test.py"],
    exec_properties = {"gpu": "A10_CU13"},
    # The CUDA router imports librtp_compute_ops, so this CPU-tensor contract
    # test still runs in the CUDA image where the runtime library is present.
    tags = ["A10_CU13"],
    deps = py_test_deps,
)
