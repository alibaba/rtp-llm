py_test(
    name = "package_boundaries_test",
    srcs = ["package_boundaries_test.py"],
    deps = [
        "//rtp_llm/utils/lifecycle:control",
        "//rtp_llm/utils/lifecycle:resources",
    ],
)

py_test(
    name = "lifecycle_controller_test",
    srcs = ["lifecycle_controller_test.py"],
    deps = [
        "//rtp_llm:utils",
        "//rtp_llm/utils/lifecycle:control",
    ],
)

py_test(
    name = "backend_shutdown_test",
    srcs = ["backend_shutdown_test.py"],
    deps = ["//rtp_llm/utils/lifecycle:core"],
)

py_test(
    name = "nccl_memory_test",
    srcs = ["nccl_memory_test.py"],
    deps = ["//rtp_llm/utils/lifecycle:resources"],
)

py_test(
    name = "sleep_gpu_reclaim_graph_test",
    srcs = ["sleep_gpu_reclaim_graph_test.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
        "//rtp_llm/utils/lifecycle:resources",
    ],
)
