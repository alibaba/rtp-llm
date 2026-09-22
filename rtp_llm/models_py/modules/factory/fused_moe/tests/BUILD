py_test(
    name = "test_config_resolver",
    srcs = ["test_config_resolver.py"],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "test_cuda_strategies",
    srcs = ["test_cuda_strategies.py"],
    deps = [
        "//rtp_llm:testlib",
    ],
    tags = ["open_skip"],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "test_batched_data_router_ep",
    srcs = ["test_batched_data_router_ep.py"],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'H20'},
)
