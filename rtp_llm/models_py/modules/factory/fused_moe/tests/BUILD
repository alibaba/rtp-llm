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
    name = "test_sm120_strategies",
    srcs = ["test_sm120_strategies.py"],
    deps = [
        "//rtp_llm:testlib",
    ],
    tags = ["open_skip", "RTX_5000_PRO"],
    exec_properties = {'gpu':'RTX_5000_PRO'},
)
