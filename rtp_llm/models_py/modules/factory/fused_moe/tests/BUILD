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

# Host-side: pure warmup-skew math + CPU-tensor index shuffling, no GPU required so it runs
# in the default (non-GPU) test gates instead of depending on a GPU smoke run.
py_test(
    name = "test_warmup_skew",
    srcs = ["test_warmup_skew.py"],
    deps = [
        "//rtp_llm:testlib",
    ],
)