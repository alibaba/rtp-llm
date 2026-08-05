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

# Runs in the default wildcard test gates: the assertions need no device, and
# exec_properties alone is enough to schedule the CUDA-linked extension load on a
# GPU host (same pattern as //rtp_llm/models_py/bindings/core/test:cache_store_op_test).
# No open_skip tag on purpose -- that would drop the only coverage of the real
# trace-memory pybind surface, since test_warmup_skew fakes the binding.
py_test(
    name = "test_warmup_bindings",
    srcs = ["test_warmup_bindings.py"],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {"gpu": "H20"},
)
