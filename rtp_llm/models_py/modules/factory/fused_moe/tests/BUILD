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

# The assertions are pure math + CPU-tensor shuffling, but the target still needs
# a GPU host: importing anything under rtp_llm.models_py.modules routes through
# compute_ops' arch dispatch, which requires torch.cuda.is_available() in the CUDA
# build (verified: with CUDA_VISIBLE_DEVICES="" the import fails with "cannot
# import name 'FusedRopeKVCacheDecodeOp'"). Same basis as test_config_resolver.
py_test(
    name = "test_warmup_skew",
    srcs = ["test_warmup_skew.py"],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {"gpu": "H20"},
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
