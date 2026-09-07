py_test(
    name = "test_config_resolver",
    srcs = ["test_config_resolver.py"],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "test_cuda_strategies",
    srcs = ["test_cuda_strategies.py"],
    deps = [
        "//rtp_llm:testlib",
    ],
    tags = ["open_skip", "H20"],
    # Back on H20 after measuring on sm86: 42 tests run, 4 fail on
    # assertTrue(strategy.can_handle(config)). The test mocks has_deep_gemm but
    # not the capability gate, and DeepGemmMaskedExecutorV2.check_conditions has
    # checker.check(get_sm()[0] >= 9) (deepgemm_masked_executor_v2.py:41), so
    # every DeepGemmMaskedExecutorV2-backed strategy correctly declines below
    # sm90. Mock the SM gate too and this can move.
    exec_properties = {'gpu':'H20'},
)
