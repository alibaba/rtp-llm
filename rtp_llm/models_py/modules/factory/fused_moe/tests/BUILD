load("//rtp_llm/models_py/modules/dsv4/test:defs.bzl", "dsv4_py_test")

py_test(
    name = "test_config_resolver",
    srcs = ["test_config_resolver.py"],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "test_strategy_registry",
    srcs = ["test_strategy_registry.py"],
    deps = [
        "//rtp_llm:testlib",
    ],
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
    # sm90. Staying on H20 by decision, not as a TODO: the test is correct about
    # what the executor does, and mocking the capability gate to relocate it would
    # make it assert something the hardware does not do.
    exec_properties = {'gpu':'H20'},
)

# Keep the routing contract in regular CI without importing optional executor
# implementations exercised by the full matrix's priority tests.
py_test(
    name = "test_cuda_strategy_routing",
    srcs = ["test_cuda_strategies.py"],
    main = "test_cuda_strategies.py",
    args = [
        "TestCudaNoQuantFallbackStrategies",
        "TestCudaFp8PerBlockNoDPStrategy.test_can_handle_single_gpu",
        "TestCudaFp8PerBlockNoDPStrategy.test_can_handle_tp_equal_ep",
        "TestCudaFp8PerBlockNoDPStrategy.test_can_handle_false_cuda_graph",
        "TestCudaFp8PerBlockNoDPStrategy.test_rejects_backend_missing_required_executor_symbols",
        "TestCudaFp8PerTensorNoDPStrategy.test_can_handle_fp8_per_tensor_compressed",
        "TestCudaFp8PerTensorNoDPStrategy.test_can_handle_fp8_dynamic_per_tensor",
        "TestCudaW4a8Int4PerChannelNoDPStrategy.test_can_handle_w4a8_int4_per_channel",
        "TestRocmEpStrategyQuantFiltering",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {
        "gpu": "H20",
        "gpu_count": "1",
    },
    tags = ["H20"],
)

py_test(
    name = "test_batched_data_router_ep",
    srcs = ["test_batched_data_router_ep.py"],
    deps = [
        "//rtp_llm:testlib",
    ],
    # The pin needs a matching tag, otherwise ut-sm8x's -H20 filter does not
    # exclude this target and it runs there while being dispatched to an H20
    # worker -- leaking sm8x work onto the pool smoke needs. Keeping upstream's
    # H20 choice, just making the routing agree with it.
    tags = ["H20"],
    exec_properties = {'gpu':'H20'},
)

dsv4_py_test(
    name = "test_strategy_select",
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    visibility = ["//visibility:public"],
)
