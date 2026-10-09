py_test(
    name = "test_dspark_proposer",
    srcs = ["test_dspark_proposer.py"],
    main = "test_dspark_proposer.py",
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_minimax_m31_prefill_router",
    srcs = ["test_minimax_m31_prefill_router.py"],
    main = "test_minimax_m31_prefill_router.py",
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["SM100_X86"],
    exec_properties = {"gpu": "SM100_X86", "gpu_count": "1"},
)

py_test(
    name = "test_dspark_reference",
    srcs = ["test_dspark_reference.py"],
    main = "test_dspark_reference.py",
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_dspark_single_simulator",
    srcs = ["test_dspark_single_simulator.py"],
    main = "test_dspark_single_simulator.py",
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_single_dspark_closed_loop",
    srcs = ["test_single_dspark_closed_loop.py"],
    main = "test_single_dspark_closed_loop.py",
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_dspark_runtime_config",
    srcs = ["test_dspark_runtime_config.py"],
    main = "test_dspark_runtime_config.py",
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "test_dspark_verify_budget",
    srcs = ["test_dspark_verify_budget.py"],
    main = "test_dspark_verify_budget.py",
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "test_minimax_m31_dspark_context",
    srcs = ["test_minimax_m31_dspark_context.py"],
    main = "test_minimax_m31_dspark_context.py",
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "test_dspark_kv_commit",
    srcs = ["test_dspark_kv_commit.py"],
    main = "test_dspark_kv_commit.py",
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["SM100_X86"],
    exec_properties = {"gpu": "SM100_X86", "gpu_count": "1"},
)

py_test(
    name = "test_dspark_swa",
    srcs = ["test_dspark_swa.py"],
    main = "test_dspark_swa.py",
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["SM100_X86"],
    exec_properties = {"gpu": "SM100_X86", "gpu_count": "1"},
)

py_test(
    name = "test_dspark_swa_fa4",
    srcs = ["test_dspark_swa_fa4.py"],
    main = "test_dspark_swa_fa4.py",
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["SM100_X86"],
    exec_properties = {"gpu": "SM100_X86", "gpu_count": "1"},
)

py_test(
    name = "test_dspark_runtime_layout",
    srcs = ["test_dspark_runtime_layout.py"],
    main = "test_dspark_runtime_layout.py",
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["SM100_X86"],
    exec_properties = {"gpu": "SM100_X86", "gpu_count": "1"},
)

py_test(
    name = "test_dspark_wheel_import",
    srcs = ["test_dspark_wheel_import.py"],
    main = "test_dspark_wheel_import.py",
    deps = ["//rtp_llm/models_py/triton_kernels:triton_kernels"],
)

py_test(
    name = "test_dspark_gemma_norm",
    srcs = ["test_dspark_gemma_norm.py"],
    main = "test_dspark_gemma_norm.py",
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["SM100_X86"],
    exec_properties = {"gpu": "SM100_X86", "gpu_count": "1"},
)

py_test(
    name = "test_dspark_gemma_rope",
    srcs = ["test_dspark_gemma_rope.py"],
    main = "test_dspark_gemma_rope.py",
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["SM100_X86"],
    exec_properties = {"gpu": "SM100_X86", "gpu_count": "1"},
)
