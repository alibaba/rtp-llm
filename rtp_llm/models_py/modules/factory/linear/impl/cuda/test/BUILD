py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "collective_projection_gemm_rs_test",
    srcs = ["collective_projection_gemm_rs_test.py"],
    data = ["//rtp_llm/models_py:collective_projection_gemm_rs_test_sources"],
    deps = ["//rtp_llm:torch"],
    env = {"GPU_COUNT": "0"},
    timeout = "short",
)

py_test(
    name = "bf16_collective_projection_test",
    srcs = ["bf16_collective_projection_test.py"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules"],
    env = {"GPU_COUNT": "2"},
    tags = ["SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "2"},
)

py_test(
    name = "bf16_attention_hooks_test",
    srcs = ["bf16_attention_hooks_test.py"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules"],
    env = {"GPU_COUNT": "2"},
    tags = ["SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "2"},
    timeout = "long",
)

py_test(
    name = "bf16_comm_microbenchmark_tp2",
    srcs = ["bf16_comm_microbenchmark.py"],
    main = "bf16_comm_microbenchmark.py",
    args = ["--world-size=2", "--phases=decode", "--samples=8", "--repeat=3"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/test/utils:hang_watchdog",
    ],
    env = {"GPU_COUNT": "2"},
    tags = ["manual", "SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "2"},
    timeout = "long",
)

py_test(
    name = "bf16_comm_microbenchmark_tp8",
    srcs = ["bf16_comm_microbenchmark.py"],
    main = "bf16_comm_microbenchmark.py",
    args = ["--world-size=8", "--profile"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/test/utils:hang_watchdog",
    ],
    env = {"GPU_COUNT": "8"},
    tags = ["manual", "SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "8"},
    timeout = "long",
)

py_test(
    name = "kda_norm_layout_test",
    srcs = ["kda_norm_layout_test.py"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules"],
    env = {"KDA_NORM_PROFILE": "1"},
    tags = ["SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
    timeout = "long",
)

py_test(
    name = "fp8_deepgemm_linear_out_test",
    srcs = ["fp8_deepgemm_linear_out_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
    # Imports the real kernel chain (rtp_llm.ops.compute_ops ->
    # librtp_compute_ops), so it needs the full testlib and a GPU pool.
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

py_test(
    name = "fp8_linear_test",
    srcs = [
        "fp8_linear_test.py",
        "fp8_linear_test_launcher.py",
    ],
    main = "fp8_linear_test_launcher.py",
    deps = py_test_deps,
    tags = ["H20_CU13"],
    exec_properties = {"gpu": "H20_CU13"},
)

py_test(
    name = "fp8_deepgemm_linear_quantized_input_contract_test",
    srcs = ["fp8_deepgemm_linear_quantized_input_contract_test.py"],
    deps = py_test_deps,
)

py_test(
    name = "fp8_deepgemm_linear_sm100_arm_test",
    srcs = [
        "fp8_deepgemm_linear_sm100_arm_test.py",
        "fp8_deepgemm_linear_sm120_test.py",
        "fp8_linear_test.py",
    ],
    deps = py_test_deps,
    tags = ["SM100_ARM_CU13"],
    exec_properties = {"gpu": "SM100_ARM_CU13"},
)

py_test(
    name = "fp8_deepgemm_linear_sm120_test",
    srcs = [
        "fp8_deepgemm_linear_sm120_test.py",
        "fp8_linear_test.py",
    ],
    deps = py_test_deps,
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13"},
)

py_test(
    name = "fp4_gemm_linear_test",
    srcs = ["fp4_gemm_linear_test.py"],
    deps = py_test_deps,
    tags = ["SM100_ARM_CU13"],
    exec_properties = {"gpu": "SM100_ARM_CU13"},
)
