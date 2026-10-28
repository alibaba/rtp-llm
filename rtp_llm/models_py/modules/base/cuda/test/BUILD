
py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test (
    name = "norm_test",
    srcs = ["norm_test.py"],
    deps = py_test_deps,
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "rmsresnorm_test",
    srcs = ["rmsresnorm_test.py"],
    deps = py_test_deps,
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "fusedqkrmsnorm_test",
    srcs = ["fusedqkrmsnorm_test.py"],
    deps = py_test_deps,
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "fusedqkrmsnorm_sm120_test",
    srcs = ["fusedqkrmsnorm_test.py"],
    main = "fusedqkrmsnorm_test.py",
    args = ["FusedQKRMSNormCudaVec8ExactTest"],
    env = {"FLASHINFER_USE_CUDA_NORM": "1"},
    deps = py_test_deps,
    tags = ["RTX_5000_PRO"],
    exec_properties = {"gpu": "RTX_5000_PRO"},
)

py_test(
    name = "fusedqkrmsnorm_ppu_test",
    srcs = ["fusedqkrmsnorm_test.py"],
    main = "fusedqkrmsnorm_test.py",
    args = ["FusedQKRMSNormPpuTest"],
    deps = py_test_deps,
    tags = ["manual", "PPU-ZW810E"],
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
)

py_test (
    name = "layernorm_test",
    srcs = ["layernorm_test.py"],
    deps = py_test_deps,
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "fused_add_layernorm_test",
    srcs = ["fused_add_layernorm_test.py"],
    deps = py_test_deps,
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "fused_bias_gelu_test",
    srcs = ["fused_bias_gelu_test.py"],
    deps = py_test_deps,
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "select_topk_op_test",
    srcs = ["select_topk_op_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "group_topk_test",
    srcs = ["group_topk_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)


py_test (
    name = "torch_symm_mem_test",
    srcs = ["torch_symm_mem_test.py"],
    deps = py_test_deps + [
        "//rtp_llm/models_py/distributed:collective_torch",
        "//rtp_llm/test/utils:test_util",
    ],
    env = {
        "TP_SIZE": "4",
        "WORLD_SIZE": "4",
    },
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "get_topk_ragged_cp_test",
    srcs = ["get_topk_ragged_cp_test.py"],
    deps = py_test_deps,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)
