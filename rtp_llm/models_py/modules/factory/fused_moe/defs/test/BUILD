py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

# The test uses CPU tensors and mocked collectives, but importing
# models_py.modules reaches the compiled librtp_compute_ops module.  Run it on
# accelerator images that provide that dependency instead of claiming that a
# default CPU runner can import it safely.
py_test(
    name = "fused_moe_allreduce_contract_test",
    srcs = ["fused_moe_allreduce_contract_test.py"],
    deps = py_test_deps + [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'1'},
)

py_test(
    name = "fused_moe_allreduce_contract_test_rocm",
    srcs = ["fused_moe_allreduce_contract_test.py"],
    main = "fused_moe_allreduce_contract_test.py",
    deps = py_test_deps + [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7', 'gpu_count':'1'},
)
