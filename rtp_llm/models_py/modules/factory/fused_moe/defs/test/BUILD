# The test uses CPU tensors and mocked collectives, but importing FusedMoe
# traverses the device-dispatch package and loads librtp_compute_ops. Schedule
# it on the CUDA test image so the runtime dependency is present.
py_test(
    name = "fused_moe_allreduce_contract_test",
    srcs = ["fused_moe_allreduce_contract_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    tags = ["H20"],
    exec_properties = {'gpu': 'H20'},
)
