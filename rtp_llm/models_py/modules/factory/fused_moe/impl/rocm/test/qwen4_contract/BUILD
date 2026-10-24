py_test(
    name = "qwen4_native_fp8_tp4_local_moe_test",
    srcs = [
        "qwen4_native_fp8_tp4_local_moe_test.py",
        "//rtp_llm/models_py/modules/factory/fused_moe/impl/rocm/test:rocm_fp8_fused_moe_test.py",
    ],
    main = "qwen4_native_fp8_tp4_local_moe_test.py",
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    env = {"GPU_COUNT": "1"},
    timeout = "long",
    tags = ["rocm", "manual"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)
