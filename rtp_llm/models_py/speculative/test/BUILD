# DSV4 python tests target the CUDA13/Blackwell stack (deep_gemm >= 2.6, dsv4
# kernels, pytest closure). They are skipped as incompatible on other platforms
# instead of failing on pools that can never satisfy those dependencies.
_DSV4_CUDA13_ONLY = select({
    "@//:using_cuda13_x86": [],
    "@//:using_cuda13_arm": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

py_test(
    name = "test_dspark_proposer",
    srcs = ["test_dspark_proposer.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_qwen3_dspark",
    srcs = ["test_qwen3_dspark.py"],
    deps = [
        "//rtp_llm:async_model",
        "//rtp_llm:models",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "test_qwen3_dflash",
    srcs = ["test_qwen3_dflash.py"],
    deps = [
        "//rtp_llm:async_model",
        "//rtp_llm:models",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "test_qwen3_dflash_weights",
    srcs = ["test_qwen3_dflash_weights.py"],
    deps = [
        "//rtp_llm:async_model",
        "//rtp_llm:models",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "test_qwen3_dspark_config",
    srcs = ["test_qwen3_dspark_config.py"],
    deps = [
        "//rtp_llm:async_model",
        "//rtp_llm:models",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "test_qwen3_dspark_weights",
    srcs = ["test_qwen3_dspark_weights.py"],
    deps = [
        "//rtp_llm:async_model",
        "//rtp_llm:models",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "test_qwen3_dflash2",
    srcs = ["test_qwen3_dflash2.py"],
    deps = [
        "//rtp_llm:async_model",
        "//rtp_llm:models",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "test_qwen3_dflash2_weights",
    srcs = ["test_qwen3_dflash2_weights.py"],
    deps = [
        "//rtp_llm:async_model",
        "//rtp_llm:models",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "dflash2_selector_contract_test",
    srcs = ["test_dflash2_selector.py"],
    main = "test_dflash2_selector.py",
    args = ["DFlash2SelectorContractTest"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "dflash2_selector_cuda_test",
    srcs = ["test_dflash2_selector.py"],
    main = "test_dflash2_selector.py",
    args = ["DFlash2SelectorGPUTest"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20", "gpu_count": "1"},
)

py_test(
    name = "dflash2_selector_rocm_test",
    srcs = ["test_dflash2_selector.py"],
    main = "test_dflash2_selector.py",
    args = ["DFlash2SelectorGPUTest"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "MI308X-ROCM7", "gpu_count": "1"},
)
