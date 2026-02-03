py_library(
    name = "fused_moe_executor_test_util",
    srcs = ["fused_moe_executor_test_util.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
    ]
)

py_test (
    name = "deepgemm_masked_executor_test",
    srcs = ["deepgemm_masked_executor_test.py"],
    deps = [
        ":fused_moe_executor_test_util",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'1'},
)

py_test (
    name = "deepgemm_masked_executor_sm100_test",
    srcs = ["deepgemm_masked_executor_test.py",
            "deepgemm_masked_executor_sm100_test.py"],
    data = ["//rtp_llm/models_py:modules/factory/sitecustomize.py"],
    deps = [
        ":fused_moe_executor_test_util",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    env = {
        "GPU_COUNT": "1",
        "PYTHONPATH": "rtp_llm/models_py/modules/factory/",
    },
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {'gpu':'SM100_ARM', 'gpu_count':'1'},
)

py_test (
    name = "deepep_normal_executor_test",
    srcs = ["deepep_normal_executor_test.py"],
    deps = [
        ":fused_moe_executor_test_util",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'1'},
)

py_test (
    name = "deepep_normal_executor_sm100_test",
    srcs = ["deepep_normal_executor_sm100_test.py",
            "deepep_normal_executor_test.py"],
    data = ["//rtp_llm/models_py:modules/factory/sitecustomize.py"],
    deps = [
        ":fused_moe_executor_test_util",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    env = {
        "GPU_COUNT": "1",
        "PYTHONPATH": "rtp_llm/models_py/modules/factory/",
    },
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {'gpu':'SM100_ARM', 'gpu_count':'1'},
)

py_test (
    name = "trtllm_fp4_executor_test",
    srcs = ["trtllm_fp4_executor_test.py"],
    data = ["//rtp_llm/models_py:modules/factory/sitecustomize.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm:flashinfer-python",
        "//rtp_llm:nvidia-cutlass-dsl",
    ],
    env = {
        "GPU_COUNT": "1",
        "PYTHONPATH": "rtp_llm/models_py/modules/factory/",
    },
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {'gpu':'SM100_ARM', 'gpu_count':'1'},
)

py_test (
    name = "cutedsl_fp4_executor_test",
    srcs = ["cutedsl_fp4_executor_test.py"],
    data = ["//rtp_llm/models_py:modules/factory/sitecustomize.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm:flashinfer-python",
    ],
    env = {
        "GPU_COUNT": "1",
        "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        "PYTHONPATH": "rtp_llm/models_py/modules/factory/",
    },
    tags = ["open_skip", "SM100_ARM"],
    exec_properties = {'gpu':'SM100_ARM', 'gpu_count':'1'},
)
