py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "test_stacked_moe_weight",
    srcs = ["test_stacked_moe_weight.py"],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

py_test(
    name = "test_compressed_w4a8_int4_per_channel",
    srcs = ["test_compressed_w4a8_int4_per_channel.py"],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

py_test(
    name = "test_per_channel_fp8_helpers",
    srcs = ["test_per_channel_fp8_helpers.py"],
    deps = py_test_deps,
)

py_test(
    name = "weight_memory_saver_test",
    srcs = ["weight_memory_saver_test.py"],
    deps = py_test_deps,
)

py_test(
    name = "test_reload_computed_weights",
    srcs = ["test_reload_computed_weights.py"],
    deps = py_test_deps,
)

py_test(
    name = "weight_memory_saver_gpu_test",
    srcs = ["weight_memory_saver_gpu_test.py"],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

py_test(
    name = "test_model_weight_info",
    srcs = ["test_model_weight_info.py"],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

py_test(
    name = "test_inline_fp8_quant",
    srcs = ["test_inline_fp8_quant.py"],
    deps = py_test_deps,
    tags = ["rocm"],
    exec_properties = {
        'gpu': 'MI308X-ROCM7',
    },
)
