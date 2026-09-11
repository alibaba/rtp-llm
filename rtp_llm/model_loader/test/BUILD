py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "test_v41_weight_loading",
    srcs = ["test_v41_weight_loading.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"DSV41_OWNER_COMPRESSOR": "1"},
    tags = ["manual", "requires-dsv41-weights"],
)

py_test(
    name = "test_v41_draft_weight_loading",
    srcs = ["test_v41_draft_weight_loading.py"],
    deps = ["//rtp_llm:testlib"],
    env = {
        "DSV41_BLOCK32_LINEAR": "1",
        "DSV41_NATIVE_COMPACT_WRITER": "1",
    },
    tags = ["manual", "requires-dsv41-weights"],
)

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
    name = "test_per_channel_fp8_helpers",
    srcs = ["test_per_channel_fp8_helpers.py"],
    deps = py_test_deps,
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
