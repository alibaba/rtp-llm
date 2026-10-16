py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "test_weight_split_storage",
    srcs = ["test_weight_split_storage.py"],
    deps = py_test_deps,
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu_count": "1"},
    tags = ["manual"],
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


py_test(
    name = "test_weight_split_storage_gpu_group",
    main = "test_weight_split_storage.py",
    srcs = ["test_weight_split_storage.py"],
    deps = py_test_deps,
    env = {"GPU_COUNT": "4"},
    exec_properties = {"gpu_count": "4"},
    tags = ["manual"],
)
