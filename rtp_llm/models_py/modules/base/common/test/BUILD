py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

test_envs = {}

py_test (
    name = "embeding_test",
    srcs = ["embeding_test.py"],
    deps = py_test_deps,
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

py_test (
    name = "multimodal_embedding_test",
    srcs = ["multimodal_embedding_test.py"],
    deps = py_test_deps,
    env = test_envs,
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

py_test(
    name = "multimodal_copy_device_guard_test",
    srcs = ["multimodal_copy_device_guard_test.py"],
    deps = py_test_deps,
    env = {"GPU_COUNT": "2"},
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "2"},
    tags = ["L20_CU13"],
)

py_test(
    name = "multimodal_copy_device_guard_rocm_test",
    srcs = ["multimodal_copy_device_guard_test.py"],
    main = "multimodal_copy_device_guard_test.py",
    deps = py_test_deps,
    env = {"GPU_COUNT": "2"},
    exec_properties = {"gpu": "MI308X-ROCM7", "gpu_count": "2"},
    tags = ["rocm", "MI308X-ROCM7"],
)
