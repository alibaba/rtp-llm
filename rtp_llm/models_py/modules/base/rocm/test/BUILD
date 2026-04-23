
test_envs = {
    "DEVICE_RESERVE_MEMORY_BYTES": "512000000",  # 512MB
}

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test (
    name = "rocm_norm_test",
    srcs = ["rocm_norm_test.py"],
    deps = py_test_deps,
    timeout = "eternal",
    env = test_envs,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test (
    name = "rocm_layer_norm_test",
    srcs = ["rocm_layer_norm_test.py"],
    deps = py_test_deps,
    timeout = "eternal",
    env = test_envs,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test (
    name = "rocm_fused_add_layernorm_test",
    srcs = ["rocm_fused_add_layernorm_test.py"],
    deps = py_test_deps,
    timeout = "eternal",
    env = test_envs,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test (
    name = "rocm_general_layer_norm_test",
    srcs = ["rocm_general_layer_norm_test.py"],
    deps = py_test_deps,
    timeout = "eternal",
    env = test_envs,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test (
    name = "rocm_embedding_test",
    srcs = ["rocm_embedding_test.py"],
    deps = py_test_deps,
    timeout = "eternal",
    env = test_envs,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test (
    name = "rocm_fusedqkrmsnorm_test",
    srcs = ["rocm_fusedqkrmsnorm_test.py"],
    deps = py_test_deps,
    timeout = "eternal",
    env = test_envs,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test (
    name = "rocm_fmha_test",
    srcs = ["rocm_fmha_test.py"],
    deps = py_test_deps,
    env = test_envs,
    timeout = "eternal",
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test(
    name = "trt_allreduce_test",
    srcs = ["trt_allreduce_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py/modules/base/rocm:trt_allreduce",
    ],
    timeout = 'moderate',
    env = {"GPU_COUNT": "2"},
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7', 'gpu_count':'2'},
)

