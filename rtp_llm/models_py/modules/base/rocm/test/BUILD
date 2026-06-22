
py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test (
    name = "rocm_norm_test",
    srcs = ["rocm_norm_test.py"],
    deps = py_test_deps,
    timeout = "eternal",
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test (
    name = "rocm_layer_norm_test",
    srcs = ["rocm_layer_norm_test.py"],
    deps = py_test_deps,
    timeout = "eternal",
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test (
    name = "rocm_fused_add_layernorm_test",
    srcs = ["rocm_fused_add_layernorm_test.py"],
    deps = py_test_deps,
    timeout = "eternal",
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test (
    name = "rocm_general_layer_norm_test",
    srcs = ["rocm_general_layer_norm_test.py"],
    deps = py_test_deps,
    timeout = "eternal",
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test (
    name = "rocm_embedding_test",
    srcs = ["rocm_embedding_test.py"],
    deps = py_test_deps,
    timeout = "eternal",
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test (
    name = "rocm_fusedqkrmsnorm_test",
    srcs = ["rocm_fusedqkrmsnorm_test.py"],
    deps = py_test_deps,
    timeout = "eternal",
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test (
    name = "rocm_fmha_test",
    srcs = ["rocm_fmha_test.py"],
    deps = py_test_deps,
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

py_test(
    name = "vllm_custom_allreduce_test",
    srcs = ["vllm_custom_allreduce_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py/modules/base/rocm:vllm_custom_allreduce",
    ],
    timeout = "moderate",
    env = {"GPU_COUNT": "2"},
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7', 'gpu_count':'2'},
)

py_test(
    name = "quick_reduce_test",
    srcs = ["quick_reduce_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py/modules/base/rocm:quick_reduce",
    ],
    timeout = "moderate",
    env = {"GPU_COUNT": "2"},
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7', 'gpu_count':'2'},
)

py_test(
    name = "test_unified_allreduce_tp",
    srcs = ["test_unified_allreduce_tp.py"],
    deps = py_test_deps + [
        "//rtp_llm:testlib",
    ],
    timeout = "moderate",
    env = {"GPU_COUNT": "2"},
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7', 'gpu_count':'2'},
)

py_test(
    name = "rocm_allreduce_perf_unit_test",
    srcs = [
        "rocm_allreduce_perf_lib.py",
        "rocm_allreduce_perf_unit_test.py",
    ],
    imports = ["."],
    deps = [
        "//rtp_llm:testlib",
    ],
    timeout = "short",
)

py_test(
    name = "rocm_allreduce_perf_test",
    srcs = [
        "rocm_allreduce_perf_lib.py",
        "rocm_allreduce_perf_test.py",
    ],
    imports = ["."],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py/modules/base/rocm:quick_reduce",
        "//rtp_llm/models_py/modules/base/rocm:trt_allreduce",
        "//rtp_llm/models_py/modules/base/rocm:vllm_custom_allreduce",
    ],
    timeout = "eternal",
    env = {"GPU_COUNT": "2"},
    tags = ["manual", "rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7", "gpu_count": "2"},
)
