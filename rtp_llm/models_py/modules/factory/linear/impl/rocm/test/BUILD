py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "f16_hot_path_dispatch_test",
    srcs = ["f16_hot_path_dispatch_test.py"],
    deps = py_test_deps,
)

py_test(
    name = "cached_f16_gemm_test",
    srcs = ["cached_f16_gemm_test.py"],
    deps = py_test_deps,
    timeout = "long",
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

py_test(
    name = "rocm_linear_test",
    srcs = ["rocm_linear_test.py"],
    deps = py_test_deps,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test(
    name = "fp8_ptpc_linear_test",
    srcs = ["fp8_ptpc_linear_test.py"],
    deps = py_test_deps,
    timeout = "eternal",
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test(
    name = "fp8_ptpc_solution_cache_test",
    srcs = ["fp8_ptpc_solution_cache_test.py"],
    deps = py_test_deps,
    timeout = "eternal",
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)
