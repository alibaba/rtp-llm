py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "fp8_ptpc_entrypoint_test",
    srcs = ["fp8_ptpc_entrypoint_test.py"],
    deps = py_test_deps + ["//rtp_llm/models_py:new_weight_loader"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
)

py_test(
    name = "fp8_ptpc_newloader_test",
    visibility = ["//rtp_llm/test/smoke:__pkg__"],
    srcs = ["fp8_ptpc_newloader_test.py"],
    deps = py_test_deps + ["//rtp_llm/models_py:new_weight_loader"],
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_rocm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
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
