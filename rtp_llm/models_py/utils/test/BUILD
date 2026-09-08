py_test (
    name = "arch_test",
    srcs = ["arch_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO"],
    exec_properties = {'gpu':'RTX_5000_PRO', 'gpu_count':'1'},
)

py_test (
    name = "deepgemm_test",
    srcs = ["deepgemm_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    # Must stay on H20: the prebuilt deep_gemm_cpp .so only instantiates
    # SM90ArchSpec/SM100ArchSpec, so sm86 hits "Unsupported architecture".
    tags = ["H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'1'},
)


py_test (
    name = "debug_test",
    srcs = ["debug_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {'gpu':'A10', 'gpu_count':'1'},
)

py_test (
    name = "typed_storage_view_test",
    srcs = ["typed_storage_view_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {'gpu':'A10', 'gpu_count':'1'},
)
