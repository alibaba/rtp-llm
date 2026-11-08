py_test(
    name = "cutlass_import_path_test",
    srcs = ["cutlass_import_path_test.py"],
    deps = ["//rtp_llm/models_py:utils"],
)

py_test (
    name = "arch_test",
    srcs = ["arch_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
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
    tags = ["H20_CU13"],
    exec_properties = {
        "gpu": "H20_CU13",
        "gpu_count": "1",
    },
)

py_test (
    name = "debug_test",
    srcs = ["debug_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "L20_CU13",
        "gpu_count": "1",
    },
    tags = ["L20_CU13"],
)

py_test (
    name = "typed_storage_view_test",
    srcs = ["typed_storage_view_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "L20_CU13",
        "gpu_count": "1",
    },
    tags = ["L20_CU13"],
)

# SM120 arch cases split out of arch_test.py.
py_test(
    name = "arch_sm120_test",
    srcs = ["arch_sm120_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    visibility = ["//rtp_llm/models_py/modules/dsv4/test:__pkg__"],
)
