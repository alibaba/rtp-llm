# Test configuration for PyHeadWise

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

# FlashInfer dependencies (包括 apache-tvm-ffi 用于 JIT 编译)
flashinfer_deps = [
    "//rtp_llm:apache-tvm-ffi",
    "//rtp_llm:flashinfer-python",
]

py_test(
    name = "test_py_headwise",
    srcs = [
        "test_py_headwise.py",
    ],
    deps = py_test_deps + flashinfer_deps + select({
        "@//:using_cuda12_9_x86": ["@pip_gpu_cuda12_9_torch//rtp_kernel:pkg"],
        "//conditions:default": [],
    }),
    tags = ["H20", "cuda12_9"],
    exec_properties = {
        "gpu": "H20",
    },
)
