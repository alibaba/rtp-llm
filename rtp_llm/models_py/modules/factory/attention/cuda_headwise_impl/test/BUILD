# Test configuration for PyHeadWise

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

# FlashInfer dependencies (包括 apache-tvm-ffi 用于 JIT 编译)
flashinfer_deps = [
    "//rtp_llm:apache-tvm-ffi",
    "//rtp_llm:flashinfer-python",
]

# The wider dependency select intentionally provides CUDA 13 analysis readiness,
# while execution remains cuda12_9-only. Add a CUDA 13 tag only after its GPU
# pool and runtime wheel combination are qualified; analysis support alone must
# not silently opt this test into a new lane.
py_test(
    name = "test_py_headwise",
    srcs = [
        "test_py_headwise.py",
    ],
    deps = py_test_deps + flashinfer_deps + select({
        "@//:using_cuda12_9_plus_x86": ["//rtp_llm:rtp-kernel"],
        "//conditions:default": [],
    }),
    tags = ["H20", "cuda12_9"],
    exec_properties = {
        "gpu": "H20",
    },
)
