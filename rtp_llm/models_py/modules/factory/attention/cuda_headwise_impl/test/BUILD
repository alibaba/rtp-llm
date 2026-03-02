# Test configuration for PyHeadWise

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

# FlashInfer dependencies (包括 apache-tvm-ffi 用于 JIT 编译)
flashinfer_deps = [
    "//rtp_llm:apache-tvm-ffi",
    "//rtp_llm:flashinfer-python",
]

# Library for attention reference implementation
py_library(
    name = "atten_test_util",
    srcs = ["atten_test_util.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:config",
    ]
)

py_test(
    name = "test_py_headwise",
    srcs = [
        "test_py_headwise.py",
    ],
    deps = py_test_deps + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": []
    }),
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)