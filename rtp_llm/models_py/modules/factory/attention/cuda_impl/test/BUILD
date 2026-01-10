# Test configuration for PyFlashinferDecodeAttnOp

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

# Library for attention reference implementation

# Test for PyFlashinferDecodeAttnOp decode attention implementation
py_test(
    name = "test_py_flashinfer_mha_decode",
    srcs = [
        "test_py_flashinfer_mha_decode.py",
        "attention_ref.py",
        "base_attention_test.py"
    ],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

py_test(
    name = "test_xqa",
    srcs = [
        "test_xqa.py",
        "attention_ref.py",
        "base_attention_test.py"
    ],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

# TRT attention tests - organized in trt_tests/ subdirectory
py_test(
    name = "test_trt_nonpadded",
    srcs = [
        "trt_tests/test_trt_nonpadded.py",
        "trt_tests/test_trt_base.py",
        "trt_tests/trt_test_utils.py",
        "attention_ref.py",
        "base_attention_test.py"
    ],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

py_test(
    name = "test_trt_padded",
    srcs = [
        "trt_tests/test_trt_padded.py",
        "trt_tests/test_trt_base.py",
        "trt_tests/trt_test_utils.py",
        "attention_ref.py",
        "base_attention_test.py"
    ],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

py_test(
    name = "test_trt_paged_nonpadded",
    srcs = [
        "trt_tests/test_trt_paged_nonpadded.py",
        "trt_tests/test_trt_base.py",
        "trt_tests/trt_test_utils.py",
        "attention_ref.py",
        "base_attention_test.py"
    ],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

py_test(
    name = "test_trt_paged_padded",
    srcs = [
        "trt_tests/test_trt_paged_padded.py",
        "trt_tests/test_trt_base.py",
        "trt_tests/trt_test_utils.py",
        "attention_ref.py",
        "base_attention_test.py"
    ],
    deps = py_test_deps,
    tags = ["H20"],
    exec_properties = {
        'gpu': 'H20',
    },
)

# Benchmark for FlashInfer tensor cores performance comparison
py_test(
    name = "bench_flashinfer_tensor_cores",
    srcs = [
        "bench_flashinfer_tensor_cores.py",
        "bench_utils.py",
    ],
    deps = py_test_deps,
    tags = ["manual", "H20"],  # manual tag for benchmark tests
    exec_properties = {
        'gpu': 'H20',
    },
)

