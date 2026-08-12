# test_attn_utils is CPU-only; kernel tests require the MI308X ROCm runner.

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "test_attn_utils",
    srcs = ["test_attn_utils.py"],
    deps = py_test_deps,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test(
    name = "test_aiter_prefill_op",
    srcs = ["test_aiter_prefill_op.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    timeout = "eternal",
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

py_test(
    name = "test_fused_qkv_transpose_v3",
    srcs = ["test_fused_qkv_transpose_v3.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    timeout = "eternal",
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

py_test(
    name = "test_aiter_decode_triton_noasm",
    srcs = ["test_aiter_decode_triton_noasm.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    timeout = "eternal",
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
)
