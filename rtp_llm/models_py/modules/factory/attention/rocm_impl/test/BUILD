# Tests for ROCm prefill/decode attention helpers.
#
# - test_attn_utils: geometry helpers, plain torch on CPU, no GPU needed.
# - test_aiter_prefill_op: numerical regression for AiterPrefillAttnOp; runs
#   on the ROCm CI runner because it calls aiter.flash_attn_varlen_func.

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
